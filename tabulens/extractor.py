import cv2
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
from tqdm import tqdm
from io import StringIO
import base64
import os
import re

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.retry import RunnableRetry
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


class TableExtractor:
    def __init__(self, model_name: str = 'gpt:gpt-4o-mini', temperature: int = 0.7, verbose: bool = False, rate_limiter: bool = False, rate_limiter_params: dict = None):
        self.temperature = temperature
        self.model_name = model_name
        self.messages = [
            SystemMessage(content="""
            Extract and intelligently restructure the table from the image into a meaningful, flat Markdown table format following these requirements:

            ANALYSIS PHASE:
            - Identify the table's hierarchical structure and parent-child relationships
            - Determine which cells are parent categories and which are subcategories
            - Map out the complete hierarchy to understand data groupings

            RESTRUCTURING STRATEGY:
            - Create separate columns for each level of hierarchy (e.g., "Main Category", "Sub Category")
            - CRITICAL: Replicate parent category values for ALL child rows that belong to that parent
            - If a parent category has multiple subcategories, the parent name must appear in every row for those subcategories
            - Create explicit columns for each hierarchical level rather than leaving parent cells blank

            PARENT CELL REPLICATION RULES:
            - When a parent category spans multiple child rows, repeat the parent category name in each child row
            - Do not leave parent category cells blank - every row must be a complete, standalone record
            - If parent categories have values (like totals), create separate rows for parent-level data
            - Ensure no row depends on previous rows to understand its category membership

            COLUMN DESIGN:
            - Structure: [Main Category] | [Sub Category] | [Year1] | [Year2] | etc.
            - Every row must have values in all category columns (no blank category cells)
            - Parent categories should be repeated verbatim for all their children
            - Use consistent naming and avoid abbreviations in category columns

            DATA HANDLING:
            - Preserve all numerical values exactly, including parentheses for negative values
            - Maintain original formatting (commas, decimals, brackets)
            - If parent rows contain summary data, include them as separate rows
            - Each row must be independently interpretable

            OUTPUT FORMAT:
            - Standard Markdown table with proper alignment
            - Every row must be a complete record with all category information filled
            - Must be directly convertible to CSV for analysis
            - Structure should enable grouping and filtering by any category level

            RESPONSE RULES:
            - Output ONLY the restructured markdown table
            - No explanations, code blocks, or additional text
            - If extraction/restructuring is impossible, respond with exactly: 'NP'

            Remember: Every row must contain the full categorical path - never leave parent category cells blank when they apply to child rows.
            Think: How can this table be redesigned to maximize analytical value while preserving all original information?
            """
            ),
        ]
        self.verbose = verbose

        if not isinstance(rate_limiter_params, dict) and rate_limiter_params is not None:
            raise TypeError("rate_limiter_params must be a dictionary or None")
        self.rate_limiter_params = {**{"requests_per_second": 0.5, "check_every_n_seconds": 0.1, "max_bucket_size": 1}, **(rate_limiter_params or {})}

        if rate_limiter:
            self.rate_limiter = InMemoryRateLimiter(**self.rate_limiter_params)
        else:
            self.rate_limiter = None

        self.__init_llm_client()

    def __init_llm_client(self):
        """
        Initializes the LLM client based on the model name.
        """
        if self.model_name.startswith("gemini:"):
            model_name = self.model_name.replace("gemini:", "", 1)
            try:
                self.client = ChatGoogleGenerativeAI(model=model_name, temperature=self.temperature, thinking_budget=0, rate_limiter=self.rate_limiter)
            except:
                self.client = ChatGoogleGenerativeAI(model=model_name, rate_limiter=self.rate_limiter)
        elif self.model_name.startswith("gpt:"):
            model_name = self.model_name.replace("gpt:", "", 1)
            try:
                self.client = ChatOpenAI(model=model_name, temperature=self.temperature, rate_limiter=self.rate_limiter)
            except:
                self.client = ChatOpenAI(model=model_name, rate_limiter=self.rate_limiter)
        elif self.model_name.startswith("groq:"):
            model_name = self.model_name.replace("groq:", "", 1)
            try:
                self.client = ChatGroq(model=model_name, temperature=self.temperature, rate_limiter=self.rate_limiter)
            except:
                self.client = ChatGroq(model=model_name, rate_limiter=self.rate_limiter)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    def __extract_tables_images(self, file_path: str, dpi: int = 300, min_table_area: int = 5000, pad: int = 5) -> list[np.ndarray]:
        """
        Extracts tables from each page of the given PDF.

        Args:
            file_path (str): Path to the PDF file.
            dpi (int): Resolution for rendering PDF pages.
            min_table_area (int): Minimum contour area to qualify as a table.
            pad (int): Extra pixels to pad around each detected table.

        Returns:
            list[np.ndarray]: List of cropped table images (BGR arrays).
        """
        if self.verbose:
            print("Starting: Extracting Table Images")

        pages = convert_from_path(file_path, dpi=dpi)

        tables = []
        for page in tqdm(pages):
            img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            # Morphological line detection
            horiz_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (img.shape[1] // 15, 1)
            )
            vert_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (1, img.shape[0] // 15)
            )
            horiz_lines = cv2.morphologyEx(
                bw, cv2.MORPH_OPEN, horiz_kernel, iterations=2
            )
            vert_lines = cv2.morphologyEx(
                bw, cv2.MORPH_OPEN, vert_kernel, iterations=2
            )

            # Combine and Close tiny gaps
            mask = cv2.add(horiz_lines, vert_lines)
            close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                        close_kernel, iterations=1)

            contours, _ = cv2.findContours(
                mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Crop detected table regions
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w * h >= min_table_area:
                    x0 = max(x - pad, 0)
                    y0 = max(y - pad, 0)
                    x1 = min(x + w + pad, img.shape[1])
                    y1 = min(y + h + pad, img.shape[0])
                    table_img = img[y0:y1, x0:x1]
                    tables.append(table_img)

        if self.verbose:
            print("Done: Extracting Table Images")

        return tables
    
    def __call_llm(self, prompt: HumanMessage) -> str:
        """
        Extracts table in Markdown format from image using LLM.
        
        Args:
            prompt: Prompt to be sent to the LLM.
        
        Returns:
            str: Extracted table in Markdown format.
        """
        
        messages = self.messages + [prompt]
        response = self.client.invoke(messages)

        if response and response.content != 'NP':
            if self.verbose:
                print(response.content)
            return response.content
        else:
            raise ValueError("Couldn't extract CSV from Image")
    
    def __md_table_to_csv(self, md_table: str) -> str:
        """
        Convert a Markdown table (given as one big string) into a CSV string.
        Assumes a well-formed table with header, separator, and rows.
        """
        lines = [line for line in md_table.splitlines() if line.strip()]
        
        # Find the separator row (---|---|--- style)
        sep_idx = next(
            (i for i, line in enumerate(lines)
            if re.match(r'^\s*\|?[:\- ]+\|[:\- \|]+$', line)),
            None
        )
        if sep_idx is None:
            raise ValueError("No Markdown table separator found")
        
        header_line = lines[sep_idx - 1]
        data_lines  = lines[sep_idx + 1:]
        
        def parse_row(line: str) -> list[str]:
            cells = [cell.strip() for cell in line.strip().split('|')]
            return cells[1:-1]  # drop the empty edge cells
        
        # Build CSV lines
        csv_rows = []
        hdr = parse_row(header_line)
        csv_rows.append(','.join(hdr))
        
        for row in data_lines:
            if not row.strip().startswith('|'):
                break
            cells = parse_row(row)
            # quote any cell containing commas
            safe_cells = [f'"{c}"' if ',' in c else c for c in cells]
            csv_rows.append(','.join(safe_cells))
        
        return '\n'.join(csv_rows)
    
    
    def __encode_image_to_base64(self, image: np.ndarray) -> str:
        """
        Encodes an image to a base64 string.
        
        Args:
            image (np.ndarray): Image to be encoded.

        Returns:
            str: Base64 encoded string of the image.
        """
        _, buffer = cv2.imencode('.jpg', image)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        return base64_image
    
    def __update_messages(self, latest_schema: str):
        """
        Updates the messages list with the schema of the latest generated DataFrame/CSV

        Args:
            latest_schema (str): Schema of the latest generated DataFrame/CSV
        """
        self.messages = self.messages[:1]
        new_human_message = HumanMessage(
            content=f"Below is the DataFrame schema of the last extracted CSV. You may use this information if necesssay (e.g maintain uniformity in naming convention or anything) or disregard it. \n\n{latest_schema}"
        )
        self.messages.append(new_human_message)

    def __extract_df_from_image(self, table_image: np.ndarray) -> pd.DataFrame:
        """
        Extracts a DataFrame from the given table image using LLM.
        
        Args:
            table_image (np.ndarray): Image of the table to be processed.

        Returns:
            pd.DataFrame: Extracted DataFrame.
        """
        prompt = HumanMessage(
            content=[
                {"type": "text", "text": "Extract the table from this image in valid and complete Markdown Table format."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{self.__encode_image_to_base64(table_image)}"},
                },
            ]
        )
        output = self.__call_llm(prompt)
        output = self.__md_table_to_csv(output)

        try:
            df = pd.read_csv(StringIO(output))
        except Exception as e:
            raise ValueError(f"Error converting image to DataFrame: {e}")
        
        return df

    def extract_tables(self, file_path: str, save: bool = False, max_tries: int = 3, verbose: bool = False) -> list[pd.DataFrame]:
        """
        Extracts tables from the given PDF file.
        
        Args:
            file_path (str): Path to the PDF file.
            save (bool): Whether or not to save the save the tables as CSV
            max_tries (int): No. of times the program should attempt to extract the table in valid CSV format
            verbose (bool): Whether or not to print the intermediate outputs or messages (errors or progress messages).

        Returns:
            list[pd.DataFrame]: List of DataFrames containing extracted tables.
        """
        self.verbose = verbose
        tables_images = self.__extract_tables_images(file_path)

        runnable = RunnableLambda(lambda img: self.__extract_df_from_image(img))
        runnable_retry = RunnableRetry(
            bound = runnable,
            max_attempt_number = max_tries,
        )
        dfs = []

        for i, table_image in tqdm(enumerate(tables_images), total=len(tables_images)):
            try:
                
                df = runnable_retry.invoke(table_image)
                dfs.append(df)
                if not df.empty:
                    self.__update_messages(latest_schema=df.dtypes)

            except Exception as e:
                if self.verbose:
                    print(f"Error Converting Table {i} After Max Attempts ({max_tries}): {e}")
                dfs.append(None)
            

        if save:
            filename = os.path.basename(file_path).split(".")[0]
            cwd = os.getcwd()
            os.makedirs(os.path.join(cwd, "output", filename), exist_ok=True)

            for i, df in enumerate(dfs):
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df.to_csv(os.path.join(cwd, "output", filename, f"table_{i+1}.csv"), index=False)
        
        return dfs