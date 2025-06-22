# Copyright (C) 2025 Soumyajit Basu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
from tqdm import tqdm
from io import StringIO
import base64
import os
import re
from typing import Optional, List
from pydantic import BaseModel, Field
from pathlib import Path
from .utils.logger import get_logger

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.retry import RunnableRetry
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from datasets import load_dataset

class SchemaValidation(BaseModel):
    answer: str = Field(..., description="Either YES or NO")
    reasoning: str = Field(..., description="Explanation for the answer")

class TableExtractor:
    def __init__(
        self, 
        model_name: str = 'gpt:gpt-4o', 
        temperature: int = 0.7, 
        show_progress: bool = True,
        verbose: bool = False, 
        rate_limiter: bool = False, 
        rate_limiter_params: dict = None, 
        yolo_params: dict = None,
        output_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        log_level: Optional[dict] = {"console": "INFO", "file": "DEBUG"},
    ):
        """
        Initializes the TableExtractor with the specified parameters.
        Args:
            model_name (str): Name of the LLM model to use. Defaults to 'gpt:gpt-4o-mini'.
            temperature (float): Temperature for the LLM model. Defaults to 0.7.
            show_progress (bool): Whether to show progress bars during extraction. Defaults to True.
            verbose (bool): Whether to print verbose output. Defaults to False.
            rate_limiter (bool): Whether to enable rate limiting for LLM calls. Defaults to False.
            rate_limiter_params (dict): Parameters for the rate limiter. Defaults to None.
            yolo_params (dict): Parameters for the YOLO table detection model. Defaults to None.
            output_dir (Optional[str]): Directory to save the extracted tables. If None, uses default './output'.
            log_dir (Optional[str]): Directory to save logs. If None, uses default './logs'.
            log_level (Optional[dict]): Logging levels for console and file handlers. Defaults to {"console": "INFO", "file": "INFO"}.
        """
        self._temperature = temperature
        self._model_name = model_name
        self._show_progress = show_progress or True
        self._verbose = verbose or False
        self._output_dir = Path(output_dir) if output_dir else Path(os.path.join(os.getcwd(), "output"))
        self._log_dir = Path(log_dir) if log_dir else Path(os.path.join(os.getcwd(), "logs"))
        self._logger = get_logger(
            name=f"{self.__class__.__name__}.{id(self)}",
            log_dir=self._log_dir,
            log_level=log_level,
            verbose=self._verbose,
            filehandler=True
        )
        self.__messages = []

        if not isinstance(rate_limiter_params, dict) and rate_limiter_params is not None:
            raise TypeError("rate_limiter_params must be a dictionary or None")
        self._rate_limiter_params = {**{"requests_per_second": 0.5, "check_every_n_seconds": 0.1, "max_bucket_size": 1}, **(rate_limiter_params or {})}

        if rate_limiter:
            self._rate_limiter = InMemoryRateLimiter(**self._rate_limiter_params)
        else:
            self._rate_limiter = None

        self.__init_llm_client()
        self.__running_error_logs = []
        self._client_structured_out = self._client.with_structured_output(SchemaValidation)

        if not isinstance(yolo_params, dict) and yolo_params is not None:
            raise TypeError("yolo_params must be a dictionary or None")
        self._yolo_params = {
            "repo_id": "astonishedrobo/table-detection", 
            "load_from_hf": True,
            "filename": "best.pt",
            "conf": 0.25, 
            "iou": 0.55, 
            "max_det": 300, 
            "agnostic_nms": False,
            "batch_size": 4,
            "ignored_labels": ["No-Lines"],
            **(yolo_params or {})
        }
        self.__load_yolo_model()
        self.__set_ignored_labels()

        self.__load_system_prompts()
    
    def __load_system_prompts(self):
        """
        Loads system prompts from Hugging Face Hub.
        """
        dataset_path = hf_hub_download(
            repo_id="astonishedrobo/tabulens-prompts",
            repo_type="dataset", 
            filename="standard/prompts_standard.jsonl",
            revision="standard"
        )
        prompts = load_dataset("json", data_files={"train": "/home/soumyajit/Documents/pdf-extract/dev-tabulens/tabulens/prompts/prompts_standard.jsonl"}, split="train")
        self.__system_messages = {row["node"]: row["system_message"] for row in prompts}        

    def __load_yolo_model(self):
        """
        Loads the YOLO model for table detection.
        """
        if self._yolo_params.get("load_from_hf", True):
            weights_path = hf_hub_download(
                repo_id=self._yolo_params.get("repo_id"), 
                filename=self._yolo_params.get("filename")
            )
            self._yolo = YOLO(weights_path)
        else:
            self._yolo = YOLO(self._yolo_params.get("filename"))

    def __init_llm_client(self):
        """
        Initializes the LLM client based on the model name.
        """
        if self._model_name.startswith("gemini:"):
            model_name = self._model_name.replace("gemini:", "", 1)
            try:
                self._client = ChatGoogleGenerativeAI(model=model_name, temperature=self._temperature, thinking_budget=0, rate_limiter=self._rate_limiter)
                _ = self._client.invoke("Hello, world!")
            except:
                self._client = ChatGoogleGenerativeAI(model=model_name, rate_limiter=self._rate_limiter)
        elif self._model_name.startswith("gpt:"):
            model_name = self._model_name.replace("gpt:", "", 1)
            try:
                self._client = ChatOpenAI(model=model_name, temperature=self._temperature, rate_limiter=self._rate_limiter)
                _ = self._client.invoke("Hello, world!")
            except:
                self._client = ChatOpenAI(model=model_name, rate_limiter=self._rate_limiter)
        elif self._model_name.startswith("groq:"):
            model_name = self._model_name.replace("groq:", "", 1)
            try:
                self._client = ChatGroq(model=model_name, temperature=self._temperature, rate_limiter=self._rate_limiter)
                _ = self._client.invoke("Hello, world!")
            except:
                self._client = ChatGroq(model=model_name, rate_limiter=self._rate_limiter)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    def __set_ignored_labels(self):
        """
        Updates and Validates the list of ignored labels for YOLO detections.
        """
        all_labels = set({label: id for id, label in self._yolo.names.items()})
        user_ignored = set(self._yolo_params.get("ignored_labels", []))
        invalid = user_ignored - all_labels
        if invalid:
            self._logger.warning(f"Warning: These ignored_labels not in model.names and will be skipped: {sorted(invalid)}")
        self._ignored_labels = user_ignored & all_labels

    def __extract_sorted_tables_from_pages(
        self, 
        page: np.ndarray, 
        result, 
        pad: int = 5
    ) -> list[np.ndarray]:
        """
        Given a single page image and its YOLO 'result', returns a list of
        cropped table images (BGR) sorted by their top-to-bottom order.

        Args:
            page (np.ndarray): The full page image (HxWx3).
            result: The YOLO result for that page.
            pad (int): Number of pixels to pad around each detected box.

        Returns:
            List[np.ndarray]: Cropped table regions in top-to-bottom order.
        """
        h, w = page.shape[:2]

        boxes = []
        # Collect padded box coords
        for xyxy, cls_id in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist()):
            label = self._yolo.names[int(cls_id)]
            if label in self._ignored_labels:
                continue

            x1, y1, x2, y2 = map(int, xyxy)
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            boxes.append((x1, y1, x2, y2))

        # Sort by top-to-bottom order
        boxes.sort(key=lambda b: b[1])

        # Crop and return only non‐empty regions
        tables = []
        for x1, y1, x2, y2 in boxes:
            crop = page[y1:y2, x1:x2]
            if crop.size:
                tables.append(crop)
        return tables
    
    def __extract_tables_images(
        self, 
        file_path: str, 
        page_idxs: Optional[List[int]] = None,
        dpi: int = 200, 
        pad: int = 5
    ) -> list[np.ndarray]:
        """
        Extracts tables from each page of the given PDF.

        Args:
            file_path (str): Path to the PDF file.
            page_idxs (Optional[List[int]]): List of page numbers (zero indexed) to extract tables from. If None, extracts from all pages.
            dpi (int): Resolution for rendering PDF pages.
            min_table_area (int): Minimum contour area to qualify as a table.
            pad (int): Extra pixels to pad around each detected table.

        Returns:
            list[np.ndarray]: List of cropped table images (BGR arrays).
        """
        pages = convert_from_path(file_path, dpi=dpi)
        if pages and isinstance(pages, list):
            pages = [pages[int(i)] for i in page_idxs if int(i) < len(pages)]
        pages = [np.array(page) for page in pages]
        tables = []
        
        results = self._yolo.predict(
            source=pages,
            conf=self._yolo_params.get("conf"),           # NMS confidence
            iou=self._yolo_params.get("iou"),             # NMS IoU
            max_det=self._yolo_params.get("max_det"),     # maximum detections per image
            agnostic_nms=self._yolo_params.get("agnostic_nms"),  # agnostic NMS
            batch=self._yolo_params.get("batch_size"),  # batch size
            verbose=False,
        )

        for page_idx, result in tqdm(enumerate(results), total=len(results), desc="Extracting Table Images", disable=not self._show_progress):
            tables.extend(
                self.__extract_sorted_tables_from_pages(
                    page=pages[page_idx],
                    result=result,
                    pad=pad
                )
            )

        return tables
    
    def __planner(
        self, 
        prompt: HumanMessage
    ) -> AIMessage:
        """
        Extracts table in Markdown format from image using LLM.
        
        Args:
            prompt: Prompt to be sent to the LLM.
        
        Returns:
            str: Extracted table in Markdown format.
        """
        self._logger.info("Generating Plan for Table Extraction...")
        messages = [
            SystemMessage(self.__system_messages.get("planner")),
            *self.__running_error_logs,
            prompt,
            *self.__messages
        ]
        
        response = self._client.invoke(messages)

        if not response or not response.content:
            self._logger.debug("Planner Response: None or Empty")
            raise ValueError("Planner Couldn't Generate a Plan!")
        
        return response
        
    def __extractor(
        self, 
        prompt: HumanMessage, 
        plan: AIMessage
    ) -> str:
        """
        Extracts table in Markdown format from image using LLM based on the planner's output.
        
        Args:
            prompt: Prompt to be sent to the LLM.
        
        Returns:
            str: Extracted table in Markdown format.
        """
        self._logger.info("Extracting Table from Image (from Plan)...")
        messages = [
            SystemMessage(self.__system_messages.get("extractor")),
            plan,
            prompt,
        ]

        response = self._client.invoke(messages)

        if not response or not response.content:
            raise ValueError("Extractor Couldn't Extract the Table Successfully!")
        self._logger.debug(f"Extracted Table: {response.content!r}")
            
        return response.content
    
    def __md_table_to_csv(
        self, 
        md_table: str
    ) -> str:
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
   
    def __encode_image_to_base64(
        self, 
        image: np.ndarray
    ) -> str:
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
    
    def __update_messages(
        self, 
        latest_schema: str
    ):
        """
        Updates the messages list with the schema of the latest generated DataFrame/CSV

        Args:
            latest_schema (str): Schema of the latest generated DataFrame/CSV
        """
        new_human_message = HumanMessage(
            content=f"Below is the DataFrame schema of the last extracted CSV. If the new table has same headers you should use this information (e.g maintain uniformity in naming convention or anything), otherwise disregard it. \n\n{latest_schema}"
        )
        self.__messages = [new_human_message]

    def __ocr(
        self,
        table_image: np.ndarray
    ) -> str:
        """ Extracts text from the given table image in Markdown format"""
        prompt = HumanMessage(
            content=[
                {"type": "text", "text": self.__system_messages.get("validator_ocr")},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{self.__encode_image_to_base64(table_image)}"},
                },
            ]
        )
        response = self._client.invoke([prompt])
        if not response or not response.content:
            raise ValueError("OCR Couldn't Extract the Table Successfully!")
        return response.content.strip()

    def __validate_llm_extraction(
        self, 
        extracted_table: str, 
        table_image: np.ndarray
    ) -> bool:
        """
        Validates if the LLM has preserved all original data values in the extracted table.
        
        Args:
            extracted_table (str): The extracted table in Markdown format.
            table_image (np.ndarray): Image of the table to be validated.

        Returns:
            bool: True if all original data values are preserved, False otherwise.
        """
        self._logger.info("Validating LLM Extraction...")
        ocr = self.__ocr(table_image)
        prompt = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": (
                        f"Here is the original table: \n\n{ocr}\n\n"
                        f"Here is the extracted table in Markdown format: \n\n{extracted_table}\n\n"
                        "You have to validate if the extracted table has preserved all original data values. "
                    )
                }
            ]
        )
        messages = [
            SystemMessage(self.__system_messages.get("validator")),
            prompt,
        ]
        
        response = self._client_structured_out.invoke(messages)
        answer = response.answer.upper()
        reasoning = response.reasoning

        if answer != "YES":
            self.__running_error_logs = [
                HumanMessage(content=f"You have to extract the table again. The last table extraction was not acceptable because: {reasoning}")
            ]
            self._logger.debug(f"Validator (Wrong Extraction): {answer} --- {reasoning}")
            return False
        
        return True
        

    def __extract_df_from_image(
        self, 
        table_image: np.ndarray
    ) -> pd.DataFrame:
        """
        Extracts a DataFrame from the given table image using LLM.
        
        Args:
            table_image (np.ndarray): Image of the table to be processed.

        Returns:
            pd.DataFrame: Extracted DataFrame.
        """
        prompt = HumanMessage(
            content=[
                {"type": "text", "text": f"Extract the table from this image in valid and complete Markdown Table format. {self.__running_error_logs if self.__running_error_logs else ''}"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{self.__encode_image_to_base64(table_image)}"},
                },
            ]
        )
        plan = self.__planner(prompt=prompt)
        table_md = self.__extractor(prompt=prompt, plan=plan)

        if not self.__validate_llm_extraction(table_md, table_image):
            raise ValueError("LLM did not preserve all original data values in the extracted table.")
        
        output = self.__md_table_to_csv(table_md)

        try:
            df = pd.read_csv(StringIO(output))
        except Exception as e:
            self._logger.debug(f"Error converting image to DataFrame: {e}")
            raise ValueError(f"Error converting image to DataFrame: {e}")
        
        return df

    def extract_tables(
        self, 
        file_path: str, 
        save: bool = False, 
        max_tries: int = 3, 
        return_df: bool = True, 
        output_dir: Optional[str] = None,
        page_idxs: Optional[List[int]] = None
    ) -> Optional[List[pd.DataFrame]]:
        """
        Extracts tables from the given PDF file.
        
        Args:
            file_path (str): Path to the PDF file.
            save (bool): Whether or not to save the save the tables as CSV
            max_tries (int): No. of times the program should attempt to extract the table in valid CSV format
            return_df (bool): Whether to return the extracted DataFrames or not. [Default: True]
            output_dir (Optional[str]): Directory to save the extracted tables. If None, saves in the default './output' directory.
            page_idxs (Optional[List[int]]): List of page numbers (zero indexed) to extract tables from. If None, extracts from all pages.

        Returns:
            Optional[List[pd.DataFrame]]: List of DataFrames containing extracted tables if 'return_df' is True, otherwise None.
        """
        if save:
            # Set up output directory
            filename = Path(file_path).stem
            cwd = os.getcwd()
            output_dir = output_dir or self._output_dir or Path(os.path.join(cwd, "output"))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            if not os.path.exists(os.path.join(output_dir, filename)):
                os.makedirs(os.path.join(output_dir, filename), exist_ok=True)
        
        # Extract the table images from the PDF file        
        tables_images = self.__extract_tables_images(file_path=file_path, page_idxs=page_idxs)

        # Extract DataFrames/CSVs from the table images
        runnable = RunnableLambda(lambda img: self.__extract_df_from_image(img))
        runnable_retry = RunnableRetry(
            bound = runnable,
            max_attempt_number = max_tries,
        )
        dfs = []

        for i, table_image in tqdm(enumerate(tables_images), total=len(tables_images), desc="Extracting Tables", disable=not self._show_progress):
            try:
                df = runnable_retry.invoke(table_image)
                dfs.append(df)
                if not df.empty:
                    self.__update_messages(latest_schema=df.dtypes)
                    if save:
                        df.to_csv(os.path.join(output_dir, filename, f"table_{i+1}.csv"), index=False)
                        self._logger.info(f"Table {i+1} saved at {os.path.join(cwd, 'output', filename, f'table_{i+1}.csv')}")
            except Exception as e:
                self._logger.exception(f"Error Converting Table {i} After Max Attempts ({max_tries}): {e}")
                dfs.append(None)
            
            self.__running_error_logs = [] 
        
        empty_dfs = len([df for df in dfs if df is None])
        if empty_dfs != 0:
            self._logger.warning(f"Warning: Some tables could not be extracted after max attempts ({max_tries}). Check the logs for details.")

        return dfs if return_df else None