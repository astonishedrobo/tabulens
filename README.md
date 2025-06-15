# Tabulens

**Tabulens** is a Python package that intelligently extracts and restructures tables from PDF files using advanced computer vision and Large Language Models (LLMs). It automatically detects table structures, manages complex hierarchical tables, and exports data into structured formats like CSV and pandas DataFrames.

## Features

* 🔍 **Automatic Table Detection**: Uses computer vision to identify table regions.
* 🧠 **Intelligent Restructuring**: Leverages LLMs to understand and restructure hierarchical tables.
* 📊 **Multiple Output Formats**: Supports CSV and pandas DataFrame outputs.
* 🎯 **High Accuracy**: Combines computer vision preprocessing with LLM analysis for robust extraction.
* 🔧 **Flexible Models**: Supports both OpenAI GPT and Google Gemini models.
* 📝 **Hierarchy Preservation**: Flattens nested tables while maintaining parent-child relationships.
* 🚀 **Easy to Use**: Simple API and command-line interface.

## Installation

From PyPI:

```bash
pip install tabulens
```

Or directly from GitHub:

```bash
pip install git+https://github.com/astonishedrobo/tabulens.git
```

## Quick Start

### Python API

```python
from tabulens import TableExtractor

extractor = TableExtractor(
    model_name='gpt:gpt-4o-mini', # gemini:gemini-2.0-flash
    temperature=0.7
)

dfs = extractor.extract_tables(
    file_path='path/to/document.pdf',
    save=True,
    max_tries=3,
    print_logs=True
)

for i, df in enumerate(dataframes):
    if df is not None:
        print(f"Table {i+1}")
        print(df.head())
```

### Command Line Interface

To extrach tables:

```bash
# OpenAI 
tabulens extract --pdf path/to/document.pdf --model gpt:gpt-4o-mini --temperature 0.7 --max_tries 3 --log

# Gemini
tabulens extract --pdf path/to/document.pdf --model gemini:gemini-2.0-flash --temperature 0.7 --max_tries 3 --log
```

### CLI Options

* `--pdf`: Path to the PDF file (required)
* `--model`: Model name (`gpt:gpt-4o-mini`, `gemini:gemini-2.0-flash`, `gpt:gpt-4o`, `gemini:gemini-2.5-flash-preview-05-20`, etc.) [default: `gpt:gpt-4o-mini`]. For OpenAI models, use the prefix `gpt:`, and for Gemini models, use the prefix `gemini:`. **(⚠️ Make sure to select models that support image inputs. You can use any of the mentioned examples for convenience.)**
* `--temperature`: Generation temperature (0.0-1.0) [default: 0.7]
* `--max_tries`: Maximum retries per table extraction [default: 3] [Increase this value to enhance accuracy, as more attempts allow the system additional opportunities to correctly extract tables.]
* `--log`: Print detailed logs

### Environment Variable Setup

Before running the program, set the required API environment variables.

For CLI usage:

```bash
export OPENAI_API_KEY=<your_openai_api_key>
export GOOGLE_API_KEY=<your_google_api_key>
```

For Python API usage, load environment variables using `python-dotenv`:

```python
from dotenv import load_dotenv
load_dotenv("path/to/.env")
```

## Credits

Tabulens depends on these excellent open-source projects:

* [LangChain](https://github.com/langchain-ai/langchain)
* [OpenCV](https://github.com/opencv/opencv-python)
* [NumPy](https://github.com/numpy/numpy)
* [Pandas](https://github.com/pandas-dev/pandas)
* [pdf2image](https://github.com/Belval/pdf2image)
* [tqdm](https://github.com/tqdm/tqdm)

## License

This project is licensed under the [MIT License](LICENSE).