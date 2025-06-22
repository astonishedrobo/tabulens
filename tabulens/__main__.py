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

"""Main entry point for the Tabulens command line interface."""

import argparse
from .extractor import TableExtractor


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tabulens",
        description="Extract tables from a PDF using image-based LLM processing."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    ex =  sub.add_parser("extract", help="Extract tables from a PDF file")
    ex.add_argument("--pdf", required=True, help="Path to the PDF file")
    ex.add_argument("--llm", default="gpt:gpt-4o", help="LLM model name. Default is 'gpt:gpt-4o'.")
    ex.add_argument("--temperature", type=float, default=0.7)
    ex.add_argument("--max_tries", type=int, default=5)
    ex.add_argument("--td-model", default=None, help="Table detection model name. If not provided, uses the default model [astonishedrobo/table-detection].")
    ex.add_argument("--pages", nargs="+", type=int, default=None, help="List of pages to process e.g --pages 0 2 4. If not provided, all pages will be processed.")
    ex.add_argument("--output", default=None, help="Output directory to save the extracted tables. If not provided, output will be saved in the default './output'.")
    ex.add_argument("--verbose", action="store_true", help="Enable verbose output")
    ex.add_argument("--rate_limiter", action="store_true", help="Enable rate limiting for LLM calls")

    args = parser.parse_args()
    log_level = {"console": "INFO", "file": "DEBUG"} if args.verbose else {"console": "WARNING", "file": "DEBUG"}
    if args.cmd == "extract":
        TableExtractor(
            model_name=args.llm,
            temperature=args.temperature,
            verbose=args.verbose,
            rate_limiter=args.rate_limiter,
            yolo_params={
                "repo_id": args.td_model,
            } if args.td_model else None,
            log_level=log_level,
        ).extract_tables(
            file_path=args.pdf,
            save=True,
            max_tries=args.max_tries,
            page_idxs=args.pages,
            return_df=False,
            output_dir=args.output,
        )


if __name__ == "__main__":
    main()
