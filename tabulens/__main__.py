"""
CLI for tabulex:  `tabulex --pdf my.pdf --model gpt:gpt-4o-mini`
"""
import argparse
from .extractor import TableExtractor


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tabulens",
        description="Extract tables from a PDF using image-based LLM processing."
    )
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--model", default="gpt:gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tries", type=int, default=3)
    parser.add_argument("--log", action="store_true")

    args = parser.parse_args()

    TableExtractor(
        model_name=args.model,
        temperature=args.temperature,
        print_logs=args.log,
    ).extract_tables(
        file_path=args.pdf,
        save=True,
        max_tries=args.max_tries,
        print_logs=args.log,
    )


if __name__ == "__main__":
    main()
