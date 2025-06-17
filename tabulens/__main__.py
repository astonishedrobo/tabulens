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
    sub = parser.add_subparsers(dest="cmd", required=True)

    ex =  sub.add_parser("extract", help="Extract tables from a PDF file")
    ex.add_argument("--pdf", required=True, help="Path to the PDF file")
    ex.add_argument("--model", default="gpt:gpt-4o-mini")
    ex.add_argument("--temperature", type=float, default=0.7)
    ex.add_argument("--max_tries", type=int, default=3)
    ex.add_argument("--verbose", action="store_true")
    ex.add_argument("--rate_limiter", action="store_true", help="Enable rate limiting for LLM calls")

    args = parser.parse_args()

    if args.cmd == "extract":
        TableExtractor(
            model_name=args.model,
            temperature=args.temperature,
            verbose=args.verbose,
            rate_limiter=args.rate_limiter,
        ).extract_tables(
            file_path=args.pdf,
            save=True,
            max_tries=args.max_tries,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
