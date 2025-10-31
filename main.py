from __future__ import annotations

import argparse
import json
import logging
import pathlib
from typing import Optional

try:
    from .ingest import (
        ColumnMapping,
        load_column_mapping_json,
        normalize_statement_file,
        summarize_transactions,
    )
except ImportError:  # pragma: no cover - fallback when run as a script
    from ingest import (  # type: ignore
        ColumnMapping,
        load_column_mapping_json,
        normalize_statement_file,
        summarize_transactions,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize historical banking or brokerage statements into the"
            " canonical schema used by the personal finance agent."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Normalize a statement file into Date/Amount/Currency/TransactionType/Description",
    )
    ingest_parser.add_argument("--input", required=True, help="Path to the statement file (CSV or Excel).")
    ingest_parser.add_argument(
        "--output",
        help="Optional output CSV path. If omitted, the normalized data is printed to stdout as JSON.",
    )
    ingest_parser.add_argument(
        "--currency",
        help="Fallback ISO currency code to use when the statement does not contain one.",
    )
    ingest_parser.add_argument(
        "--column-mapping",
        help="Path to a JSON file specifying column names (date, amount, credit, debit, currency, description, transaction_type).",
    )

    # Allow quick column overrides without providing a JSON file
    ingest_parser.add_argument("--date-column")
    ingest_parser.add_argument("--amount-column")
    ingest_parser.add_argument("--credit-column")
    ingest_parser.add_argument("--debit-column")
    ingest_parser.add_argument("--currency-column")
    ingest_parser.add_argument("--description-column")
    ingest_parser.add_argument("--transaction-type-column")

    ingest_parser.add_argument(
        "--summary",
        action="store_true",
        help="Print aggregate inflow/outflow summary after normalization.",
    )

    ingest_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity for diagnostics.",
    )

    return parser


def _configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level, logging.INFO))


def _command_ingest(args: argparse.Namespace) -> None:
    _configure_logging(args.log_level)

    input_path = pathlib.Path(args.input).expanduser().resolve()
    output_path: Optional[pathlib.Path] = (
        pathlib.Path(args.output).expanduser().resolve() if args.output else None
    )

    explicit_mapping = load_column_mapping_json(args.column_mapping)

    mapping = ColumnMapping.from_user_input(
        explicit_mapping,
        date=args.date_column,
        amount=args.amount_column,
        credit=args.credit_column,
        debit=args.debit_column,
        currency=args.currency_column,
        description=args.description_column,
        transaction_type=args.transaction_type_column,
    )

    normalized = normalize_statement_file(input_path, mapping, args.currency)

    if output_path:
        normalized.to_csv(output_path, index=False)
        print(f"Normalized data written to {output_path}")
    else:
        print(normalized.to_json(orient="records", date_format="iso"))

    if args.summary:
        summary = summarize_transactions(normalized)
        print(json.dumps(summary, indent=2))


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        _command_ingest(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
