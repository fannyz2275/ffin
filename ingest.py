"""Utilities for normalizing banking and brokerage statements.

This module currently focuses on the initial feature of the personal finance
agent: accepting raw statement exports (CSV/Excel) or routing other formats through
an OpenAI extraction step before transforming them into a
normalized schema with the following columns:

- Date
- Amount
- Currency
- TransactionType
- Description

The normalization logic attempts to be pragmatic and flexible by allowing
explicit column mappings while also providing light heuristics for common
column names.
"""

from __future__ import annotations

import enum
import json
import logging
import os
import pathlib
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import pandas as pd


OPENAI_MODEL_ENV = "FFIN_OPENAI_MODEL"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
OPENAI_KEY_ENV = "OPENAI_API_KEY"

LOGGER = logging.getLogger(__name__)


class TransactionType(str, enum.Enum):
    """Supported transaction directions."""

    INFLOW = "inflow"
    OUTFLOW = "outflow"


@dataclass(slots=True)
class ColumnMapping:
    """Describe how to map the raw statement columns to the normalized schema."""

    date: Optional[str] = None
    amount: Optional[str] = None
    credit: Optional[str] = None
    debit: Optional[str] = None
    transaction_type: Optional[str] = None
    currency: Optional[str] = None
    description: Optional[str] = None

    @classmethod
    def from_user_input(
        cls,
        mapping: Optional[Mapping[str, str]] = None,
        **overrides: Optional[str],
    ) -> "ColumnMapping":
        """Build a column mapping from a dictionary and/or keyword overrides."""

        mapping = dict(mapping or {})
        mapping.update({k: v for k, v in overrides.items() if v})

        return cls(
            date=mapping.get("date"),
            amount=mapping.get("amount"),
            credit=mapping.get("credit"),
            debit=mapping.get("debit"),
            transaction_type=mapping.get("transaction_type"),
            currency=mapping.get("currency"),
            description=mapping.get("description"),
        )


HEURISTIC_CANDIDATES = {
    "date": ("date", "transaction_date", "posted_date", "value date"),
    "amount": (
        "amount",
        "transaction_amount",
        "amt",
        "value",
        "usd amount",
    ),
    "credit": ("credit", "inflow", "deposit", "credit amount"),
    "debit": ("debit", "outflow", "withdrawal", "debit amount"),
    "transaction_type": ("type", "transaction type", "direction"),
    "description": (
        "description",
        "memo",
        "narrative",
        "details",
        "transaction",
    ),
    "currency": ("currency", "ccy", "iso_currency", "currency code"),
}


def _casefold_columns(columns: Iterable[str]) -> dict[str, str]:
    """Return a mapping of lower-cased column names to their original spelling."""

    return {str(col).strip().casefold(): col for col in columns}


def _find_column(columns: Mapping[str, str], candidates: Iterable[str]) -> Optional[str]:
    """Locate the first matching column by heuristic candidates."""

    for candidate in candidates:
        lowered = candidate.casefold()
        if lowered in columns:
            return columns[lowered]
    return None


AMOUNT_SANITIZE_PATTERN = re.compile(r"[^0-9.+-]")


def _to_numeric(value: Any) -> Optional[float]:
    """Best-effort conversion of a raw amount field to float."""

    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        cleaned = AMOUNT_SANITIZE_PATTERN.sub("", value)
        if not cleaned or cleaned in {"+", "-"}:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None

    return None


def _infer_column_mapping(frame: pd.DataFrame, mapping: ColumnMapping) -> ColumnMapping:
    """Fill in missing mapping fields using heuristic column matches."""

    column_lookup = _casefold_columns(frame.columns)

    resolved = ColumnMapping(
        date=mapping.date or _find_column(column_lookup, HEURISTIC_CANDIDATES["date"]),
        amount=mapping.amount or _find_column(column_lookup, HEURISTIC_CANDIDATES["amount"]),
        credit=mapping.credit or _find_column(column_lookup, HEURISTIC_CANDIDATES["credit"]),
        debit=mapping.debit or _find_column(column_lookup, HEURISTIC_CANDIDATES["debit"]),
        transaction_type=
        mapping.transaction_type
        or _find_column(column_lookup, HEURISTIC_CANDIDATES["transaction_type"]),
        currency=mapping.currency or _find_column(column_lookup, HEURISTIC_CANDIDATES["currency"]),
        description=
        mapping.description or _find_column(column_lookup, HEURISTIC_CANDIDATES["description"]),
    )

    missing = []
    if not resolved.date:
        missing.append("date")
    if not (resolved.amount or (resolved.credit and resolved.debit)):
        missing.append("amount or credit/debit pair")

    if missing:
        LOGGER.warning(
            "Column mapping may be incomplete. Missing: %s."
            " Consider providing explicit column flags.",
            ", ".join(missing),
        )

    return resolved


def _normalize_amounts(frame: pd.DataFrame, mapping: ColumnMapping) -> pd.Series:
    """Return a signed amount series derived from the mapping."""

    if mapping.amount:
        if mapping.amount not in frame:
            raise KeyError(
                f"Amount column '{mapping.amount}' not found in statement."
            )
        amount_series = frame[mapping.amount].apply(_to_numeric)
    else:
        if not mapping.credit and not mapping.debit:
            raise ValueError(
                "Unable to determine transaction amounts. Provide an amount column or"
                " both credit and debit columns."
            )

        if mapping.credit and mapping.credit not in frame:
            raise KeyError(
                f"Credit column '{mapping.credit}' not found in statement."
            )
        if mapping.debit and mapping.debit not in frame:
            raise KeyError(
                f"Debit column '{mapping.debit}' not found in statement."
            )

        credit_series = (
            frame[mapping.credit].apply(_to_numeric) if mapping.credit else pd.Series(0.0, index=frame.index)
        )
        debit_series = (
            frame[mapping.debit].apply(_to_numeric) if mapping.debit else pd.Series(0.0, index=frame.index)
        )

        amount_series = credit_series.fillna(0.0) - debit_series.fillna(0.0)

    return amount_series


def _normalize_currency(frame: pd.DataFrame, mapping: ColumnMapping, default_currency: Optional[str]) -> pd.Series:
    """Return a currency series, falling back to a default if necessary."""

    if mapping.currency and mapping.currency in frame:
        series = frame[mapping.currency].astype(str).str.strip().str.upper()
        if default_currency:
            series = series.fillna(default_currency).replace("", default_currency)
        return series

    if default_currency:
        return pd.Series([default_currency] * len(frame), index=frame.index)

    return pd.Series(["UNKNOWN"] * len(frame), index=frame.index)


def _normalize_transaction_type(
    frame: pd.DataFrame,
    mapping: ColumnMapping,
    amounts: pd.Series,
) -> pd.Series:
    """Derive transaction direction from explicit column or amount sign."""

    if mapping.transaction_type and mapping.transaction_type in frame:
        raw_series = frame[mapping.transaction_type].astype(str).str.strip().str.lower()
        normalized = raw_series.replace(
            {
                "credit": TransactionType.INFLOW.value,
                "debit": TransactionType.OUTFLOW.value,
                "income": TransactionType.INFLOW.value,
                "expense": TransactionType.OUTFLOW.value,
                "withdrawal": TransactionType.OUTFLOW.value,
                "deposit": TransactionType.INFLOW.value,
            }
        )
        normalized = normalized.where(
            normalized.isin({TransactionType.INFLOW.value, TransactionType.OUTFLOW.value}),
            other=None,
        )
    else:
        normalized = pd.Series([None] * len(frame), index=frame.index)

    # Fallback to amount sign when direction is undefined
    missing_mask = normalized.isna()
    if missing_mask.any():
        inferred = pd.Series(
            [
                TransactionType.INFLOW.value
                if amt is not None and amt >= 0
                else TransactionType.OUTFLOW.value
                for amt in amounts
            ],
            index=amounts.index,
        )
        normalized = normalized.where(~missing_mask, inferred)

    return normalized


def _normalize_description(frame: pd.DataFrame, mapping: ColumnMapping) -> pd.Series:
    """Return the best available textual description."""

    if mapping.description and mapping.description in frame:
        return frame[mapping.description].astype(str).fillna("")

    return pd.Series([""] * len(frame), index=frame.index)


def _extract_with_openai(path: pathlib.Path) -> Optional[pd.DataFrame]:
    api_key = os.getenv(OPENAI_KEY_ENV)
    if not api_key:
        LOGGER.warning(
            "Cannot use OpenAI extraction because %s is not set in the environment.",
            OPENAI_KEY_ENV,
        )
        return None

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        LOGGER.error(
            "OpenAI python client is not installed. Install the optional dependency or"
            " disable the LLM extraction step."
        )
        return None

    client = OpenAI(api_key=api_key)
    model = os.getenv(OPENAI_MODEL_ENV, DEFAULT_OPENAI_MODEL)

    instructions = textwrap.dedent(
        """
        You are an expert financial data extractor. Parse every transaction you can
        find in the supplied statement file. Return a JSON object with a top-level key
        "transactions" whose value is an array. Each transaction must include the
        keys: date, description, amount, currency, transaction_type. Use ISO 8601
        (YYYY-MM-DD) for dates when available. Amounts must be numeric (positive for
        inflows, negative for outflows when possible). The transaction_type must be
        either "inflow" or "outflow". Populate null when data is missing and ignore
        totals, headers, footers, or explanatory notes.
        """
    ).strip()

    uploaded_file_id: Optional[str] = None
    try:
        with path.open("rb") as file_handle:
            uploaded = client.files.create(file=file_handle, purpose="assistants")
        uploaded_file_id = uploaded.id
    except Exception as exc:  # pragma: no cover - network interaction
        LOGGER.error("Failed to upload statement %s to OpenAI: %s", path, exc)
        return None

    try:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": "You convert financial statements into structured transaction data.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": instructions},
                        {"type": "input_file", "file_id": uploaded_file_id},
                    ],
                },
            ],
        )
    except Exception as exc:  # pragma: no cover - network interaction
        LOGGER.error("OpenAI extraction request failed: %s", exc)
        return None
    # finally:
    #     if uploaded_file_id:
    #         try:
    #             client.files.delete(uploaded_file_id)
    #         except Exception:  # pragma: no cover - best effort cleanup
    #             LOGGER.debug("Failed to delete temporary OpenAI file", exc_info=True)

    print(output_text)
    output_text = getattr(response, "output_text", None)
    if not output_text:
        text_fragments: list[str] = []
        for item in getattr(response, "output", []):
            for content in getattr(item, "content", []):
                if getattr(content, "type", None) == "output_text":
                    text_fragments.append(getattr(content, "text", ""))
        output_text = "".join(text_fragments).strip()

    if not output_text:
        LOGGER.error("OpenAI response did not contain textual output.")
        return None

    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError as exc:
        LOGGER.error("Failed to decode OpenAI JSON payload: %s", exc)
        return None

    transactions = payload.get("transactions")
    if not transactions:
        LOGGER.warning("OpenAI response did not contain any transactions.")
        return None

    frame = pd.DataFrame(transactions)
    frame.columns = [str(col).strip().casefold() for col in frame.columns]

    for required in ("date", "amount", "description", "currency", "transaction_type"):
        if required not in frame.columns:
            frame[required] = None

    return frame


def load_raw_statement(path: pathlib.Path) -> pd.DataFrame:
    """Load a statement file as a Pandas DataFrame."""

    suffix = path.suffix.lower()

    try:
        if suffix in {".csv", ".txt"}:
            return pd.read_csv(path)
        if suffix in {".xls", ".xlsx", ".xlsm"}:
            return pd.read_excel(path)
    except Exception as exc:
        LOGGER.warning(
            "Failed to parse %s using pandas (%s). Falling back to OpenAI extraction.",
            path,
            exc,
        )

    extracted = _extract_with_openai(path)
    print(extracted)
    if extracted is not None:
        return extracted

    raise ValueError(
        f"Unsupported or unreadable file format: {suffix}. Provide CSV/Excel or configure OpenAI extraction."
    )


def normalize_statement(
    frame: pd.DataFrame,
    column_mapping: ColumnMapping,
    default_currency: Optional[str] = None,
) -> pd.DataFrame:
    """Normalize a raw statement DataFrame into the canonical schema."""

    resolved_mapping = _infer_column_mapping(frame, column_mapping)

    if not resolved_mapping.date:
        raise ValueError(
            "A date column is required for normalization. Provide --date-column or"
            " include a 'date' field in the column mapping JSON."
        )

    amounts = _normalize_amounts(frame, resolved_mapping)
    currency = _normalize_currency(frame, resolved_mapping, default_currency)
    transaction_type = _normalize_transaction_type(frame, resolved_mapping, amounts)
    description = _normalize_description(frame, resolved_mapping)

    dates = pd.to_datetime(frame[resolved_mapping.date], errors="coerce")
    if isinstance(dates, pd.Series):
        dates = dates.dt.date

    normalized = pd.DataFrame(
        {
            "Date": dates,
            "Amount": amounts,
            "Currency": currency,
            "TransactionType": transaction_type,
            "Description": description,
        }
    )

    normalized = normalized.dropna(subset=["Date", "Amount"])
    normalized = normalized.assign(
        TransactionType=normalized["TransactionType"].fillna(TransactionType.OUTFLOW.value),
        Description=normalized["Description"].fillna("").str.strip(),
    )

    return normalized


def load_column_mapping_json(path: Optional[str]) -> Optional[Mapping[str, str]]:
    """Load a column mapping dictionary from a JSON file."""

    if not path:
        return None

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_statement_file(
    path: pathlib.Path,
    column_mapping: ColumnMapping,
    default_currency: Optional[str] = None,
) -> pd.DataFrame:
    """End-to-end convenience function to normalize a statement file."""

    frame = load_raw_statement(path)
    return normalize_statement(frame, column_mapping, default_currency)


def summarize_transactions(normalized: pd.DataFrame) -> dict[str, Any]:
    """Compute summary statistics for a normalized transaction DataFrame."""

    totals = normalized.groupby("TransactionType")["Amount"].sum()
    inflow_total = float(totals.get(TransactionType.INFLOW.value, 0.0))
    outflow_total = float(totals.get(TransactionType.OUTFLOW.value, 0.0))

    return {
        "rows": int(len(normalized)),
        "total_inflow": inflow_total,
        "total_outflow": outflow_total,
        "net": inflow_total - outflow_total,
    }


__all__ = [
    "TransactionType",
    "ColumnMapping",
    "load_raw_statement",
    "normalize_statement",
    "normalize_statement_file",
    "summarize_transactions",
    "load_column_mapping_json",
    "main",
]


def main(argv: Optional[list[str]] = None) -> None:
    """Allow running this module directly by delegating to the CLI entry point."""

    try:
        from .main import main as cli_main  # type: ignore
    except ImportError:
        import importlib

        cli_module = importlib.import_module("main")
        cli_main = getattr(cli_module, "main")

    cli_main(argv)


if __name__ == "__main__":
    main()


