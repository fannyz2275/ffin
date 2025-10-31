## Personal Finance Agent

This project explores an AI-powered personal finance assistant. The first
feature focuses on understanding historical cashflows by normalizing exported
banking and brokerage statements into a canonical schema:

- `Date`
- `Amount`
- `Currency`
- `TransactionType` (`inflow` or `outflow`)
- `Description`

The normalized data powers downstream analytics such as baseline cashflow
projections and scenario planning for future life events.

### Quick Start

1. **Install dependencies**

   ```bash
   uv sync
   ```

2. **Normalize a statement file**

   ```bash
   uv run python -m ffin.main ingest --input ~/Downloads/checking.csv --currency USD --summary
   ```

   - Supports CSV (`.csv`, `.txt`) and Excel (`.xls`, `.xlsx`, `.xlsm`)
     natively. Other formats are routed through an OpenAI extraction call when
     `OPENAI_API_KEY` is set. Override the default model (`gpt-4o-mini`) with
     `FFIN_OPENAI_MODEL` if desired.
   - If the output path is omitted, records are printed to stdout as JSON. Use
     `--output normalized.csv` to persist a file.
   - Pass `--summary` to see total inflows, outflows, and net cashflow.

3. **Column mapping**

   The CLI attempts to infer common column names. Provide overrides if the
   inference is insufficient:

   ```bash
   uv run python -m ffin.main ingest \
     --input ~/Downloads/brokerage.xlsx \
     --date-column "Trade Date" \
     --amount-column "Net Amount" \
     --currency-column "Currency" \
     --description-column "Description" \
     --summary
   ```

   For more complex mappings, supply a JSON file via `--column-mapping` with
   keys among `date`, `amount`, `credit`, `debit`, `currency`, `description`,
   and `transaction_type`.

### LLM Extraction Notes

- The OpenAI pathway uploads the raw statement file and asks the model to
  return structured JSON transactions. Expect higher latency and API charges
  relative to CSV/Excel parsing.
- Ensure `OPENAI_API_KEY` (and optionally `FFIN_OPENAI_MODEL`) is exported before
  running the CLI. The model should support multimodal/file inputs (e.g.
  `gpt-4o-mini`, `gpt-4.1`).
- Image-only PDFs still require an OCR step before model extraction can succeed.

### Development Notes

- `ffin/ingest.py` contains the normalization logic and helper utilities.
- `ffin/main.py` exposes a command-line entry point for the ingestion workflow.
- Future milestones include modeling projections, budgeting guidance, and
  recommendation loops powered by the normalized data foundation.
