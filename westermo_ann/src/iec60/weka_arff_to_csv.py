"""
Stream a WEKA ARFF file (often mis-saved as *.csv) into a real comma-separated UTF-8 CSV.

Usage:
  python src/iec60/weka_arff_to_csv.py --input train.csv --output outputs/train_real.csv
  python src/iec60/weka_arff_to_csv.py --input train.csv --output outputs/train_sample.csv --max-rows 50000

The last attribute @class@ (WEKA label column) is renamed to \"class\" in the CSV header.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter

# @attribute <name> <type...>  ; name may be quoted or a bare token like @class@
_ATTR_RE = re.compile(
    r"^@attribute\s+(?P<name>'[^']*'|\"[^\"]*\"|[^\s]+)\s+(?P<type>.+?)\s*$",
    re.IGNORECASE,
)


def _strip_attr_name(raw: str) -> str:
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        return raw[1:-1]
    return raw


def parse_attribute_names(lines_iter) -> list[str]:
    """Consume lines from start of ARFF until @data; return column names."""
    names: list[str] = []
    for raw in lines_iter:
        line = raw.strip()
        if not line or line.lstrip().startswith("%"):
            continue
        low = line.lower()
        if low.startswith("@relation"):
            continue
        if low.startswith("@data"):
            break
        if not low.startswith("@attribute"):
            continue
        m = _ATTR_RE.match(line)
        if not m:
            continue
        col = _strip_attr_name(m.group("name"))
        if col == "@class@":
            col = "class"
        names.append(col)
    if not names:
        raise ValueError("No @attribute declarations found before @data.")
    return names


def iter_arff_data_rows(lines_iter, *, max_rows: int, progress_every: int) -> Iterator[list[str]]:
    """Yield parsed CSV cell lists for each @data row (caller already consumed header through @data)."""
    n = 0
    for raw in lines_iter:
        line = raw.rstrip("\n\r")
        if not line.strip() or line.lstrip().startswith("%"):
            continue
        try:
            row = next(csv.reader(io.StringIO(line)))
        except Exception as e:
            raise ValueError(f"Failed to parse @data line {n + 1}: {e!s}") from e
        n += 1
        yield row
        if progress_every > 0 and n % progress_every == 0:
            print(f"[convert] wrote {n} data rows...", flush=True)
        if max_rows > 0 and n >= max_rows:
            break


def convert_arff_to_csv(
    input_path: Path,
    output_path: Path,
    *,
    max_rows: int = 0,
    progress_every: int = 100_000,
) -> int:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = perf_counter()
    with input_path.open("r", encoding="utf-8", errors="replace", newline="") as fin:
        header_lines = fin
        names = parse_attribute_names(header_lines)

        with output_path.open("w", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout, lineterminator="\n")
            writer.writerow(names)
            count = 0
            for row in iter_arff_data_rows(fin, max_rows=max_rows, progress_every=progress_every):
                if len(row) != len(names):
                    raise ValueError(
                        f"Column count mismatch at data row {count + 1}: "
                        f"expected {len(names)} fields, got {len(row)}"
                    )
                writer.writerow(row)
                count += 1

    elapsed = perf_counter() - t0
    print(f"[convert] done: {count} rows -> {output_path} ({elapsed:.1f}s)", flush=True)
    return count


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream WEKA ARFF (or ARFF mislabeled as .csv) to UTF-8 CSV.")
    p.add_argument("--input", type=Path, required=True, help="Source ARFF path.")
    p.add_argument("--output", type=Path, required=True, help="Destination CSV path.")
    p.add_argument("--max-rows", type=int, default=0, help="Cap data rows (0 = no cap).")
    p.add_argument(
        "--progress-every",
        type=int,
        default=100_000,
        help="Print progress every N data rows (0 = disable).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")
    convert_arff_to_csv(
        args.input,
        args.output,
        max_rows=int(args.max_rows),
        progress_every=int(args.progress_every),
    )


if __name__ == "__main__":
    main()
