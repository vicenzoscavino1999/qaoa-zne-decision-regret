"""Read-only stdlib checks of unified manuscript tables against saved JSON/CSV.

This verifies numeric transcription, not the underlying inferential assumptions.
No generated TeX fragments or prose substrings are used as evidence.
"""
from __future__ import annotations
import csv
from decimal import Decimal
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ORDER = {"linear": 1, "quadratic": 2, "cubic": 3}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def manuscript_source(root=ROOT):
    """Expand inputs from the actual unified manuscript entry point."""
    base = Path(root) / "MANUSCRITO_ACTUAL/fuentes"

    def expand(path, stack):
        path = path.resolve()
        require(path not in stack, f"Recursive manuscript input: {path}")
        source = re.sub(r"(?<!\\)%[^\n]*", "", path.read_text(encoding="utf-8"))

        def include(match):
            target = base / match.group(1)
            return expand(target if target.suffix else target.with_suffix(".tex"), stack + (path,))

        return re.sub(r"\\(?:input|include)\s*\{([^{}]+)\}", include, source)

    return expand(base / "main.tex", ())


def plain(cell):
    """Ignore presentational commands but retain mathematical tokens."""
    cell = cell.replace("$", "").replace(r"\(", "").replace(r"\)", "")
    cell = re.sub(r"\\(?:,|;|!|quad\b|qquad\b|enspace\b|thinspace\b)", "", cell)
    cell = re.sub(r"\\(?:textbf|textit|textrm|textnormal|mathrm|mathbf|mathit|ensuremath|shortstack|num|text|mbox)\s*", "", cell)
    return re.sub(r"[\s{}]", "", cell.replace(r"\\", ""))


def split_rows(body):
    """Split top-level row ends; shortstack line breaks remain within cells."""
    rows, start, depth, i = [], 0, 0, 0
    while i < len(body):
        if body[i] == "{":
            depth += 1
        elif body[i] == "}":
            depth -= 1
            require(depth >= 0, "Unbalanced table braces")
        elif body[i:i + 2] == r"\\" and depth == 0:
            rows.append(body[start:i])
            i += 2
            spacing = re.match(r"\s*\[[^\]]*\]", body[i:])
            if spacing:
                i += spacing.end()
            start = i
            continue
        i += 1
    require(depth == 0, "Unbalanced table braces")
    rows.append(body[start:])
    cleaned = []
    for row in rows:
        row = re.sub(r"\\(?:hline|toprule|midrule|bottomrule|addlinespace)\b(?:\s*\[[^\]]*\])?", "", row).strip()
        if row:
            cleaned.append([cell.strip() for cell in row.split("&")])
    return cleaned


def table_rows(source, label):
    source = re.sub(r"(?<!\\)%[^\n]*", "", source)
    label_pattern = r"\\label\s*\{\s*" + re.escape(label) + r"\s*\}"
    require(len(re.findall(label_pattern, source)) == 1, f"Expected one unique label {label}")
    tables = re.findall(r"\\begin\s*\{table\*?\}(.*?)\\end\s*\{table\*?\}", source, re.S)
    selected = [table for table in tables if re.search(label_pattern, table)]
    require(len(selected) == 1, f"Expected one table labeled {label}; found {len(selected)}")
    bodies = re.findall(r"\\begin\s*\{tabular\}\s*(?:\[[^\]]*\]\s*)?\{[^{}]*\}(.*?)\\end\s*\{tabular\}", selected[0], re.S)
    require(len(bodies) == 1, f"Expected one tabular in {label}")
    return split_rows(bodies[0])


def number(cell):
    """Return a decimal and the magnitude of its last displayed place."""
    token = plain(cell).replace(r"\times", "e").replace(r"\cdot", "e")
    token = re.sub(r"e10\^([+-]?\d+)$", r"e\1", token)
    require(re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", token) is not None,
            f"Not a numeric TeX cell: {cell!r}")
    value = Decimal(token)
    return value, Decimal(1).scaleb(value.as_tuple().exponent)


def rounded_equal(cell, expected, max_quantum, context):
    value, quantum = number(cell)
    require(quantum <= max_quantum, f"{context}: insufficient displayed precision in {cell!r}")
    reference = Decimal(str(expected))
    require(abs(value - reference) <= quantum / 2,
            f"{context}: printed {cell!r}, saved value {reference}")


def condition(cells, context):
    graph = plain(cells[0])
    require(re.fullmatch(r"H\d+", graph) is not None, f"{context}: invalid graph {graph!r}")
    tokens = plain(cells[1])
    require(tokens.startswith("(") and tokens.endswith(")"), f"{context}: invalid condition tuple")
    parts = tokens[1:-1].split(",")
    require(len(parts) == 3, f"{context}: expected noise, budget, order")
    values = [number(part)[0] for part in parts]
    require(values[1] == values[1].to_integral_value() and values[2] == values[2].to_integral_value(),
            f"{context}: noninteger budget/order")
    return graph, values[0], int(values[1]), int(values[2])


def verify_manuscript(root=ROOT, *, source_text=None, tables=("confirmation", "revised")):
    """source_text allows isolated mutation tests without editing canonical files."""
    root = Path(root)
    source = manuscript_source(root) if source_text is None else source_text
    result = {}
    for name in tables:
        require(name in ("confirmation", "revised"), f"Unknown table: {name}")
        confirmation = name == "confirmation"
        if confirmation:
            saved = json.loads((root / "confirmation_20261019/analysis/results.json").read_text(encoding="utf-8"))["cells"]
        else:
            with (root / "revision_20261015/statistics/cells_D.csv").open(newline="", encoding="utf-8") as handle:
                saved = [row for row in csv.DictReader(handle) if row["revised_holm_reject"].lower() == "true"]
        expected = {}
        for row in saved:
            key = (row["graph"] if confirmation else row["instance"], Decimal(str(row["noise"])),
                   int(row["budget"] if confirmation else row["shots_total"]),
                   ORDER[row["order"] if confirmation else row["zne_order"]])
            require(key not in expected, f"Duplicate saved condition in {name}: {key}")
            expected[key] = row
        rows = table_rows(source, f"tab:{name}")
        header = ["Graph", "(p,B,m)"] + (["K_r"] if confirmation else []) + ["K_d", "r", "d_R", "Nativeshift", "Holmp"]
        require(rows and [plain(cell) for cell in rows[0]] == header,
                f"tab:{name}: wrong columns or column order")
        seen = set()
        for cells in rows[1:]:
            context = f"tab:{name} row {len(seen) + 1}"
            require(len(cells) == len(header), f"{context}: wrong column count")
            key = condition(cells, context)
            require(key in expected, f"{context}: unexpected condition {key}")
            require(key not in seen, f"{context}: duplicate condition {key}")
            seen.add(key)
            row = expected[key]
            offset = 2
            for field in (["positive_mse", "positive_regret"] if confirmation else ["positive_regret"]):
                value, _ = number(cells[offset])
                require(value == Decimal(str(row[field])) and value == value.to_integral_value(),
                        f"{context}: wrong {field}")
                offset += 1
            fields = ["median_r_mse", "median_d_R", "native_median_d" if confirmation else "median_d_objective", "p_holm" if confirmation else "sign_holm"]
            p_reference = Decimal(str(row[fields[-1]]))
            # Preserve >=3 significant figures (confirmation) or >=6 decimal places (screen).
            p_quantum = Decimal(1).scaleb(p_reference.adjusted() - 2) if confirmation else Decimal("0.000001")
            precisions = [Decimal("0.0001"), Decimal("0.00001"), Decimal("0.00001"), p_quantum]
            for cell, field, precision in zip(cells[offset:], fields, precisions):
                rounded_equal(cell, row[field], precision, f"{context} {field}")
        require(seen == set(expected), f"tab:{name}: missing conditions {sorted(set(expected) - seen)}")
        result[f"tab:{name}"] = {"rows": len(seen), "numeric_cells_verified": len(seen) * (6 if confirmation else 5),
                                  "condition_keys_and_column_order_verified": True,
                                  "rounding": "half the last displayed place, with minimum published precision"}
    return result


if __name__ == "__main__":
    print(json.dumps({"status": "PASS", "tables": verify_manuscript()}, indent=2))
