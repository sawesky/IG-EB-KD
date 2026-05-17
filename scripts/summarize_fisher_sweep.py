from pathlib import Path
import csv


RESULTS_DIR = Path("results")
PATTERN = "mnist_student_kd_fisher_a*.csv"
OUT_PATH = RESULTS_DIR / "fisher_sweep_summary.csv"


def parse_alpha(path):
    # Example:
    # mnist_student_kd_fisher_a0p01.csv -> 0.01
    stem = path.stem
    alpha_part = stem.split("_a")[-1]
    return float(alpha_part.replace("p", "."))


def read_rows(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def get_float(row, key, default=0.0):
    value = row.get(key, "")

    if value == "" or value is None:
        return default

    return float(value)


def summarize_file(path):
    rows = read_rows(path)

    val_rows = [row for row in rows if row.get("phase") == "val"]
    test_rows = [row for row in rows if row.get("phase") == "test"]

    if len(val_rows) == 0:
        print(f"no validation rows found: {path}")
        return None

    if len(test_rows) == 0:
        print(f"no final test row found: {path}")
        return None

    best_val_row = min(val_rows, key=lambda row: get_float(row, "val_nll"))
    test_row = test_rows[-1]

    return {
        "alpha": parse_alpha(path),
        "file": str(path),

        "best_epoch": int(float(best_val_row["epoch"])),

        "best_val_acc": get_float(best_val_row, "val_acc"),
        "best_val_nll": get_float(best_val_row, "val_nll"),
        "best_val_ece": get_float(best_val_row, "val_ece"),
        "best_val_ts_kl": get_float(best_val_row, "val_teacher_student_kl"),
        "best_val_fisher_mismatch": get_float(best_val_row, "val_fisher_mismatch"),
        "best_val_energy_mismatch": get_float(best_val_row, "val_energy_mismatch"),

        "test_acc": get_float(test_row, "test_acc"),
        "test_nll": get_float(test_row, "test_nll"),
        "test_ece": get_float(test_row, "test_ece"),
        "test_ts_kl": get_float(test_row, "test_teacher_student_kl"),
        "test_fisher_mismatch": get_float(test_row, "test_fisher_mismatch"),
        "test_energy_mismatch": get_float(test_row, "test_energy_mismatch"),
    }


def print_table(rows):
    columns = [
        "alpha",
        "best_epoch",
        "best_val_acc",
        "best_val_nll",
        "best_val_ece",
        "best_val_fisher_mismatch",
        "test_acc",
        "test_nll",
        "test_ece",
        "test_fisher_mismatch",
    ]

    widths = {
        "alpha": 10,
        "best_epoch": 10,
        "best_val_acc": 14,
        "best_val_nll": 14,
        "best_val_ece": 14,
        "best_val_fisher_mismatch": 24,
        "test_acc": 10,
        "test_nll": 10,
        "test_ece": 10,
        "test_fisher_mismatch": 22,
    }

    header = " | ".join(col.ljust(widths[col]) for col in columns)
    print("\nFISHER SWEEP SUMMARY")
    print(header)
    print("-" * len(header))

    for row in rows:
        values = []

        for col in columns:
            value = row[col]

            if col == "best_epoch":
                values.append(str(value).ljust(widths[col]))
            elif col == "alpha":
                values.append(str(value).ljust(widths[col]))
            else:
                values.append(f"{value:.6f}".ljust(widths[col]))

        print(" | ".join(values))


def save_summary(rows):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "alpha",
        "file",
        "best_epoch",

        "best_val_acc",
        "best_val_nll",
        "best_val_ece",
        "best_val_ts_kl",
        "best_val_fisher_mismatch",
        "best_val_energy_mismatch",

        "test_acc",
        "test_nll",
        "test_ece",
        "test_ts_kl",
        "test_fisher_mismatch",
        "test_energy_mismatch",
    ]

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nsaved Fisher sweep summary: {OUT_PATH}")


def main():
    files = sorted(RESULTS_DIR.glob(PATTERN), key=parse_alpha)

    if len(files) == 0:
        print(f"No files found with pattern: {RESULTS_DIR / PATTERN}")
        return

    rows = []

    for path in files:
        summary = summarize_file(path)

        if summary is not None:
            rows.append(summary)

    if len(rows) == 0:
        print("No valid Fisher sweep files found.")
        return

    # Main sorting rule: validation NLL.
    rows_by_val_nll = sorted(rows, key=lambda row: row["best_val_nll"])

    print_table(rows_by_val_nll)
    save_summary(rows_by_val_nll)

    best_nll = rows_by_val_nll[0]
    best_fisher = min(rows, key=lambda row: row["best_val_fisher_mismatch"])

    print(
        "\nBest by validation NLL: "
        f"alpha={best_nll['alpha']}, "
        f"val_nll={best_nll['best_val_nll']:.6f}, "
        f"test_nll={best_nll['test_nll']:.6f}, "
        f"test_acc={best_nll['test_acc']:.6f}"
    )

    print(
        "Best by validation Fisher mismatch: "
        f"alpha={best_fisher['alpha']}, "
        f"val_fisher={best_fisher['best_val_fisher_mismatch']:.6f}, "
        f"test_fisher={best_fisher['test_fisher_mismatch']:.6f}, "
        f"test_nll={best_fisher['test_nll']:.6f}"
    )


if __name__ == "__main__":
    main()