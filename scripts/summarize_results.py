from pathlib import Path
import csv


RUNS = [
    ("T", "results/mnist_teacher.csv"),
    ("CE", "results/mnist_student_ce.csv"),
    ("KD", "results/mnist_student_kd.csv"),
    ("KD + Fisher", "results/mnist_student_kd_fisher.csv"),
    ("KD + Energy", "results/mnist_student_kd_energy.csv"),
    ("KD + Fisher + Energy", "results/mnist_student_kd_fisher_energy.csv"),
]


SUMMARY_PATH = Path("results/mnist_summary.csv")


def read_csv(path):
    path = Path(path)

    if not path.exists():
        print(f"missing: {path}")
        return []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_float(row, key, default=0.0):
    value = row.get(key, "")

    if value == "" or value is None:
        return default

    return float(value)


def summarize_run(method, path):
    rows = read_csv(path)

    if len(rows) == 0:
        return None

    test_rows = [row for row in rows if row.get("phase") == "test"]

    if len(test_rows) == 0:
        print(f"no final test row found in: {path}")
        return None

    test_row = test_rows[-1]

    return {
        "method": method,
        "best_epoch": int(float(test_row.get("best_epoch", 0))),
        "test_acc": get_float(test_row, "test_acc"),
        "test_nll": get_float(test_row, "test_nll"),
        "test_ece": get_float(test_row, "test_ece"),
        "test_ts_kl": get_float(test_row, "test_teacher_student_kl"),
        "test_fisher_mismatch": get_float(test_row, "test_fisher_mismatch"),
        "test_energy_mismatch": get_float(test_row, "test_energy_mismatch"),
    }

def print_table(summary_rows):
    columns = [
        "method",
        "best_epoch",
        "test_acc",
        "test_nll",
        "test_ece",
        "test_ts_kl",
        "test_fisher_mismatch",
        "test_energy_mismatch",
    ]

    widths = {
        "method": 24,
        "best_epoch": 10,
        "test_acc": 10,
        "test_nll": 10,
        "test_ece": 10,
        "test_ts_kl": 12,
        "test_fisher_mismatch": 22,
        "test_energy_mismatch": 22,
    }

    header = " | ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    print("-" * len(header))

    for row in summary_rows:
        values = []

        for col in columns:
            value = row.get(col, "")

            if col == "method":
                values.append(str(value).ljust(widths[col]))
            elif col == "best_epoch":
                values.append(str(value).ljust(widths[col]))
            else:
                values.append(f"{float(value):.6f}".ljust(widths[col]))

        print(" | ".join(values))


def save_summary(summary_rows):
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "method",
        "best_epoch",
        "test_acc",
        "test_nll",
        "test_ece",
        "test_ts_kl",
        "test_fisher_mismatch",
        "test_energy_mismatch",
    ]

    with open(SUMMARY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nsaved summary: {SUMMARY_PATH}")


def main():
    summary_rows = []

    for method, path in RUNS:
        summary = summarize_run(method, path)

        if summary is not None:
            summary_rows.append(summary)

    if len(summary_rows) == 0:
        print("No result files found.")
        return

    print_table(summary_rows)
    save_summary(summary_rows)


if __name__ == "__main__":
    main()