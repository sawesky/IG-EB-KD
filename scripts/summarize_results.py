from pathlib import Path
import csv


RUNS = [
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

    final_row = rows[-1]

    best_row = max(rows, key=lambda r: get_float(r, "test_acc"))

    return {
        "method": method,
        "best_epoch": int(get_float(best_row, "epoch")),
        "best_acc": get_float(best_row, "test_acc"),
        "final_acc": get_float(final_row, "test_acc"),
        "final_nll": get_float(final_row, "test_nll"),
        "final_ece": get_float(final_row, "test_ece"),
        "final_ts_kl": get_float(final_row, "test_teacher_student_kl"),
        "final_fisher_mismatch": get_float(final_row, "test_fisher_mismatch"),
        "final_energy_mismatch": get_float(final_row, "test_energy_mismatch"),
        "final_train_fisher": get_float(final_row, "train_fisher"),
        "final_train_energy_margin": get_float(final_row, "train_energy_margin"),
    }


def print_table(summary_rows):
    columns = [
        "method",
        "best_epoch",
        "best_acc",
        "final_acc",
        "final_nll",
        "final_ece",
        "final_ts_kl",
        "final_fisher_mismatch",
        "final_energy_mismatch",
    ]

    widths = {
        "method": 24,
        "best_epoch": 10,
        "best_acc": 10,
        "final_acc": 10,
        "final_nll": 10,
        "final_ece": 10,
        "final_ts_kl": 12,
        "final_fisher_mismatch": 22,
        "final_energy_mismatch": 22,
    }

    header = " | ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    print("-" * len(header))

    for row in summary_rows:
        values = []

        for col in columns:
            value = row[col]

            if col == "method":
                values.append(str(value).ljust(widths[col]))
            elif col == "best_epoch":
                values.append(str(value).ljust(widths[col]))
            else:
                values.append(f"{value:.6f}".ljust(widths[col]))

        print(" | ".join(values))


def save_summary(summary_rows):
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "method",
        "best_epoch",
        "best_acc",
        "final_acc",
        "final_nll",
        "final_ece",
        "final_ts_kl",
        "final_fisher_mismatch",
        "final_energy_mismatch",
        "final_train_fisher",
        "final_train_energy_margin",
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