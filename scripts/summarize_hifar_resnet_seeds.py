import csv
from pathlib import Path
from statistics import mean, stdev


SEEDS = [42, 43, 44, 45, 46]

TEACHER_PATH = Path("results/hifar_resnet56_teacher.csv")

RUNS = [
    ("ResNet-20 CE", "hifar_resnet20_student_ce"),
    ("ResNet-20 KD", "hifar_resnet20_student_kd"),
    ("ResNet-20 KD + Fisher", "hifar_resnet20_student_kd_fisher"),
    ("ResNet-20 KD + Energy", "hifar_resnet20_student_kd_energy"),
    ("ResNet-20 KD + Fisher + Energy", "hifar_resnet20_student_kd_fisher_energy"),
]

RESULT_DIR = Path("results/seeds")
OUT_PATH = Path("results/hifar_resnet_seeds_summary.csv")


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
    test_rows = [row for row in rows if row.get("phase") == "test"]

    if not test_rows:
        raise RuntimeError(f"No test row found in {path}")

    test_row = test_rows[-1]

    return {
        "best_epoch": int(float(test_row.get("best_epoch", 0))),
        "test_acc": get_float(test_row, "test_acc"),
        "test_nll": get_float(test_row, "test_nll"),
        "test_ece": get_float(test_row, "test_ece"),
        "test_ts_kl": get_float(test_row, "test_teacher_student_kl"),
        "test_fisher_mismatch": get_float(test_row, "test_fisher_mismatch"),
        "test_energy_mismatch": get_float(test_row, "test_energy_mismatch"),
    }


def mean_std(values):
    if len(values) == 1:
        return mean(values), 0.0
    return mean(values), stdev(values)


def add_metric_summary(out, seed_rows, metric):
    values = [row[metric] for row in seed_rows]
    m, s = mean_std(values)
    out[f"{metric}_mean"] = m
    out[f"{metric}_std"] = s


def make_teacher_row():
    row = summarize_file(TEACHER_PATH)

    return {
        "method": "ResNet-56 teacher",
        "n_seeds": 1,
        "test_acc_mean": row["test_acc"],
        "test_acc_std": 0.0,
        "test_nll_mean": row["test_nll"],
        "test_nll_std": 0.0,
        "test_ece_mean": row["test_ece"],
        "test_ece_std": 0.0,
        "test_ts_kl_mean": "",
        "test_ts_kl_std": "",
        "test_fisher_mismatch_mean": "",
        "test_fisher_mismatch_std": "",
        "test_energy_mismatch_mean": "",
        "test_energy_mismatch_std": "",
    }


def main():
    summary_rows = []

    if TEACHER_PATH.exists():
        summary_rows.append(make_teacher_row())
    else:
        print(f"missing teacher file: {TEACHER_PATH}")

    for method, base_name in RUNS:
        seed_rows = []

        for seed in SEEDS:
            path = RESULT_DIR / f"{base_name}_seed{seed}.csv"

            if not path.exists():
                print(f"missing: {path}")
                continue

            row = summarize_file(path)
            row["seed"] = seed
            seed_rows.append(row)

        if not seed_rows:
            continue

        out = {
            "method": method,
            "n_seeds": len(seed_rows),
        }

        metrics = [
            "test_acc",
            "test_nll",
            "test_ece",
            "test_ts_kl",
            "test_fisher_mismatch",
            "test_energy_mismatch",
        ]

        for metric in metrics:
            add_metric_summary(out, seed_rows, metric)

        summary_rows.append(out)

    fieldnames = [
        "method",
        "n_seeds",
        "test_acc_mean",
        "test_acc_std",
        "test_nll_mean",
        "test_nll_std",
        "test_ece_mean",
        "test_ece_std",
        "test_ts_kl_mean",
        "test_ts_kl_std",
        "test_fisher_mismatch_mean",
        "test_fisher_mismatch_std",
        "test_energy_mismatch_mean",
        "test_energy_mismatch_std",
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nHIFAR / CIFAR-100 ResNet multi-seed summary")
    print("method | n | acc mean std | nll mean std | ece mean std")
    print("-" * 90)

    for row in summary_rows:
        print(
            f"{row['method']} | "
            f"{row['n_seeds']} | "
            f"{float(row['test_acc_mean']):.4f}_{float(row['test_acc_std']):.4f} | "
            f"{float(row['test_nll_mean']):.4f}_{float(row['test_nll_std']):.4f} | "
            f"{float(row['test_ece_mean']):.4f}_{float(row['test_ece_std']):.4f}"
        )

    print(f"\nsaved summary: {OUT_PATH}")


if __name__ == "__main__":
    main()