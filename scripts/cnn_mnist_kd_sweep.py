import copy
import csv
import os
import subprocess
import sys
from pathlib import Path

import yaml


BASE_CONFIG = "configs/mnist_student_kd.yaml"

TEMPERATURES = [1.0, 1.5, 2.0, 3.0, 4.0]
LAMBDAS = [0.2, 0.4, 0.6, 0.8, 1.0]

OUT_PATH = Path("results/mnist_kd_sweep_summary.csv")


def value_to_name(value):
    return str(value).replace(".", "p")


def read_rows(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def get_float(row, key, default=0.0):
    value = row.get(key, "")
    if value == "" or value is None:
        return default
    return float(value)


def summarize_run(metrics_path):
    rows = read_rows(metrics_path)

    val_rows = [row for row in rows if row.get("phase") == "val"]
    test_rows = [row for row in rows if row.get("phase") == "test"]

    if not val_rows:
        raise RuntimeError(f"No validation rows found in {metrics_path}")

    best_val_row = min(val_rows, key=lambda row: get_float(row, "val_nll"))
    test_row = test_rows[-1] if test_rows else {}

    return {
        "best_epoch": int(float(best_val_row["epoch"])),
        "best_val_acc": get_float(best_val_row, "val_acc"),
        "best_val_nll": get_float(best_val_row, "val_nll"),
        "best_val_ece": get_float(best_val_row, "val_ece"),
        "test_acc": get_float(test_row, "test_acc"),
        "test_nll": get_float(test_row, "test_nll"),
        "test_ece": get_float(test_row, "test_ece"),
        "test_ts_kl": get_float(test_row, "test_teacher_student_kl"),
        "test_fisher_mismatch": get_float(test_row, "test_fisher_mismatch"),
        "test_energy_mismatch": get_float(test_row, "test_energy_mismatch"),
    }


def print_summary(summary_rows):
    columns = [
        "T",
        "lambda",
        "best_epoch",
        "best_val_acc",
        "best_val_nll",
        "best_val_ece",
        "test_acc",
        "test_nll",
        "test_ece",
        "test_ts_kl",
        "test_fisher_mismatch",
        "test_energy_mismatch",
    ]

    widths = {
        "T": 8,
        "lambda": 8,
        "best_epoch": 10,
        "best_val_acc": 14,
        "best_val_nll": 14,
        "best_val_ece": 14,
        "test_acc": 10,
        "test_nll": 10,
        "test_ece": 10,
        "test_ts_kl": 12,
        "test_fisher_mismatch": 22,
        "test_energy_mismatch": 22,
    }

    header = " | ".join(col.ljust(widths[col]) for col in columns)
    print("\nMNIST LENET KD SWEEP SUMMARY")
    print(header)
    print("-" * len(header))

    for row in summary_rows:
        values = []
        for col in columns:
            value = row[col]
            if col in ["T", "lambda", "best_epoch"]:
                values.append(str(value).ljust(widths[col]))
            else:
                values.append(f"{value:.6f}".ljust(widths[col]))
        print(" | ".join(values))


def save_summary(summary_rows):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "T",
        "lambda",
        "experiment_name",
        "best_epoch",
        "best_val_acc",
        "best_val_nll",
        "best_val_ece",
        "test_acc",
        "test_nll",
        "test_ece",
        "test_ts_kl",
        "test_fisher_mismatch",
        "test_energy_mismatch",
        "metrics_path",
    ]

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nsaved KD sweep summary: {OUT_PATH}")


def main():
    with open(BASE_CONFIG, "r") as f:
        base_cfg = yaml.safe_load(f)

    os.makedirs("configs/generated", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    summary_rows = []

    for temperature in TEMPERATURES:
        for lambda_kd in LAMBDAS:
            cfg = copy.deepcopy(base_cfg)

            name = (
                f"mnist_lenet_kd_"
                f"T{value_to_name(temperature)}_"
                f"l{value_to_name(lambda_kd)}"
            )

            cfg["experiment_name"] = name
            cfg["kd"]["temperature"] = temperature
            cfg["kd"]["lambda_kd"] = lambda_kd

            # Pure KD sweep: all extensions remain disabled.
            cfg.setdefault("extensions", {})
            cfg["extensions"]["fisher_alpha"] = 0.0
            cfg["extensions"]["energy_beta"] = 0.0
            cfg["extensions"]["param_fisher_gamma"] = 0.0
            cfg["extensions"]["grad_field_delta"] = 0.0

            cfg["save"]["checkpoint_path"] = f"checkpoints/{name}.pt"
            cfg["save"]["metrics_path"] = f"results/{name}.csv"

            generated_config = f"configs/generated/{name}.yaml"

            with open(generated_config, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)

            print("\n" + "=" * 80)
            print(
                f"Running MNIST LeNet KD sweep: "
                f"T={temperature}, lambda={lambda_kd}"
            )
            print("=" * 80)

            subprocess.run(
                [sys.executable, "src/train.py", "--config", generated_config],
                check=True,
            )

            summary = summarize_run(cfg["save"]["metrics_path"])
            summary["T"] = temperature
            summary["lambda"] = lambda_kd
            summary["experiment_name"] = name
            summary["metrics_path"] = cfg["save"]["metrics_path"]

            summary_rows.append(summary)

    summary_rows = sorted(summary_rows, key=lambda row: row["best_val_nll"])

    print_summary(summary_rows)
    save_summary(summary_rows)

    best = summary_rows[0]
    print(
        "\nBest by validation NLL: "
        f"T={best['T']}, lambda={best['lambda']}, "
        f"val_nll={best['best_val_nll']:.6f}, "
        f"test_nll={best['test_nll']:.6f}, "
        f"test_acc={best['test_acc']:.6f}, "
        f"test_ece={best['test_ece']:.6f}"
    )


if __name__ == "__main__":
    main()