import copy
import csv
import os
import subprocess
import sys
from pathlib import Path

import yaml


BASE_CONFIG = "configs/cifar_wrn_student_kd_energy.yaml"

BETAS = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

BEST_KD_T = 1.5
BEST_KD_LAMBDA = 0.8

OUT_PATH = Path("results/cifar_wrn_energy_sweep_summary.csv")


def value_to_name(value):
    text = f"{value:.8f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def read_rows(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def get_float(row, key, default=0.0):
    value = row.get(key, "")
    if value == "" or value is None:
        return default
    return float(value)


def summarize_run(beta, metrics_path):
    rows = read_rows(metrics_path)

    val_rows = [row for row in rows if row.get("phase") == "val"]
    test_rows = [row for row in rows if row.get("phase") == "test"]

    if not val_rows:
        raise RuntimeError(f"No validation rows found in {metrics_path}")
    if not test_rows:
        raise RuntimeError(f"No test row found in {metrics_path}")

    best_val_row = min(val_rows, key=lambda row: get_float(row, "val_nll"))
    test_row = test_rows[-1]

    return {
        "beta": beta,
        "metrics_path": metrics_path,
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


def print_summary(rows):
    columns = [
        "beta",
        "best_epoch",
        "best_val_acc",
        "best_val_nll",
        "best_val_ece",
        "best_val_energy_mismatch",
        "test_acc",
        "test_nll",
        "test_ece",
        "test_energy_mismatch",
    ]

    widths = {
        "beta": 10,
        "best_epoch": 10,
        "best_val_acc": 14,
        "best_val_nll": 14,
        "best_val_ece": 14,
        "best_val_energy_mismatch": 24,
        "test_acc": 10,
        "test_nll": 10,
        "test_ece": 10,
        "test_energy_mismatch": 22,
    }

    print("\nCIFAR-10 WRN ENERGY SWEEP SUMMARY")
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    print("-" * len(header))

    for row in rows:
        values = []
        for col in columns:
            value = row[col]

            if col == "best_epoch":
                values.append(str(value).ljust(widths[col]))
            elif col == "beta":
                values.append(f"{value:g}".ljust(widths[col]))
            else:
                values.append(f"{value:.6f}".ljust(widths[col]))

        print(" | ".join(values))


def save_summary(rows):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "beta",
        "metrics_path",
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

    print(f"\nsaved CIFAR-10 WRN Energy sweep summary: {OUT_PATH}")


def main():
    with open(BASE_CONFIG, "r") as f:
        base_cfg = yaml.safe_load(f)

    os.makedirs("configs/generated", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    summary_rows = []

    for beta in BETAS:
        cfg = copy.deepcopy(base_cfg)

        name = f"cifar_wrn16_2_kd_energy_T{value_to_name(BEST_KD_T)}_l{value_to_name(BEST_KD_LAMBDA)}_b{value_to_name(beta)}"

        cfg["experiment_name"] = name

        # Fixed best KD baseline from KD sweep.
        cfg["kd"]["temperature"] = BEST_KD_T
        cfg["kd"]["lambda_kd"] = BEST_KD_LAMBDA

        # Energy sweep only.
        cfg["extensions"]["fisher_alpha"] = 0.0
        cfg["extensions"]["energy_beta"] = beta
        cfg["extensions"]["param_fisher_gamma"] = 0.0
        cfg["extensions"]["grad_field_delta"] = 0.0

        cfg["save"]["checkpoint_path"] = f"checkpoints/{name}.pt"
        cfg["save"]["metrics_path"] = f"results/{name}.csv"

        generated_config = f"configs/generated/{name}.yaml"

        with open(generated_config, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        print("\n" + "=" * 80)
        print(
            f"Running CIFAR-10 WRN Energy sweep: "
            f"T={BEST_KD_T}, lambda={BEST_KD_LAMBDA}, beta={beta}"
        )
        print("=" * 80)

        subprocess.run(
            [sys.executable, "src/train.py", "--config", generated_config],
            check=True,
        )

        summary_rows.append(summarize_run(beta, cfg["save"]["metrics_path"]))

    summary_rows = sorted(summary_rows, key=lambda row: row["best_val_nll"])

    print_summary(summary_rows)
    save_summary(summary_rows)

    best_nll = summary_rows[0]
    best_energy = min(summary_rows, key=lambda row: row["best_val_energy_mismatch"])

    print(
        "\nBest by validation NLL: "
        f"beta={best_nll['beta']:g}, "
        f"val_nll={best_nll['best_val_nll']:.6f}, "
        f"test_nll={best_nll['test_nll']:.6f}, "
        f"test_acc={best_nll['test_acc']:.6f}, "
        f"test_ece={best_nll['test_ece']:.6f}"
    )

    print(
        "Best by validation Energy mismatch: "
        f"beta={best_energy['beta']:g}, "
        f"val_energy={best_energy['best_val_energy_mismatch']:.6f}, "
        f"test_energy={best_energy['test_energy_mismatch']:.6f}, "
        f"test_nll={best_energy['test_nll']:.6f}, "
        f"test_acc={best_energy['test_acc']:.6f}"
    )


if __name__ == "__main__":
    main()