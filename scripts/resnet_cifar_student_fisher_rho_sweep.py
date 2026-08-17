import copy
import csv
import os
import subprocess
import sys
from pathlib import Path

import yaml


BASE_CONFIG = "configs/cifar_resnet_student_kd.yaml"

RHOS = [0.1, 0.5, 1.0]
SWEEP_SEED = 42

BEST_KD_T = 1.0
BEST_KD_LAMBDA = 0.6

OUT_PATH = Path(
    "results/head_metric/rho_sweeps/"
    "cifar_resnet_student_fisher_rho_sweep_summary.csv"
)


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


def summarize_run(rho, metrics_path):
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
        "rho": rho,
        "metrics_path": metrics_path,
        "best_epoch": int(float(best_val_row["epoch"])),
        "best_val_acc": get_float(best_val_row, "val_acc"),
        "best_val_nll": get_float(best_val_row, "val_nll"),
        "best_val_ece": get_float(best_val_row, "val_ece"),
        "best_val_ts_kl": get_float(best_val_row, "val_teacher_student_kl"),
        "best_val_fisher_mismatch": get_float(best_val_row, "val_fisher_mismatch"),
        "head_direction_ratio": get_float(best_val_row, "head_direction_ratio"),
        "head_direction_cosine": get_float(best_val_row, "head_direction_cosine"),
        "cg_iterations_mean": get_float(best_val_row, "cg_iterations_mean"),
        "cg_relative_residual_mean": get_float(
            best_val_row, "cg_relative_residual_mean"
        ),
        "test_acc": get_float(test_row, "test_acc"),
        "test_nll": get_float(test_row, "test_nll"),
        "test_ece": get_float(test_row, "test_ece"),
        "test_ts_kl": get_float(test_row, "test_teacher_student_kl"),
        "test_fisher_mismatch": get_float(test_row, "test_fisher_mismatch"),
    }


def print_summary(rows):
    print("\nCIFAR-10 RESNET STUDENT-FISHER RHO SWEEP SUMMARY")
    print(
        "rho | best_epoch | val_acc | val_nll | val_ece | val_ts_kl | "
        "||d||/||g|| | cos(g,d) | CG iters | CG relres"
    )
    print("-" * 125)

    for row in rows:
        print(
            f"{row['rho']:g} | "
            f"{row['best_epoch']:>10d} | "
            f"{row['best_val_acc']:.6f} | "
            f"{row['best_val_nll']:.6f} | "
            f"{row['best_val_ece']:.6f} | "
            f"{row['best_val_ts_kl']:.6f} | "
            f"{row['head_direction_ratio']:.4f} | "
            f"{row['head_direction_cosine']:.4f} | "
            f"{row['cg_iterations_mean']:.2f} | "
            f"{row['cg_relative_residual_mean']:.3e}"
        )


def save_summary(rows):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rho",
        "metrics_path",
        "best_epoch",
        "best_val_acc",
        "best_val_nll",
        "best_val_ece",
        "best_val_ts_kl",
        "best_val_fisher_mismatch",
        "head_direction_ratio",
        "head_direction_cosine",
        "cg_iterations_mean",
        "cg_relative_residual_mean",
        "test_acc",
        "test_nll",
        "test_ece",
        "test_ts_kl",
        "test_fisher_mismatch",
    ]

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nsaved Student-Fisher rho sweep summary: {OUT_PATH}")


def main():
    with open(BASE_CONFIG, "r") as f:
        base_cfg = yaml.safe_load(f)

    os.makedirs("configs/generated/head_metric", exist_ok=True)
    os.makedirs("results/head_metric/rho_sweeps", exist_ok=True)
    os.makedirs("checkpoints/head_metric/rho_sweeps", exist_ok=True)

    summary_rows = []

    for rho in RHOS:
        cfg = copy.deepcopy(base_cfg)

        name = (
            "cifar_resnet20_kd_head_student_fisher_"
            f"rho{value_to_name(rho)}_sweep_seed{SWEEP_SEED}"
        )

        cfg["experiment_name"] = name
        cfg["seed"] = SWEEP_SEED

        # Fixed best KD baseline from the previous CIFAR-10 ResNet KD sweep
        cfg["kd"]["temperature"] = BEST_KD_T
        cfg["kd"]["lambda_kd"] = BEST_KD_LAMBDA

        # Primary objective is CE + KD only
        cfg["extensions"]["fisher_alpha"] = 0.0
        cfg["extensions"]["energy_beta"] = 0.0
        cfg["extensions"]["param_fisher_gamma"] = 0.0
        cfg["extensions"]["grad_field_delta"] = 0.0

        cfg["head_update"] = {
            "mode": "student_fisher",
            "metric_temperature": 1.0,
            "rho": rho,
            "cg_tol": 1.0e-6,
            "cg_max_iter": 50,
        }

        cfg["save"]["checkpoint_path"] = (
            f"checkpoints/head_metric/rho_sweeps/{name}.pt"
        )
        cfg["save"]["metrics_path"] = (
            f"results/head_metric/rho_sweeps/{name}.csv"
        )

        generated_config = f"configs/generated/head_metric/{name}.yaml"

        with open(generated_config, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        print("\n" + "=" * 80)
        print(
            "Running CIFAR-10 ResNet Student-Fisher rho sweep: "
            f"T={BEST_KD_T}, lambda={BEST_KD_LAMBDA}, "
            f"rho={rho}, seed={SWEEP_SEED}"
        )
        print("=" * 80)

        subprocess.run(
            [sys.executable, "src/train.py", "--config", generated_config],
            check=True,
        )

        summary_rows.append(summarize_run(rho, cfg["save"]["metrics_path"]))

    # this ordering is also the selection rule for rho
    summary_rows = sorted(summary_rows, key=lambda row: row["best_val_nll"])

    print_summary(summary_rows)
    save_summary(summary_rows)

    best = summary_rows[0]
    print(
        "\nSelected by validation NLL: "
        f"rho={best['rho']:g}, "
        f"val_nll={best['best_val_nll']:.6f}, "
        f"val_acc={best['best_val_acc']:.6f}, "
        f"val_ece={best['best_val_ece']:.6f}"
    )

if __name__ == "__main__":
    main()
