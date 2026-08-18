#!/usr/bin/env python3
"""Summarize the paired alternating-update CIFAR-10 ResNet pilot."""

import argparse
import csv
from pathlib import Path

from summarize_cifar_resnet_head_metric_seeds import (
    add_metric_summary,
    summarize_file,
    value_to_name,
)


DEFAULT_SEEDS = [42, 43, 44]
RESULT_DIR = Path("results/head_metric/alternating/seeds")
OUT_PATH = Path(
    "results/head_metric/alternating/"
    "cifar_resnet_alternating_pilot_summary.csv"
)


def make_runs(student_rho, teacher_rho):
    return [
        {
            "method": "ResNet-20 KD + Euclidean head [alternating]",
            "rho": 0.0,
            "base_name": "cifar_resnet20_kd_head_euclidean_alternating",
        },
        {
            "method": "ResNet-20 KD + Student-Fisher [alternating]",
            "rho": student_rho,
            "base_name": (
                "cifar_resnet20_kd_head_student_fisher_"
                f"rho{value_to_name(student_rho)}_alternating"
            ),
        },
        {
            "method": "ResNet-20 KD + Teacher-induced Fisher [alternating]",
            "rho": teacher_rho,
            "base_name": (
                "cifar_resnet20_kd_head_teacher_fisher_"
                f"rho{value_to_name(teacher_rho)}_alternating"
            ),
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-rho", type=float, default=0.1)
    parser.add_argument("--teacher-rho", type=float, default=0.1)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )
    args = parser.parse_args()

    if args.student_rho <= 0.0 or args.teacher_rho <= 0.0:
        raise ValueError("Fisher rho values must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must not contain duplicates")

    summary_rows = []
    for run in make_runs(args.student_rho, args.teacher_rho):
        seed_rows = []
        for seed in args.seeds:
            path = RESULT_DIR / f"{run['base_name']}_seed{seed}.csv"
            if not path.exists():
                print(f"missing: {path}")
                continue

            row = summarize_file(path)
            row["seed"] = seed
            seed_rows.append(row)

        if not seed_rows:
            continue

        out = {
            "method": run["method"],
            "rho": run["rho"],
            "n_seeds": len(seed_rows),
        }
        for metric in (
            "best_epoch",
            "test_acc",
            "test_nll",
            "test_ece",
            "test_ts_kl",
            "test_fisher_mismatch",
            "test_energy_mismatch",
        ):
            add_metric_summary(out, seed_rows, metric)
        summary_rows.append(out)

    fieldnames = [
        "method",
        "rho",
        "n_seeds",
        "best_epoch_mean",
        "best_epoch_std",
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
    with open(OUT_PATH, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nCIFAR-10 ResNet alternating pilot summary")
    print("method | rho | n | acc mean_std | nll mean_std | ece mean_std")
    print("-" * 110)
    for row in summary_rows:
        print(
            f"{row['method']} | "
            f"{float(row['rho']):g} | "
            f"{row['n_seeds']} | "
            f"{float(row['test_acc_mean']):.4f}_"
            f"{float(row['test_acc_std']):.4f} | "
            f"{float(row['test_nll_mean']):.4f}_"
            f"{float(row['test_nll_std']):.4f} | "
            f"{float(row['test_ece_mean']):.4f}_"
            f"{float(row['test_ece_std']):.4f}"
        )

    print(f"\nsaved summary: {OUT_PATH}")


if __name__ == "__main__":
    main()
