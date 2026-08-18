#!/usr/bin/env python3
"""Summarize final alternating runs after rho selection is locked."""

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev


DEFAULT_SEEDS = [42, 43, 44, 45, 46]
RESULT_DIR = Path("results/head_metric/alternating/seeds")
OUT_PATH = Path(
    "results/head_metric/alternating/"
    "cifar_resnet_alternating_final_seeds_summary.csv"
)


def value_to_name(value):
    text = f"{value:.8f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def get_float(row, key, default=0.0):
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def summarize_file(path):
    with path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    test_rows = [row for row in rows if row.get("phase") == "test"]
    if not test_rows:
        raise RuntimeError(f"No test row found in {path}")

    test = test_rows[-1]
    return {
        "best_epoch": int(float(test.get("best_epoch", 0))),
        "test_acc": get_float(test, "test_acc"),
        "test_nll": get_float(test, "test_nll"),
        "test_ece": get_float(test, "test_ece"),
        "test_ts_kl": get_float(test, "test_teacher_student_kl"),
        "test_fisher_mismatch": get_float(test, "test_fisher_mismatch"),
        "test_energy_mismatch": get_float(test, "test_energy_mismatch"),
    }


def mean_std(values):
    if len(values) == 1:
        return mean(values), 0.0
    return mean(values), stdev(values)


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
    parser = argparse.ArgumentParser(
        description="Aggregate the locked-rho alternating five-seed comparison."
    )
    parser.add_argument("--student-rho", type=float, required=True)
    parser.add_argument("--teacher-rho", type=float, required=True)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write a provisional summary even if some expected runs are missing.",
    )
    args = parser.parse_args()

    if args.student_rho <= 0.0 or args.teacher_rho <= 0.0:
        raise ValueError("Selected Fisher rho values must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must not contain duplicates")

    summary_rows = []
    missing = []
    for run in make_runs(args.student_rho, args.teacher_rho):
        seed_rows = []
        for seed in args.seeds:
            path = RESULT_DIR / f"{run['base_name']}_seed{seed}.csv"
            if not path.exists():
                missing.append(str(path))
                continue
            try:
                row = summarize_file(path)
            except RuntimeError:
                missing.append(f"{path} (no test row)")
                continue
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
            values = [row[metric] for row in seed_rows]
            metric_mean, metric_std = mean_std(values)
            out[f"{metric}_mean"] = metric_mean
            out[f"{metric}_std"] = metric_std
        summary_rows.append(out)

    if missing and not args.allow_missing:
        print("Final summary not written because expected runs are missing:")
        for item in missing:
            print(f"  - {item}")
        raise SystemExit(2)

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
    with OUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nCIFAR-10 ResNet alternating final five-seed summary")
    print("method | rho | n | test acc mean +/- std | test NLL | test ECE")
    print("-" * 115)
    for row in summary_rows:
        print(
            f"{row['method']} | "
            f"{float(row['rho']):g} | "
            f"{row['n_seeds']} | "
            f"{float(row['test_acc_mean']):.4f} +/- "
            f"{float(row['test_acc_std']):.4f} | "
            f"{float(row['test_nll_mean']):.4f} +/- "
            f"{float(row['test_nll_std']):.4f} | "
            f"{float(row['test_ece_mean']):.4f} +/- "
            f"{float(row['test_ece_std']):.4f}"
        )

    if missing:
        print(f"\nWARNING: provisional summary; {len(missing)} run(s) missing.")
    print(f"\nsaved summary: {OUT_PATH}")


if __name__ == "__main__":
    main()
