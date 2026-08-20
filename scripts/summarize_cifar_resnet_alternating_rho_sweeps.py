#!/usr/bin/env python3
"""Select rho using validation NLL, then report validation and test metrics."""

import argparse
import csv
from pathlib import Path


DEFAULT_RHOS = [0.1, 0.5, 1.0]
DEFAULT_SEED = 42
DEFAULT_TIE_TOLERANCE = 0.005
RESULT_DIR = Path("results/head_metric/alternating/seeds")
OUT_PATH = Path(
    "results/head_metric/alternating/rho_sweeps/"
    "cifar_resnet_alternating_rho_sweeps_validation_summary.csv"
)


def value_to_name(value):
    text = f"{value:.8f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def get_float(row, key, default=0.0):
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def metrics_path(metric, rho, seed):
    if metric == "student":
        stem = "cifar_resnet20_kd_head_student_fisher"
    elif metric == "teacher":
        stem = "cifar_resnet20_kd_head_teacher_fisher"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return RESULT_DIR / (
        f"{stem}_rho{value_to_name(rho)}_alternating_seed{seed}.csv"
    )


def summarize_run(metric, rho, seed):
    path = metrics_path(metric, rho, seed)
    if not path.exists():
        print(f"missing: {path}")
        return None

    with path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))

    val_rows = [row for row in rows if row.get("phase") == "val"]
    test_rows = [row for row in rows if row.get("phase") == "test"]
    if not val_rows:
        print(f"no validation rows: {path}")
        return None
    if not test_rows:
        print(f"no test row: {path}")
        return None

    best = min(val_rows, key=lambda row: get_float(row, "val_nll"))
    test = test_rows[-1]
    return {
        "metric": metric,
        "rho": rho,
        "seed": seed,
        "metrics_path": str(path),
        "best_epoch": int(float(best["epoch"])),
        "best_val_acc": get_float(best, "val_acc"),
        "best_val_nll": get_float(best, "val_nll"),
        "best_val_ece": get_float(best, "val_ece"),
        "best_val_ts_kl": get_float(best, "val_teacher_student_kl"),
        "best_val_fisher_mismatch": get_float(best, "val_fisher_mismatch"),
        "best_val_energy_mismatch": get_float(best, "val_energy_mismatch"),
        "test_acc": get_float(test, "test_acc"),
        "test_nll": get_float(test, "test_nll"),
        "test_ece": get_float(test, "test_ece"),
        "test_ts_kl": get_float(test, "test_teacher_student_kl"),
        "test_fisher_mismatch": get_float(test, "test_fisher_mismatch"),
        "test_energy_mismatch": get_float(test, "test_energy_mismatch"),
        "head_loss_before": get_float(best, "head_loss_before"),
        "head_loss_after": get_float(best, "head_loss_after"),
        "head_loss_decrease": get_float(best, "head_loss_decrease"),
        "head_direction_ratio": get_float(best, "head_direction_ratio"),
        "head_direction_cosine": get_float(best, "head_direction_cosine"),
        "cg_iterations_mean": get_float(best, "cg_iterations_mean"),
        "cg_relative_residual_mean": get_float(
            best,
            "cg_relative_residual_mean",
        ),
        "selected": False,
    }


def print_table(metric, rows, tie_tolerance):
    label = "STUDENT-FISHER" if metric == "student" else "TEACHER-INDUCED FISHER"
    print(f"\nCIFAR-10 RESNET ALTERNATING {label} RHO SWEEP")
    print(
        "rho | epoch | val_acc | val_nll | val_ece | val_ts_kl | "
        "||d||/||g|| | cos(g,d) | CG iters | CG relres"
    )
    print("-" * 125)
    for row in sorted(rows, key=lambda item: item["rho"]):
        print(
            f"{row['rho']:g} | "
            f"{row['best_epoch']:>5d} | "
            f"{row['best_val_acc']:.6f} | "
            f"{row['best_val_nll']:.6f} | "
            f"{row['best_val_ece']:.6f} | "
            f"{row['best_val_ts_kl']:.6f} | "
            f"{row['head_direction_ratio']:.4f} | "
            f"{row['head_direction_cosine']:.4f} | "
            f"{row['cg_iterations_mean']:.2f} | "
            f"{row['cg_relative_residual_mean']:.3e}"
        )

    raw_best = min(rows, key=lambda item: item["best_val_nll"])
    tied = [
        row
        for row in rows
        if row["best_val_nll"] <= raw_best["best_val_nll"] + tie_tolerance
    ]
    selected = min(tied, key=lambda item: item["rho"])
    selected["selected"] = True

    print(
        f"Raw minimum: rho={raw_best['rho']:g}, "
        f"val_nll={raw_best['best_val_nll']:.6f}"
    )
    print(
        f"Selected with tie tolerance {tie_tolerance:g}: "
        f"rho={selected['rho']:g}, "
        f"val_nll={selected['best_val_nll']:.6f}"
    )
    print(
        "Selected-run test report (not used for rho selection): "
        f"acc={selected['test_acc']:.6f}, "
        f"nll={selected['test_nll']:.6f}, "
        f"ece={selected['test_ece']:.6f}"
    )


def save_summary(rows):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "metric",
        "rho",
        "seed",
        "selected",
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
        "head_loss_before",
        "head_loss_after",
        "head_loss_decrease",
        "head_direction_ratio",
        "head_direction_cosine",
        "cg_iterations_mean",
        "cg_relative_residual_mean",
    ]
    with OUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(
        "\nsaved summary (rho selected by validation NLL; test metrics "
        f"reported only): {OUT_PATH}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Select alternating rho using validation NLL and report the final "
            "test metrics without using them for selection."
        )
    )
    parser.add_argument(
        "--metric",
        choices=["student", "teacher", "both"],
        default="both",
    )
    parser.add_argument(
        "--rhos",
        nargs="+",
        type=float,
        default=DEFAULT_RHOS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--tie-tolerance",
        type=float,
        default=DEFAULT_TIE_TOLERANCE,
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Print a provisional comparison before the full grid is complete.",
    )
    args = parser.parse_args()

    if args.tie_tolerance < 0.0:
        raise ValueError("Tie tolerance must be non-negative")

    metrics = ["student", "teacher"] if args.metric == "both" else [args.metric]
    all_rows = []
    incomplete = []
    for metric in metrics:
        rows = [
            row
            for rho in args.rhos
            if (row := summarize_run(metric, rho, args.seed)) is not None
        ]
        if not rows:
            print(f"\nNo completed {metric} sweep runs found")
            incomplete.append(metric)
            continue
        if len(rows) != len(args.rhos):
            incomplete.append(metric)
            if not args.allow_missing:
                print(
                    f"\n{metric} grid incomplete: found {len(rows)} of "
                    f"{len(args.rhos)} requested rho runs"
                )
                continue
        print_table(metric, rows, args.tie_tolerance)
        all_rows.extend(rows)

    if incomplete and not args.allow_missing:
        print(
            "\nNo rho selection was saved. Finish every requested run, or use "
            "--allow-missing only for a provisional inspection."
        )
        raise SystemExit(2)

    if all_rows:
        save_summary(all_rows)


if __name__ == "__main__":
    main()
