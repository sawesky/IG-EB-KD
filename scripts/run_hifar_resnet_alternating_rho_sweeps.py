#!/usr/bin/env python3
"""Run validation-tuning rho sweeps with a locked manual head step size."""

import argparse
import copy
import csv
import math
import subprocess
import sys
from pathlib import Path

import yaml


BASE_CONFIG = "configs/hifar_resnet_student_kd.yaml"
DEFAULT_RHOS = [0.1, 0.5, 1.0]
DEFAULT_SEED = 42
BEST_KD_T = 1.0
BEST_KD_LAMBDA = 0.7
HEAD_LR = 0.04


def value_to_name(value):
    text = f"{value:.8f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def metrics_status(path):
    """Classify one CSV, including its manual-head LR provenance."""
    path = Path(path)
    if not path.exists():
        return "missing"

    try:
        with path.open("r", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return "incomplete"

    if not any(row.get("phase") == "test" for row in rows):
        return "incomplete"

    val_rows = [row for row in rows if row.get("phase") == "val"]
    try:
        recorded_head_lrs = [float(row["head_lr"]) for row in val_rows]
    except (KeyError, TypeError, ValueError):
        return "incompatible"

    if val_rows and all(
        math.isclose(lr, HEAD_LR, rel_tol=1e-9, abs_tol=1e-12)
        for lr in recorded_head_lrs
    ):
        return "complete"
    return "incompatible"


def make_run(metric, rho):
    if metric == "student":
        label = "Student-Fisher"
        mode = "student_fisher"
        name = (
            "hifar_resnet20_kd_head_student_fisher_"
            f"rho{value_to_name(rho)}_alternating"
        )
    elif metric == "teacher":
        label = "Teacher-induced Fisher"
        mode = "teacher_fisher"
        name = (
            "hifar_resnet20_kd_head_teacher_fisher_"
            f"rho{value_to_name(rho)}_alternating"
        )
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return {
        "label": label,
        "name": name,
        "head_update": {
            "mode": mode,
            "scheme": "alternating",
            "metric_temperature": 1.0,
            "rho": rho,
            "lr": HEAD_LR,
            "cg_tol": 1.0e-6,
            "cg_max_iter": 50,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run seed-42 validation sweeps for alternating Student-Fisher "
            "and Teacher-induced Fisher. Existing completed runs are skipped."
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
        help="Rho grid (default: 0.1 0.5 1.0).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Rerun and overwrite completed metric files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write configs and print commands without training.",
    )
    args = parser.parse_args()

    if any(rho <= 0.0 for rho in args.rhos):
        raise ValueError("All Fisher rho values must be positive")
    if len(set(args.rhos)) != len(args.rhos):
        raise ValueError("Rho values must not contain duplicates")

    metrics = ["student", "teacher"] if args.metric == "both" else [args.metric]
    config_dir = Path("configs/generated/head_metric_alternating")
    result_dir = Path("results/head_metric/alternating/seeds")
    checkpoint_dir = Path("checkpoints/head_metric/alternating/seeds")
    for directory in (config_dir, result_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)

    with open(BASE_CONFIG, "r") as handle:
        base_cfg = yaml.safe_load(handle)

    scheduled = 0
    skipped = 0
    blocked = 0
    replaced = 0
    for metric in metrics:
        for rho in args.rhos:
            run = make_run(metric, rho)
            name = f"{run['name']}_seed{args.seed}"
            metrics_path = result_dir / f"{name}.csv"
            checkpoint_path = checkpoint_dir / f"{name}.pt"
            generated_config = config_dir / f"{name}.yaml"

            cfg = copy.deepcopy(base_cfg)
            cfg["experiment_name"] = name
            cfg["seed"] = args.seed
            cfg["kd"]["temperature"] = BEST_KD_T
            cfg["kd"]["lambda_kd"] = BEST_KD_LAMBDA
            cfg["train"].pop("deterministic", None)
            cfg["extensions"]["fisher_alpha"] = 0.0
            cfg["extensions"]["energy_beta"] = 0.0
            cfg["extensions"]["param_fisher_gamma"] = 0.0
            cfg["extensions"]["grad_field_delta"] = 0.0
            cfg["head_update"] = copy.deepcopy(run["head_update"])
            cfg["save"]["checkpoint_path"] = str(checkpoint_path)
            cfg["save"]["metrics_path"] = str(metrics_path)

            with generated_config.open("w") as handle:
                yaml.safe_dump(cfg, handle, sort_keys=False)

            print("\n" + "=" * 80)
            print(
                f"Alternating {run['label']} | rho={rho:g} | "
                f"constant head LR={HEAD_LR:g} | tuning seed={args.seed}"
            )

            status = metrics_status(metrics_path)
            if status == "complete" and not args.rerun_existing:
                print(f"SKIP completed: {metrics_path}")
                skipped += 1
                continue
            if status == "incomplete" and not args.rerun_existing:
                print(f"BLOCKED incomplete CSV: {metrics_path}")
                print("Use --rerun-existing only if you intend to overwrite it.")
                blocked += 1
                continue
            if status == "incompatible":
                print(
                    "REPLACE old completed CSV: it was not produced with "
                    f"constant head LR={HEAD_LR:g}: {metrics_path}"
                )
                replaced += 1

            command = [
                sys.executable,
                "src/train.py",
                "--config",
                str(generated_config),
            ]
            print(" ".join(command))
            scheduled += 1
            if not args.dry_run:
                subprocess.run(command, check=True)

    print(
        f"\nSweep finished: scheduled={scheduled}, skipped={skipped}, "
        f"replacing_old={replaced}, blocked={blocked}.\n"
        "Select rho using validation only:\n"
        "python scripts/summarize_hifar_resnet_alternating_rho_sweeps.py "
        f"--rhos {' '.join(f'{rho:g}' for rho in args.rhos)} "
        f"--seed {args.seed}"
    )

if __name__ == "__main__":
    main()
