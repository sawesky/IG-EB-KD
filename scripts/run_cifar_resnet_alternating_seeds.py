#!/usr/bin/env python3
"""Run the final five-seed alternating head-geometry comparison safely."""

import argparse
import copy
import csv
import subprocess
import sys
from pathlib import Path

import yaml


BASE_CONFIG = "configs/cifar_resnet_student_kd.yaml"
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_METHODS = ["euclidean", "student", "teacher"]
BEST_KD_T = 1.0
BEST_KD_LAMBDA = 0.6


def value_to_name(value):
    text = f"{value:.8f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def metrics_status(path):
    """Return missing, complete, or incomplete for one training CSV."""
    path = Path(path)
    if not path.exists():
        return "missing"

    try:
        with path.open("r", newline="") as handle:
            if any(row.get("phase") == "test" for row in csv.DictReader(handle)):
                return "complete"
    except (OSError, csv.Error):
        pass
    return "incomplete"


def make_runs(student_rho, teacher_rho):
    return {
        "euclidean": {
            "label": "Euclidean head",
            "name": "cifar_resnet20_kd_head_euclidean_alternating",
            "head_update": {
                "mode": "euclidean",
                "scheme": "alternating",
                "rho": 0.0,
            },
        },
        "student": {
            "label": f"Student-Fisher head (rho={student_rho:g})",
            "name": (
                "cifar_resnet20_kd_head_student_fisher_"
                f"rho{value_to_name(student_rho)}_alternating"
            ),
            "head_update": {
                "mode": "student_fisher",
                "scheme": "alternating",
                "metric_temperature": 1.0,
                "rho": student_rho,
                "cg_tol": 1.0e-6,
                "cg_max_iter": 50,
            },
        },
        "teacher": {
            "label": f"Teacher-induced Fisher head (rho={teacher_rho:g})",
            "name": (
                "cifar_resnet20_kd_head_teacher_fisher_"
                f"rho{value_to_name(teacher_rho)}_alternating"
            ),
            "head_update": {
                "mode": "teacher_fisher",
                "scheme": "alternating",
                "metric_temperature": 1.0,
                "rho": teacher_rho,
                "cg_tol": 1.0e-6,
                "cg_max_iter": 50,
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the final alternating Euclidean, Student-Fisher, and "
            "Teacher-induced Fisher comparison. Completed runs are reused."
        )
    )
    parser.add_argument(
        "--student-rho",
        type=float,
        required=True,
        help="Student-Fisher rho selected from the seed-42 validation sweep.",
    )
    parser.add_argument(
        "--teacher-rho",
        type=float,
        required=True,
        help="Teacher-Fisher rho selected from the seed-42 validation sweep.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Final seeds (default: 42 43 44 45 46).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=DEFAULT_METHODS,
        default=DEFAULT_METHODS,
        help="Methods to run (default: euclidean student teacher).",
    )
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Explicitly rerun and overwrite existing complete/incomplete runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write configs and print commands without training.",
    )
    args = parser.parse_args()

    if args.student_rho <= 0.0 or args.teacher_rho <= 0.0:
        raise ValueError("Selected Fisher rho values must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must not contain duplicates")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("Methods must not contain duplicates")

    all_runs = make_runs(args.student_rho, args.teacher_rho)
    runs = [all_runs[method] for method in args.methods]
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
    for seed in args.seeds:
        for run in runs:
            name = f"{run['name']}_seed{seed}"
            metrics_path = result_dir / f"{name}.csv"
            checkpoint_path = checkpoint_dir / f"{name}.pt"
            generated_config = config_dir / f"{name}.yaml"

            cfg = copy.deepcopy(base_cfg)
            cfg["experiment_name"] = name
            cfg["seed"] = seed
            cfg["kd"]["temperature"] = BEST_KD_T
            cfg["kd"]["lambda_kd"] = BEST_KD_LAMBDA

            # Match the user's current code: no deterministic-mode injection.
            cfg["train"].pop("deterministic", None)

            # The common primary objective is CE + KD only.
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
            print(f"Alternating {run['label']} | seed={seed}")

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
        f"\nFinal runs finished: scheduled={scheduled}, "
        f"skipped={skipped}, blocked={blocked}."
    )
    if blocked:
        print(
            "One or more incomplete CSVs were left untouched. Inspect them, then "
            "rerun intentionally with --rerun-existing if appropriate."
        )
    print(
        "Summarize the five final seeds with:\n"
        "python scripts/summarize_cifar_resnet_alternating_seeds.py "
        f"--student-rho {args.student_rho:g} "
        f"--teacher-rho {args.teacher_rho:g} "
        "--seeds " + " ".join(str(seed) for seed in args.seeds)
    )


if __name__ == "__main__":
    main()
