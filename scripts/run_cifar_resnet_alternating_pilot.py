import argparse
import copy
import os
import subprocess
import sys

import yaml


DEFAULT_SEEDS = [42, 43, 44]
BASE_CONFIG = "configs/cifar_resnet_student_kd.yaml"
BEST_KD_T = 1.0
BEST_KD_LAMBDA = 0.6


def value_to_name(value):
    text = f"{value:.8f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def make_runs(student_rho, teacher_rho):
    return [
        {
            "method": "ResNet-20 KD + Euclidean head [alternating]",
            "name": "cifar_resnet20_kd_head_euclidean_alternating",
            "head_update": {
                "mode": "euclidean",
                "scheme": "alternating",
                "rho": 0.0,
            },
        },
        {
            "method": (
                "ResNet-20 KD + Student-Fisher "
                f"(rho={student_rho:g}) [alternating]"
            ),
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
        {
            "method": (
                "ResNet-20 KD + Teacher-induced Fisher "
                f"(rho={teacher_rho:g}) [alternating]"
            ),
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
    ]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a paired alternating-update pilot: Euclidean, "
            "Student-Fisher, and Teacher-induced Fisher."
        )
    )
    parser.add_argument("--student-rho", type=float, default=0.1)
    parser.add_argument("--teacher-rho", type=float, default=0.1)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seeds to run (default: 42 43 44).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write configs and print commands without starting training.",
    )
    args = parser.parse_args()

    if args.student_rho <= 0.0 or args.teacher_rho <= 0.0:
        raise ValueError("Fisher rho values must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must not contain duplicates")

    runs = make_runs(args.student_rho, args.teacher_rho)
    config_dir = "configs/generated/head_metric_alternating"
    result_dir = "results/head_metric/alternating/seeds"
    checkpoint_dir = "checkpoints/head_metric/alternating/seeds"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for seed in args.seeds:
        for run in runs:
            with open(BASE_CONFIG, "r") as handle:
                cfg = yaml.safe_load(handle)

            cfg = copy.deepcopy(cfg)
            name = f"{run['name']}_seed{seed}"
            cfg["experiment_name"] = name
            cfg["seed"] = seed
            cfg["kd"]["temperature"] = BEST_KD_T
            cfg["kd"]["lambda_kd"] = BEST_KD_LAMBDA
            cfg["train"]["deterministic"] = True
            cfg["extensions"]["fisher_alpha"] = 0.0
            cfg["extensions"]["energy_beta"] = 0.0
            cfg["extensions"]["param_fisher_gamma"] = 0.0
            cfg["extensions"]["grad_field_delta"] = 0.0
            cfg["head_update"] = copy.deepcopy(run["head_update"])
            cfg["save"]["checkpoint_path"] = f"{checkpoint_dir}/{name}.pt"
            cfg["save"]["metrics_path"] = f"{result_dir}/{name}.csv"

            generated_config = f"{config_dir}/{name}.yaml"
            with open(generated_config, "w") as handle:
                yaml.safe_dump(cfg, handle, sort_keys=False)

            command = [sys.executable, "src/train.py", "--config", generated_config]
            print("\n" + "=" * 80)
            print(f"{run['method']} | seed={seed}")
            print(" ".join(command))
            if not args.dry_run:
                subprocess.run(command, check=True)

    print(
        "\nAlternating pilot finished "
        f"({len(args.seeds)} seeds x {len(runs)} methods)."
    )
    print(
        "Summarize with:\n"
        "python scripts/summarize_cifar_resnet_alternating_pilot.py "
        f"--student-rho {args.student_rho:g} "
        f"--teacher-rho {args.teacher_rho:g} "
        "--seeds " + " ".join(str(seed) for seed in args.seeds)
    )


if __name__ == "__main__":
    main()
