import argparse
import copy
import os
import subprocess
import sys

import yaml


SEEDS = [42, 43, 44, 45, 46]

BASE_CONFIG = "configs/cifar_resnet_student_kd.yaml"
BEST_KD_T = 1.0
BEST_KD_LAMBDA = 0.6


def value_to_name(value):
    text = f"{value:.8f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def make_runs(student_rho, teacher_rho):
    return [
        {
            "method": "ResNet-20 KD + Euclidean head",
            "name": "cifar_resnet20_kd_head_euclidean",
            # rho=0 is the conceptual Euclidean case; the code uses the exact
            # d = g fast path rather than invoking CG with rho=0 
            "head_update": {
                "mode": "euclidean",
                "rho": 0.0,
            },
        },
        {
            "method": f"ResNet-20 KD + Student-Fisher (rho={student_rho:g})",
            "name": (
                "cifar_resnet20_kd_head_student_fisher_"
                f"rho{value_to_name(student_rho)}"
            ),
            "head_update": {
                "mode": "student_fisher",
                "metric_temperature": 1.0,
                "rho": student_rho,
                "cg_tol": 1.0e-6,
                "cg_max_iter": 50,
            },
        },
        {
            "method": (
                "ResNet-20 KD + Teacher-induced Fisher "
                f"(rho={teacher_rho:g})"
            ),
            "name": (
                "cifar_resnet20_kd_head_teacher_fisher_"
                f"rho{value_to_name(teacher_rho)}"
            ),
            "head_update": {
                "mode": "teacher_fisher",
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
            "Run final 5-seed CIFAR-10 ResNet head-geometry experiments after "
            "rho has been selected separately for Student-Fisher and "
            "Teacher-induced Fisher."
        )
    )
    parser.add_argument(
        "--student-rho",
        type=float,
        required=True,
        help="Student-Fisher rho selected by validation NLL.",
    )
    parser.add_argument(
        "--teacher-rho",
        type=float,
        required=True,
        help="Teacher-induced Fisher rho selected by validation NLL.",
    )
    args = parser.parse_args()

    if args.student_rho <= 0.0 or args.teacher_rho <= 0.0:
        raise ValueError("Selected Fisher rho values must be positive")

    runs = make_runs(args.student_rho, args.teacher_rho)

    os.makedirs("configs/generated/head_metric", exist_ok=True)
    os.makedirs("results/head_metric/seeds", exist_ok=True)
    os.makedirs("checkpoints/head_metric/seeds", exist_ok=True)

    for seed in SEEDS:
        for run in runs:
            with open(BASE_CONFIG, "r") as f:
                cfg = yaml.safe_load(f)

            cfg = copy.deepcopy(cfg)
            name = f"{run['name']}_seed{seed}"

            cfg["experiment_name"] = name
            cfg["seed"] = seed

            # Fixed best KD baseline from the previous CIFAR-10 ResNet KD sweep
            cfg["kd"]["temperature"] = BEST_KD_T
            cfg["kd"]["lambda_kd"] = BEST_KD_LAMBDA

            # Primary objective is CE + KD only for all three controls
            cfg["extensions"]["fisher_alpha"] = 0.0
            cfg["extensions"]["energy_beta"] = 0.0
            cfg["extensions"]["param_fisher_gamma"] = 0.0
            cfg["extensions"]["grad_field_delta"] = 0.0

            cfg["head_update"] = copy.deepcopy(run["head_update"])

            cfg["save"]["checkpoint_path"] = (
                f"checkpoints/head_metric/seeds/{name}.pt"
            )
            cfg["save"]["metrics_path"] = (
                f"results/head_metric/seeds/{name}.csv"
            )

            generated_config = f"configs/generated/head_metric/{name}.yaml"

            with open(generated_config, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)

            print("\n" + "=" * 80)
            print(f"Running {run['method']} | seed={seed}")
            print("=" * 80)

            subprocess.run(
                [sys.executable, "src/train.py", "--config", generated_config],
                check=True,
            )

    print("\nFinal CIFAR-10 ResNet head-metric 5-seed runs finished.")
    print(
        "Summarize with:\n"
        "python scripts/summarize_cifar_resnet_head_metric_seeds.py "
        f"--student-rho {args.student_rho:g} "
        f"--teacher-rho {args.teacher_rho:g}"
    )


if __name__ == "__main__":
    main()
