import copy
import os
import subprocess
import sys

import yaml


SEEDS = [42, 43, 44]

RUNS = [
    {
        "method": "ResNet-20 CE",
        "base_config": "configs/cifar_resnet_student_ce.yaml",
        "name": "cifar_resnet20_student_ce",
        "kd": None,
        "extensions": None,
    },
    {
        "method": "ResNet-20 KD",
        "base_config": "configs/cifar_resnet_student_kd.yaml",
        "name": "cifar_resnet20_student_kd",
        "kd": {"temperature": 1.0, "lambda_kd": 0.6},
        "extensions": {
            "fisher_alpha": 0.0,
            "energy_beta": 0.0,
            "param_fisher_gamma": 0.0,
            "grad_field_delta": 0.0,
        },
    },
    {
        "method": "ResNet-20 KD + Fisher",
        "base_config": "configs/cifar_resnet_student_kd_fisher.yaml",
        "name": "cifar_resnet20_student_kd_fisher",
        "kd": {"temperature": 1.0, "lambda_kd": 0.6},
        "extensions": {
            "fisher_alpha": 0.01,
            "energy_beta": 0.0,
            "param_fisher_gamma": 0.0,
            "grad_field_delta": 0.0,
        },
    },
    {
        "method": "ResNet-20 KD + Energy",
        "base_config": "configs/cifar_resnet_student_kd_energy.yaml",
        "name": "cifar_resnet20_student_kd_energy",
        "kd": {"temperature": 1.0, "lambda_kd": 0.6},
        "extensions": {
            "fisher_alpha": 0.0,
            "energy_beta": 0.01,
            "param_fisher_gamma": 0.0,
            "grad_field_delta": 0.0,
        },
    },
    {
        "method": "ResNet-20 KD + Fisher + Energy",
        "base_config": "configs/cifar_resnet_student_kd_fisher_energy.yaml",
        "name": "cifar_resnet20_student_kd_fisher_energy",
        "kd": {"temperature": 1.0, "lambda_kd": 0.6},
        "extensions": {
            "fisher_alpha": 0.01,
            "energy_beta": 0.01,
            "param_fisher_gamma": 0.0,
            "grad_field_delta": 0.0,
        },
    },
]


def main():
    os.makedirs("configs/generated", exist_ok=True)
    os.makedirs("results/seeds", exist_ok=True)
    os.makedirs("checkpoints/seeds", exist_ok=True)

    for seed in SEEDS:
        for run in RUNS:
            with open(run["base_config"], "r") as f:
                cfg = yaml.safe_load(f)

            name = f"{run['name']}_seed{seed}"

            cfg["experiment_name"] = name

            cfg["seed"] = seed

            if "data" not in cfg:
                cfg["data"] = {}
            cfg["data"]["seed"] = seed

            if "train" not in cfg:
                cfg["train"] = {}
            cfg["train"]["seed"] = seed

            if run["kd"] is not None:
                cfg["kd"]["temperature"] = run["kd"]["temperature"]
                cfg["kd"]["lambda_kd"] = run["kd"]["lambda_kd"]

            if run["extensions"] is not None:
                for key, value in run["extensions"].items():
                    cfg["extensions"][key] = value

            cfg["save"]["checkpoint_path"] = f"checkpoints/seeds/{name}.pt"
            cfg["save"]["metrics_path"] = f"results/seeds/{name}.csv"

            generated_config = f"configs/generated/{name}.yaml"

            with open(generated_config, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)

            print("\n" + "=" * 80)
            print(f"Running {run['method']} | seed={seed}")
            print("=" * 80)

            subprocess.run(
                [sys.executable, "src/train.py", "--config", generated_config],
                check=True,
            )

    print("\nFinal ResNet CIFAR-10 multi-seed runs finished.")


if __name__ == "__main__":
    main()