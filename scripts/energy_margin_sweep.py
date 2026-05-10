import copy
import os
import sys
import subprocess
import yaml

BASE_CONFIG = "configs/mnist_student_kd_energy.yaml"
BETAS = [0.001, 0.01, 0.1, 1.0]

def beta_to_name(beta):
    return str(beta).replace(".", "p")

with open(BASE_CONFIG, "r") as f:
    base_cfg = yaml.safe_load(f)

os.makedirs("configs/generated", exist_ok=True)

for beta in BETAS:
    cfg = copy.deepcopy(base_cfg)

    name = f"mnist_student_kd_energy_b{beta_to_name(beta)}"

    cfg["experiment_name"] = name
    cfg["extensions"]["energy_beta"] = beta
    cfg["save"]["checkpoint_path"] = f"checkpoints/{name}.pt"
    cfg["save"]["metrics_path"] = f"results/{name}.csv"

    temp_config_path = f"configs/generated/{name}.yaml"

    with open(temp_config_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"\n=== Running {name} ===")
    subprocess.run(
        [sys.executable, "src/train.py", "--config", temp_config_path],
        check=True,
    )