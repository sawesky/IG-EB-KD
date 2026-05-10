import copy
import os
import sys
import subprocess
import yaml

BASE_CONFIG = "configs/mnist_student_kd_fisher.yaml"
ALPHAS = [0.01, 0.1, 1.0, 10.0]

def alpha_to_name(alpha):
    return str(alpha).replace(".", "p")

with open(BASE_CONFIG, "r") as f:
    base_cfg = yaml.safe_load(f)

os.makedirs("configs/generated", exist_ok=True)

for alpha in ALPHAS:
    cfg = copy.deepcopy(base_cfg)

    name = f"mnist_student_kd_fisher_a{alpha_to_name(alpha)}"

    cfg["experiment_name"] = name
    cfg["extensions"]["fisher_alpha"] = alpha
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