import subprocess
import sys


COMMANDS = [
    ["src/train.py", "--config", "configs/cifar_teacher.yaml"],
    ["src/train.py", "--config", "configs/cifar_student_ce.yaml"],
    ["src/train.py", "--config", "configs/cifar_student_kd.yaml"],
    ["src/train.py", "--config", "configs/cifar_student_kd_fisher.yaml"],
    ["src/train.py", "--config", "configs/cifar_student_kd_energy.yaml"],
    ["src/train.py", "--config", "configs/cifar_student_kd_fisher_energy.yaml"],
    ["scripts/summarize_cifar.py"],
]


def main():
    for command in COMMANDS:
        full_command = [sys.executable] + command

        print("\n" + "=" * 80)
        print("Running:", " ".join(full_command))
        print("=" * 80)

        subprocess.run(full_command, check=True)

    print("\nAll CIFAR-10 experiments finished.")


if __name__ == "__main__":
    main()