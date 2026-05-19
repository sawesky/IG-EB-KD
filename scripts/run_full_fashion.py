import subprocess
import sys


COMMANDS = [
    ["src/train.py", "--config", "configs/fashion_teacher.yaml"],
    ["src/train.py", "--config", "configs/fashion_student_ce.yaml"],
    ["src/train.py", "--config", "configs/fashion_student_kd.yaml"],
    ["src/train.py", "--config", "configs/fashion_student_kd_fisher.yaml"],
    ["src/train.py", "--config", "configs/fashion_student_kd_energy.yaml"],
    ["src/train.py", "--config", "configs/fashion_student_kd_fisher_energy.yaml"],
    ["scripts/summarize_fashion.py"],
]


def main():
    for command in COMMANDS:
        full_command = [sys.executable] + command

        print("\n" + "=" * 80)
        print("Running:", " ".join(full_command))
        print("=" * 80)

        subprocess.run(full_command, check=True)

    print("\nAll FashionMNIST experiments finished.")


if __name__ == "__main__":
    main()