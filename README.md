## Install

```bash
pip install -r requirements.txt
```

## Run

Train teacher:

```bash
python src/train.py --config configs/mnist_teacher.yaml
```

Train student with CE only:

```bash
python src/train.py --config configs/mnist_student_ce.yaml
```

Train student with KD:

```bash
python src/train.py --config configs/mnist_student_kd.yaml
```

## Repo structure

```text
configs/                 small experiment configs
src/data.py              MNIST dataloaders
src/models.py            for now lenet teacher/student
src/losses.py            losses
src/metrics.py           accuracy, NLL, teacher-student KL, ECE
src/train.py             training/eval
src/utils.py             utilities
```
