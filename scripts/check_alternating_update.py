#!/usr/bin/env python3
"""Small CPU smoke test for the head-then-backbone training path."""

import math
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from train import train_one_epoch_alternating  # noqa: E402


class TrackingLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.forward_weights = []

    def forward(self, inputs):
        self.forward_weights.append(self.weight.detach().clone())
        return super().forward(inputs)


class ToyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 3)
        self.fc = TrackingLinear(3, 2)

    def forward(self, inputs, return_features=False):
        features = torch.tanh(self.backbone(inputs))
        logits = self.fc(features)
        if return_features:
            return logits, features
        return logits


class ToyTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, inputs):
        return self.fc(inputs)


def parameter_change(before, parameters):
    return sum(
        torch.sum(torch.abs(old - new.detach())).item()
        for old, new in zip(before, parameters)
    )


def check_mode(mode, *, explicit_head_lr=True):
    torch.manual_seed(7)
    model = ToyStudent()
    teacher = ToyTeacher()
    backbone_parameters = list(model.backbone.parameters())
    optimizer = torch.optim.Adam(backbone_parameters, lr=0.05)

    images = torch.tensor(
        [
            [0.2, -0.1, 0.4, 0.3],
            [-0.5, 0.6, 0.2, -0.4],
            [0.7, 0.1, -0.3, 0.2],
            [-0.2, -0.7, 0.5, 0.6],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 0, 1])
    loader = [(images, labels)]

    cfg = {
        "mode": "kd",
        "kd": {"temperature": 1.0, "lambda_kd": 0.6},
        "extensions": {"fisher_alpha": 0.0, "energy_beta": 0.0},
        "head_update": {
            "mode": mode,
            "scheme": "alternating",
            "rho": 0.1,
            "metric_temperature": 1.0,
            "cg_tol": 1.0e-6,
            "cg_max_iter": 50,
        },
    }
    if explicit_head_lr:
        cfg["head_update"]["lr"] = 0.01

    backbone_before = [parameter.detach().clone() for parameter in backbone_parameters]
    head_parameters = list(model.fc.parameters())
    head_before = [parameter.detach().clone() for parameter in head_parameters]

    stats = train_one_epoch_alternating(
        model,
        teacher,
        loader,
        optimizer,
        cfg,
        torch.device("cpu"),
        epoch=1,
    )

    assert parameter_change(head_before, head_parameters) > 0.0
    assert parameter_change(backbone_before, backbone_parameters) > 0.0
    assert len(model.fc.forward_weights) == 3
    assert torch.equal(
        model.fc.forward_weights[0],
        model.fc.forward_weights[1],
    ), "The first two head forwards should use the pre-update head"
    assert not torch.equal(
        model.fc.forward_weights[1],
        model.fc.forward_weights[2],
    ), "The outer loss did not use the updated head"
    assert all(parameter.grad is None for parameter in head_parameters), (
        "The outer backward pass accumulated head gradients"
    )
    assert math.isfinite(stats["head_loss_decrease"])
    expected_head_lr = 0.01 if explicit_head_lr else 0.05
    assert math.isclose(stats["head_lr"], expected_head_lr)
    if mode != "euclidean":
        assert stats["cg_relative_residual_mean"] <= cfg["head_update"]["cg_tol"]

    print(
        f"ok: {mode:15s} | "
        f"head lr={stats['head_lr']:.3e} | "
        f"head loss decrease={stats['head_loss_decrease']:+.3e} | "
        f"CG relres={stats['cg_relative_residual_mean']:.3e}"
    )


def main():
    for mode in ("euclidean", "student_fisher", "teacher_fisher"):
        check_mode(mode)
    check_mode("euclidean", explicit_head_lr=False)
    print("alternating update smoke test passed")


if __name__ == "__main__":
    main()
