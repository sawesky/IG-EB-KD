import csv
import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model


def append_metrics(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def make_row(phase, epoch, train_stats=None, val_stats=None, test_stats=None, best_epoch=""):
    train_stats = train_stats or {}
    val_stats = val_stats or {}
    test_stats = test_stats or {}

    return {
        "phase": phase,
        "epoch": epoch,
        "best_epoch": best_epoch,

        "train_loss": train_stats.get("train_loss", ""),
        "train_acc": train_stats.get("train_acc", ""),
        "train_ce": train_stats.get("train_ce", ""),
        "train_kd_kl": train_stats.get("train_kd_kl", ""),
        "train_fisher": train_stats.get("train_fisher", ""),
        "train_energy_margin": train_stats.get("train_energy_margin", ""),

        "val_acc": val_stats.get("acc", ""),
        "val_nll": val_stats.get("nll", ""),
        "val_ece": val_stats.get("ece", ""),
        "val_teacher_student_kl": val_stats.get("teacher_student_kl", ""),
        "val_fisher_mismatch": val_stats.get("fisher_mismatch", ""),
        "val_energy_mismatch": val_stats.get("energy_mismatch", ""),

        "test_acc": test_stats.get("acc", ""),
        "test_nll": test_stats.get("nll", ""),
        "test_ece": test_stats.get("ece", ""),
        "test_teacher_student_kl": test_stats.get("teacher_student_kl", ""),
        "test_fisher_mismatch": test_stats.get("fisher_mismatch", ""),
        "test_energy_mismatch": test_stats.get("energy_mismatch", ""),
    }
