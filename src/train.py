import argparse

import torch
import yaml
import os
from tqdm import tqdm

from data import get_mnist_loaders
from losses import ce_loss, kd_loss, output_fisher_loss, energy_margin_loss
from metrics import accuracy, expected_calibration_error, nll, teacher_student_kl
from models import make_model
from utils import append_metrics, get_device, load_checkpoint, save_checkpoint, set_seed


def train_one_epoch(model, teacher, loader, optimizer, cfg, device, epoch):
    model.train()
    if teacher is not None:
        teacher.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_ce = 0.0
    total_kd_kl = 0.0
    total_fisher = 0.0
    total_energy_margin = 0.0
    n_batches = 0

    progress = tqdm(loader, desc=f"epoch {epoch:03d} train", leave=False)

    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        student_logits = model(images)

        if cfg["mode"] == "ce":

            loss = ce_loss(student_logits, labels)
            terms = {
                    "ce": loss.item(),
                    "kd_kl": 0.0,
                    "fisher": 0.0,
                    "energy_margin": 0.0,
            }

        elif cfg["mode"] == "kd":

            with torch.no_grad():
                teacher_logits = teacher(images)
            loss, terms = kd_loss(
                student_logits,
                teacher_logits,
                labels,
                temperature=cfg["kd"]["temperature"],
                lambda_kd=cfg["kd"]["lambda_kd"],
            )

            fisher_alpha = cfg["extensions"]["fisher_alpha"]

            if fisher_alpha > 0.0:
                fisher_loss = output_fisher_loss(student_logits, teacher_logits)
                loss = loss + fisher_alpha * fisher_loss
                terms["fisher"] = fisher_loss.item()
            else:
                terms["fisher"] = 0.0

            energy_beta = cfg["extensions"]["energy_beta"]

            if energy_beta > 0.0:
                margin_loss = energy_margin_loss(student_logits, teacher_logits)
                loss = loss + energy_beta * margin_loss
                terms["energy_margin"] = margin_loss.item()
            else:
                terms["energy_margin"] = 0.0

        else:
            raise ValueError(f"Unknown mode: {cfg['mode']}")

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(student_logits.detach(), labels)
        total_ce += terms.get("ce", 0.0)
        total_kd_kl += terms.get("kd_kl", 0.0)
        total_fisher += terms.get("fisher", 0.0)
        total_energy_margin += terms.get("energy_margin", 0.0)
        n_batches += 1
        
        progress.set_postfix(
            loss=total_loss / n_batches,
            acc=total_acc / n_batches,
        )

    return {
        "train_loss": total_loss / n_batches,
        "train_acc": total_acc / n_batches,
        "train_ce": total_ce / n_batches,
        "train_kd_kl": total_kd_kl / n_batches,
        "train_fisher": total_fisher / n_batches,
        "train_energy_margin": total_energy_margin / n_batches,
    }


@torch.no_grad()
def evaluate(model, teacher, loader, cfg, device, epoch):
    model.eval()
    if teacher is not None:
        teacher.eval()

    total_acc = 0.0
    total_nll = 0.0
    total_ece = 0.0
    total_ts_kl = 0.0
    total_fisher_mismatch = 0.0
    total_energy_mismatch = 0.0
    n_batches = 0
    
    progress = tqdm(loader, desc=f"epoch {epoch:03d} eval", leave=False)

    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        student_logits = model(images)

        total_acc += accuracy(student_logits, labels)
        total_nll += nll(student_logits, labels)
        total_ece += expected_calibration_error(
            student_logits, labels, n_bins=cfg["eval"]["ece_bins"]
        )

        if teacher is not None:
            teacher_logits = teacher(images)
            total_ts_kl += teacher_student_kl(student_logits, teacher_logits)
            total_fisher_mismatch += output_fisher_loss(student_logits, teacher_logits).item()
            total_energy_mismatch += energy_margin_loss(student_logits, teacher_logits).item()

        n_batches += 1

        progress.set_postfix(
            acc=total_acc / n_batches,
            nll=total_nll / n_batches,
            ece=total_ece / n_batches,
        )

    return {
        "test_acc": total_acc / n_batches,
        "test_nll": total_nll / n_batches,
        "test_ece": total_ece / n_batches,
        "test_teacher_student_kl": total_ts_kl / n_batches if teacher is not None else 0.0,
        "test_fisher_mismatch": total_fisher_mismatch / n_batches if teacher is not None else 0.0,
        "test_energy_mismatch": total_energy_mismatch / n_batches if teacher is not None else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if os.path.exists(cfg["save"]["metrics_path"]):
        os.remove(cfg["save"]["metrics_path"])
    
    set_seed(cfg["seed"])
    device = get_device()

    train_loader, test_loader = get_mnist_loaders(
        root=cfg["data"]["root"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    model = make_model(cfg["model"]).to(device)

    teacher = None

    if cfg["mode"] == "kd":
        teacher = make_model(cfg["teacher"]["model"]).to(device)
        teacher.load_state_dict(torch.load(cfg["teacher"]["checkpoint_path"], map_location=device))
        teacher.eval()
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    best_acc = 0.0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_stats = train_one_epoch(model, teacher, train_loader, optimizer, cfg, device, epoch)
        test_stats = evaluate(model, teacher, test_loader, cfg, device, epoch)

        row = {"epoch": epoch, **train_stats, **test_stats}
        append_metrics(cfg["save"]["metrics_path"], row)

        print(
            f"epoch {epoch:03d} | "
            f"train loss {row['train_loss']:.4f} | "
            f"test acc {row['test_acc']:.4f} | "
            f"test nll {row['test_nll']:.4f} | "
            f"ece {row['test_ece']:.4f}"
        )

        if row["test_acc"] > best_acc:
            best_acc = row["test_acc"]
            save_checkpoint(model, cfg["save"]["checkpoint_path"])

    print(f"best test acc: {best_acc:.4f}")
    print(f"saved checkpoint: {cfg['save']['checkpoint_path']}")
    print(f"saved metrics: {cfg['save']['metrics_path']}")


if __name__ == "__main__":
    main()
