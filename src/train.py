import argparse

import torch
import yaml
import os
from tqdm import tqdm

from data import get_image_loaders
from losses import ce_loss, kd_loss, output_fisher_loss, energy_margin_loss
from metrics import accuracy, expected_calibration_error, nll, teacher_student_kl
from models import make_model
from head_metric import (
    apply_linear_head_direction,
    pack_linear_head_gradient,
    solve_metric_direction,
)
from utils import append_metrics, get_device, load_checkpoint, save_checkpoint, set_seed, make_row


def compute_training_objective(student_logits, teacher_logits, labels, cfg):
    """Return the configured scalar objective and its logging components."""
    if cfg["mode"] == "ce":
        loss = ce_loss(student_logits, labels)
        return loss, {
            "ce": loss.item(),
            "kd_kl": 0.0,
            "fisher": 0.0,
            "energy_margin": 0.0,
        }

    if cfg["mode"] != "kd":
        raise ValueError(f"Unknown mode: {cfg['mode']}")

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

    return loss, terms


def compute_head_direction(
    model,
    head_update_mode,
    student_features,
    student_logits,
    teacher_logits,
    head_cfg,
):
    """Build the explicit final-layer direction from the current head gradient."""
    head_gradient = pack_linear_head_gradient(model.fc)

    if head_update_mode == "euclidean":
        return head_gradient, head_gradient, 0, 0.0

    metric_temperature = float(head_cfg.get("metric_temperature", 1.0))
    if head_update_mode == "student_fisher":
        metric_logits = student_logits
    elif head_update_mode == "teacher_fisher":
        if teacher_logits is None:
            raise ValueError("Teacher-Fisher head updates require a teacher")
        metric_logits = teacher_logits
    else:
        raise ValueError(f"Unknown manual head update mode: {head_update_mode}")

    metric_probs = torch.softmax(
        metric_logits.detach() / metric_temperature,
        dim=1,
    )
    direction, cg_iterations, cg_relative_residual = solve_metric_direction(
        head_gradient,
        student_features,
        metric_probs,
        rho=float(head_cfg["rho"]),
        tol=float(head_cfg.get("cg_tol", 1e-6)),
        max_iter=int(head_cfg.get("cg_max_iter", 50)),
    )

    if (
        float(head_cfg["rho"]) > 0.0
        and cg_iterations == int(head_cfg.get("cg_max_iter", 50))
        and cg_relative_residual > float(head_cfg.get("cg_tol", 1e-6))
    ):
        raise RuntimeError(
            f"{head_update_mode} CG did not converge: "
            f"iterations={cg_iterations}, "
            f"relative_residual={cg_relative_residual:.3e}"
        )

    return head_gradient, direction, cg_iterations, cg_relative_residual


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

    # diagnostics for manual final-layer updates
    total_head_grad_norm = 0.0
    total_head_direction_norm = 0.0
    total_head_direction_ratio = 0.0
    total_head_direction_cosine = 0.0
    total_cg_iterations = 0.0
    total_cg_relative_residual = 0.0
    head_diag_batches = 0

    n_batches = 0

    progress = tqdm(loader, desc=f"epoch {epoch:03d} train", leave=False)

    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        # clear gradients on the whole model
        model.zero_grad(set_to_none=True)
        head_update_mode = cfg.get("head_update", {}).get("mode", "optimizer")

        if head_update_mode in {"student_fisher", "teacher_fisher"}:
            student_logits, student_features = model(images, return_features=True)
        else:
            student_logits = model(images)
            student_features = None

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

        head_gradient = None
        direction = None
        cg_iterations = 0
        cg_relative_residual = 0.0

        if head_update_mode == "optimizer":
            optimizer.step()
        elif head_update_mode == "euclidean":
            # g_h = [grad_W | grad_b], matching the augmented [W | b] layout.
            head_gradient = pack_linear_head_gradient(model.fc)
            direction = head_gradient
            head_lr = optimizer.param_groups[0]["lr"]

            # Adam updates only the backbone; the head receives the explicit euclidean direction d_E = g_h.
            optimizer.step()
            apply_linear_head_direction(model.fc, direction, head_lr)

        elif head_update_mode == "student_fisher":

            head_cfg = cfg["head_update"]
            metric_temperature = float(head_cfg.get("metric_temperature", 1.0))

            head_gradient = pack_linear_head_gradient(model.fc)
            student_metric_probs = torch.softmax(
                student_logits.detach() / metric_temperature, dim=1
            )
            direction, cg_iterations, cg_relative_residual = solve_metric_direction(
                head_gradient,
                student_features,
                student_metric_probs,
                rho=float(head_cfg["rho"]),
                tol=float(head_cfg.get("cg_tol", 1e-6)),
                max_iter=int(head_cfg.get("cg_max_iter", 50)),
            )

            if (
                float(head_cfg["rho"]) > 0.0
                and cg_iterations == int(head_cfg.get("cg_max_iter", 50))
                and cg_relative_residual > float(head_cfg.get("cg_tol", 1e-6))
            ):
                raise RuntimeError(
                    "Student-Fisher CG did not converge: "
                    f"iterations={cg_iterations}, "
                    f"relative_residual={cg_relative_residual:.3e}"
                )

            head_lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            apply_linear_head_direction(model.fc, direction, head_lr)

        elif head_update_mode == "teacher_fisher":

            head_cfg = cfg["head_update"]
            metric_temperature = float(head_cfg.get("metric_temperature", 1.0))

            head_gradient = pack_linear_head_gradient(model.fc)
            teacher_metric_probs = torch.softmax(
                teacher_logits.detach() / metric_temperature, dim=1
            )
            direction, cg_iterations, cg_relative_residual = solve_metric_direction(
                head_gradient,
                student_features,
                teacher_metric_probs,
                rho=float(head_cfg["rho"]),
                tol=float(head_cfg.get("cg_tol", 1e-6)),
                max_iter=int(head_cfg.get("cg_max_iter", 50)),
            )

            if (
                float(head_cfg["rho"]) > 0.0
                and cg_iterations == int(head_cfg.get("cg_max_iter", 50))
                and cg_relative_residual > float(head_cfg.get("cg_tol", 1e-6))
            ):
                raise RuntimeError(
                    "Teacher-Fisher CG did not converge: "
                    f"iterations={cg_iterations}, "
                    f"relative_residual={cg_relative_residual:.3e}"
                )

            head_lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            apply_linear_head_direction(model.fc, direction, head_lr)
        else:
            raise ValueError(f"Unknown head update mode: {head_update_mode}")

        if head_gradient is not None and direction is not None:
            grad_norm = torch.linalg.vector_norm(head_gradient).item()
            direction_norm = torch.linalg.vector_norm(direction).item()
            eps = torch.finfo(head_gradient.dtype).eps

            if grad_norm > eps and direction_norm > eps:
                direction_ratio = direction_norm / grad_norm
                direction_cosine = (
                    torch.sum(head_gradient * direction).item()
                    / (grad_norm * direction_norm)
                )
            else:
                direction_ratio = 0.0
                direction_cosine = 1.0

            total_head_grad_norm += grad_norm
            total_head_direction_norm += direction_norm
            total_head_direction_ratio += direction_ratio
            total_head_direction_cosine += direction_cosine
            total_cg_iterations += float(cg_iterations)
            total_cg_relative_residual += float(cg_relative_residual)
            head_diag_batches += 1

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

    stats = {
        "train_loss": total_loss / n_batches,
        "train_acc": total_acc / n_batches,
        "train_ce": total_ce / n_batches,
        "train_kd_kl": total_kd_kl / n_batches,
        "train_fisher": total_fisher / n_batches,
        "train_energy_margin": total_energy_margin / n_batches,
    }

    if head_diag_batches > 0:
        stats.update({
            "head_grad_norm": total_head_grad_norm / head_diag_batches,
            "head_direction_norm": total_head_direction_norm / head_diag_batches,
            "head_direction_ratio": total_head_direction_ratio / head_diag_batches,
            "head_direction_cosine": total_head_direction_cosine / head_diag_batches,
            "cg_iterations_mean": total_cg_iterations / head_diag_batches,
            "cg_relative_residual_mean": (
                total_cg_relative_residual / head_diag_batches
            ),
        })

    return stats


def train_one_epoch_alternating(model, teacher, loader, optimizer, cfg, device, epoch):
    """Update the head first, then the backbone through the updated head.

    The backbone is evaluated once per minibatch. Its feature graph is retained,
    while detached features are used for the inner head step. The outer loss is
    then rebuilt with the updated head and differentiated only with respect to
    the parameters owned by the backbone optimizer. This is a first-order
    block-coordinate update; it deliberately does not differentiate through the
    head update itself.
    """
    model.train()
    if teacher is not None:
        teacher.eval()

    head_cfg = cfg.get("head_update", {})
    head_update_mode = head_cfg.get("mode", "optimizer")
    manual_modes = {"euclidean", "student_fisher", "teacher_fisher"}
    if head_update_mode not in manual_modes:
        raise ValueError(
            "Alternating updates require one of "
            f"{sorted(manual_modes)}, got {head_update_mode!r}"
        )

    backbone_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    if not backbone_parameters:
        raise ValueError("The backbone optimizer has no trainable parameters")

    total_loss = 0.0
    total_acc = 0.0
    total_ce = 0.0
    total_kd_kl = 0.0
    total_fisher = 0.0
    total_energy_margin = 0.0
    total_head_loss_before = 0.0
    total_head_loss_after = 0.0
    total_head_loss_decrease = 0.0

    total_head_grad_norm = 0.0
    total_head_direction_norm = 0.0
    total_head_direction_ratio = 0.0
    total_head_direction_cosine = 0.0
    total_cg_iterations = 0.0
    total_cg_relative_residual = 0.0
    n_batches = 0

    progress = tqdm(loader, desc=f"epoch {epoch:03d} train", leave=False)

    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)
        model.zero_grad(set_to_none=True)

        # One stochastic backbone pass. Discard its old-head logits, but retain
        # the feature graph for the later backbone-only backward pass.
        unused_logits, student_features = model(images, return_features=True)
        del unused_logits

        teacher_logits = None
        if cfg["mode"] == "kd":
            with torch.no_grad():
                teacher_logits = teacher(images)

        # Inner step: detached features ensure that only the head gets grads.
        head_logits = model.fc(student_features.detach())
        head_loss, _ = compute_training_objective(
            head_logits,
            teacher_logits,
            labels,
            cfg,
        )
        head_loss.backward()

        (
            head_gradient,
            direction,
            cg_iterations,
            cg_relative_residual,
        ) = compute_head_direction(
            model,
            head_update_mode,
            student_features,
            head_logits,
            teacher_logits,
            head_cfg,
        )

        head_lr = optimizer.param_groups[0]["lr"]
        apply_linear_head_direction(model.fc, direction, head_lr)

        # Outer step: rebuild the objective with the newly updated head. Limit
        # gradient accumulation to the backbone parameters owned by Adam.
        model.zero_grad(set_to_none=True)
        student_logits = model.fc(student_features)
        loss, terms = compute_training_objective(
            student_logits,
            teacher_logits,
            labels,
            cfg,
        )
        torch.autograd.backward(loss, inputs=backbone_parameters)
        optimizer.step()

        grad_norm = torch.linalg.vector_norm(head_gradient).item()
        direction_norm = torch.linalg.vector_norm(direction).item()
        eps = torch.finfo(head_gradient.dtype).eps

        if grad_norm > eps and direction_norm > eps:
            direction_ratio = direction_norm / grad_norm
            direction_cosine = (
                torch.sum(head_gradient * direction).item()
                / (grad_norm * direction_norm)
            )
        else:
            direction_ratio = 0.0
            direction_cosine = 1.0

        head_loss_before = head_loss.item()
        head_loss_after = loss.item()
        total_head_loss_before += head_loss_before
        total_head_loss_after += head_loss_after
        total_head_loss_decrease += head_loss_before - head_loss_after
        total_head_grad_norm += grad_norm
        total_head_direction_norm += direction_norm
        total_head_direction_ratio += direction_ratio
        total_head_direction_cosine += direction_cosine
        total_cg_iterations += float(cg_iterations)
        total_cg_relative_residual += float(cg_relative_residual)

        total_loss += head_loss_after
        total_acc += accuracy(student_logits.detach(), labels)
        total_ce += terms.get("ce", 0.0)
        total_kd_kl += terms.get("kd_kl", 0.0)
        total_fisher += terms.get("fisher", 0.0)
        total_energy_margin += terms.get("energy_margin", 0.0)
        n_batches += 1

        progress.set_postfix(
            loss=total_loss / n_batches,
            acc=total_acc / n_batches,
            head_delta=total_head_loss_decrease / n_batches,
        )

    return {
        "train_loss": total_loss / n_batches,
        "train_acc": total_acc / n_batches,
        "train_ce": total_ce / n_batches,
        "train_kd_kl": total_kd_kl / n_batches,
        "train_fisher": total_fisher / n_batches,
        "train_energy_margin": total_energy_margin / n_batches,
        "head_loss_before": total_head_loss_before / n_batches,
        "head_loss_after": total_head_loss_after / n_batches,
        "head_loss_decrease": total_head_loss_decrease / n_batches,
        "head_grad_norm": total_head_grad_norm / n_batches,
        "head_direction_norm": total_head_direction_norm / n_batches,
        "head_direction_ratio": total_head_direction_ratio / n_batches,
        "head_direction_cosine": total_head_direction_cosine / n_batches,
        "cg_iterations_mean": total_cg_iterations / n_batches,
        "cg_relative_residual_mean": total_cg_relative_residual / n_batches,
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
        "acc": total_acc / n_batches,
        "nll": total_nll / n_batches,
        "ece": total_ece / n_batches,
        "teacher_student_kl": total_ts_kl / n_batches if teacher is not None else 0.0,
        "fisher_mismatch": total_fisher_mismatch / n_batches if teacher is not None else 0.0,
        "energy_mismatch": total_energy_mismatch / n_batches if teacher is not None else 0.0,
    }

def make_scheduler(optimizer, cfg):
    scheduler_cfg = cfg["train"].get("scheduler", None)

    if scheduler_cfg is None or scheduler_cfg == "none":
        return None

    if isinstance(scheduler_cfg, str):
        scheduler_name = scheduler_cfg
    else:
        scheduler_name = scheduler_cfg.get("name", "none")

    if scheduler_name == "none":
        return None

    if scheduler_name == "cosine":
        if isinstance(scheduler_cfg, str):
            t_max = cfg["train"].get("scheduler_t_max", cfg["train"]["epochs"])
            min_lr = cfg["train"].get("min_lr", 0.0)
        else:
            t_max = scheduler_cfg.get("t_max", cfg["train"]["epochs"])
            min_lr = scheduler_cfg.get("min_lr", cfg["train"].get("min_lr", 0.0))

        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=min_lr,
        )

    if scheduler_name == "multistep":
        milestones = scheduler_cfg.get("milestones", [60, 90, 120])
        gamma = scheduler_cfg.get("gamma", 0.2)

        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
        )

    raise ValueError(f"Unknown scheduler: {scheduler_name}")


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

    train_loader, val_loader, test_loader = get_image_loaders(
        dataset_name=cfg["data"]["dataset"],
        root=cfg["data"]["root"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        val_size=cfg["data"]["val_size"],
        seed=cfg["seed"],
    )

    model = make_model(cfg["model"]).to(device)

    teacher = None

    if cfg["mode"] == "kd":
        teacher = make_model(cfg["teacher"]["model"]).to(device)
        teacher.load_state_dict(torch.load(cfg["teacher"]["checkpoint_path"], map_location=device))
        teacher.eval()
    
    head_update_cfg = cfg.get("head_update", {})
    head_update_mode = head_update_cfg.get("mode", "optimizer")
    head_update_scheme = head_update_cfg.get("scheme", "simultaneous")

    if head_update_scheme not in {"simultaneous", "alternating"}:
        raise ValueError(f"Unknown head update scheme: {head_update_scheme}")
    if head_update_scheme == "alternating" and head_update_mode == "optimizer":
        raise ValueError(
            "Alternating updates require an explicit manual head mode; choose "
            "euclidean, student_fisher, or teacher_fisher"
        )
    if head_update_mode == "teacher_fisher" and teacher is None:
        raise ValueError("Teacher-Fisher head updates require mode: kd")
    if head_update_mode != "optimizer" and not isinstance(model.fc, torch.nn.Linear):
        raise TypeError("Manual head updates currently require model.fc to be nn.Linear")

    if head_update_mode == "optimizer":
        optimizer_parameters = list(model.parameters())
    elif head_update_mode in {"euclidean", "student_fisher", "teacher_fisher"}:
        head_parameter_ids = {id(p) for p in model.fc.parameters()}
        optimizer_parameters = [p for p in model.parameters() if id(p) not in head_parameter_ids]
    else:
        raise ValueError(f"Unknown head update mode: {head_update_mode}")

    optimizer = torch.optim.Adam(
        optimizer_parameters,
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    
    scheduler = make_scheduler(optimizer, cfg)

    if head_update_mode == "optimizer":
        print("head update: optimizer")
    else:
        print(f"head update: {head_update_mode} ({head_update_scheme})")

    best_val_acc = 0
    best_val_nll = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    patience = cfg["train"]["patience"]

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        if head_update_scheme == "alternating":
            train_stats = train_one_epoch_alternating(
                model,
                teacher,
                train_loader,
                optimizer,
                cfg,
                device,
                epoch,
            )
        else:
            train_stats = train_one_epoch(
                model,
                teacher,
                train_loader,
                optimizer,
                cfg,
                device,
                epoch,
            )
        val_stats = evaluate(model, teacher, val_loader, cfg, device, epoch)

        row = make_row(
            phase="val",
            epoch=epoch,
            train_stats=train_stats,
            val_stats=val_stats,
        )
        row["head_loss_before"] = train_stats.get("head_loss_before", "")
        row["head_loss_after"] = train_stats.get("head_loss_after", "")
        row["head_loss_decrease"] = train_stats.get("head_loss_decrease", "")
        append_metrics(cfg["save"]["metrics_path"], row)

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_stats['train_loss']:.4f} | "
            f"val acc {val_stats['acc']:.4f} | "
            f"val nll {val_stats['nll']:.4f} | "
            f"val ece {val_stats['ece']:.4f}"
        )

        if "head_direction_ratio" in train_stats:
            print(
                "           head | "
                f"||g|| {train_stats['head_grad_norm']:.3e} | "
                f"||d|| {train_stats['head_direction_norm']:.3e} | "
                f"||d||/||g|| {train_stats['head_direction_ratio']:.4f} | "
                f"cos(g,d) {train_stats['head_direction_cosine']:.4f} | "
                f"CG iters {train_stats['cg_iterations_mean']:.2f} | "
                f"CG relres {train_stats['cg_relative_residual_mean']:.3e}"
            )

        if "head_loss_decrease" in train_stats:
            print(
                "    alternating | "
                f"loss before {train_stats['head_loss_before']:.4f} | "
                f"loss after {train_stats['head_loss_after']:.4f} | "
                f"decrease {train_stats['head_loss_decrease']:.3e}"
            )

        if val_stats["nll"] < best_val_nll:
            best_val_nll = val_stats["nll"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(model, cfg["save"]["checkpoint_path"])
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"early stopping at epoch {epoch:03d}")
            break
        
        if scheduler is not None:
            scheduler.step()

    model.load_state_dict(torch.load(cfg["save"]["checkpoint_path"], map_location=device))
    model.eval()

    test_stats = evaluate(model, teacher, test_loader, cfg, device, epoch=best_epoch)

    test_row = make_row(
        phase="test",
        epoch="final",
        test_stats=test_stats,
        best_epoch=best_epoch,
    )
    append_metrics(cfg["save"]["metrics_path"], test_row)

    print(
        f"final test | "
        f"best epoch {best_epoch:03d} | "
        f"test acc {test_stats['acc']:.4f} | "
        f"test nll {test_stats['nll']:.4f} | "
        f"test ece {test_stats['ece']:.4f}"
    )

    print(f"saved checkpoint: {cfg['save']['checkpoint_path']}")
    print(f"saved metrics: {cfg['save']['metrics_path']}")


if __name__ == "__main__":
    main()
