"""Re-evaluate final checkpoints with dataset-level test metrics"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import get_image_test_loader  # noqa: E402
from models import make_model  # noqa: E402


DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_GROUPS = (
    "mnist",
    "cifar10_resnet",
    "cifar10_wrn",
    "cifar100_resnet",
    "cifar100_wrn",
)


@dataclass(frozen=True)
class GroupSpec:
    key: str
    label: str
    dataset: str
    architecture: str
    checkpoint_prefix: str
    student_config: str
    teacher_config: str
    teacher_checkpoint: str
    baseline_variants: tuple[str, ...]


GROUPS = {
    "mnist": GroupSpec(
        key="mnist",
        label="MNIST LeNet",
        dataset="mnist",
        architecture="LeNet",
        checkpoint_prefix="mnist_",
        student_config="configs/mnist_student_ce.yaml",
        teacher_config="configs/mnist_teacher.yaml",
        teacher_checkpoint="mnist_teacher.pt",
        baseline_variants=(
            "mnist_student_ce",
            "mnist_student_kd",
            "mnist_student_kd_fisher",
            "mnist_student_kd_energy",
            "mnist_student_kd_fisher_energy",
        ),
    ),
    "cifar10_resnet": GroupSpec(
        key="cifar10_resnet",
        label="CIFAR-10 ResNet",
        dataset="cifar10",
        architecture="ResNet-20",
        checkpoint_prefix="cifar_resnet20_",
        student_config="configs/cifar_resnet_student_ce.yaml",
        teacher_config="configs/cifar_resnet_teacher.yaml",
        teacher_checkpoint="cifar_resnet56_teacher.pt",
        baseline_variants=(
            "cifar_resnet20_student_ce",
            "cifar_resnet20_student_kd",
            "cifar_resnet20_student_kd_fisher",
            "cifar_resnet20_student_kd_energy",
            "cifar_resnet20_student_kd_fisher_energy",
        ),
    ),
    "cifar10_wrn": GroupSpec(
        key="cifar10_wrn",
        label="CIFAR-10 WRN",
        dataset="cifar10",
        architecture="WRN-16-2",
        checkpoint_prefix="cifar_wrn16_2_",
        student_config="configs/cifar_wrn_student_ce.yaml",
        teacher_config="configs/cifar_wrn_teacher.yaml",
        teacher_checkpoint="cifar_wrn40_2_teacher.pt",
        baseline_variants=(
            "cifar_wrn16_2_student_ce",
            "cifar_wrn16_2_student_kd",
            "cifar_wrn16_2_student_kd_fisher",
            "cifar_wrn16_2_student_kd_energy",
            "cifar_wrn16_2_student_kd_fisher_energy",
        ),
    ),
    "cifar100_resnet": GroupSpec(
        key="cifar100_resnet",
        label="CIFAR-100 ResNet",
        dataset="cifar100",
        architecture="ResNet-20",
        checkpoint_prefix="hifar_resnet20_",
        student_config="configs/hifar_resnet_student_ce.yaml",
        teacher_config="configs/hifar_resnet_teacher.yaml",
        teacher_checkpoint="hifar_resnet56_teacher.pt",
        baseline_variants=(
            "hifar_resnet20_student_ce",
            "hifar_resnet20_student_kd",
            "hifar_resnet20_student_kd_fisher",
            "hifar_resnet20_student_kd_energy",
            "hifar_resnet20_student_kd_fisher_energy",
        ),
    ),
    "cifar100_wrn": GroupSpec(
        key="cifar100_wrn",
        label="CIFAR-100 WRN",
        dataset="cifar100",
        architecture="WRN-16-2",
        checkpoint_prefix="hifar_wrn16_2_",
        student_config="configs/hifar_wrn_student_ce.yaml",
        teacher_config="configs/hifar_wrn_teacher.yaml",
        teacher_checkpoint="hifar_wrn40_2_teacher.pt",
        baseline_variants=(
            "hifar_wrn16_2_student_ce",
            "hifar_wrn16_2_student_kd",
            "hifar_wrn16_2_student_kd_fisher",
            "hifar_wrn16_2_student_kd_energy",
            "hifar_wrn16_2_student_kd_fisher_energy",
        ),
    ),
}


@dataclass(frozen=True)
class EvaluationTarget:
    group: GroupSpec
    family: str
    method: str
    variant: str
    seed: Optional[int]
    rho: Optional[float]
    checkpoint_path: Path
    config_path: Path
    original_metrics_path: Path


PER_CHECKPOINT_FIELDS = (
    "group",
    "dataset",
    "architecture",
    "family",
    "method",
    "variant",
    "seed",
    "rho",
    "n_test",
    "ece_bins",
    "test_acc",
    "test_nll",
    "test_ece",
    "best_epoch",
    "old_batch_test_acc",
    "old_batch_test_nll",
    "old_batch_test_ece",
    "test_acc_delta",
    "test_nll_delta",
    "test_ece_delta",
    "checkpoint_path",
    "original_metrics_path",
    "source_config_path",
    "checkpoint_size_bytes",
    "checkpoint_mtime_ns",
)


SUMMARY_FIELDS = (
    "group",
    "dataset",
    "architecture",
    "family",
    "method",
    "variant",
    "rho",
    "n_seeds",
    "ece_bins",
    "test_acc_mean",
    "test_acc_std",
    "test_nll_mean",
    "test_nll_std",
    "test_ece_mean",
    "test_ece_std",
)


def repository_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def parse_seed(stem: str) -> tuple[str, Optional[int]]:
    match = re.fullmatch(r"(.+)_seed(\d+)", stem)
    if match is None:
        return stem, None
    return match.group(1), int(match.group(2))


def parse_rho(variant: str) -> Optional[float]:
    if "head_euclidean_alternating" in variant:
        return 0.0
    match = re.search(r"_rho([0-9]+(?:p[0-9]+)?)_alternating$", variant)
    if match is None:
        return None
    return float(match.group(1).replace("p", "."))


def method_label(variant: str, family: str) -> str:
    if family == "teacher":
        return "Teacher"
    if "head_euclidean_alternating" in variant:
        return "KD + Euclidean head [alternating]"
    if "head_student_fisher" in variant:
        return "KD + Student-Fisher head [alternating]"
    if "head_teacher_fisher" in variant:
        return "KD + Teacher-induced Fisher head [alternating]"
    if variant.endswith("_student_ce"):
        return "CE"
    if variant.endswith("_student_kd_fisher_energy"):
        return "KD + output Fisher + energy"
    if variant.endswith("_student_kd_fisher"):
        return "KD + output Fisher"
    if variant.endswith("_student_kd_energy"):
        return "KD + energy"
    if variant.endswith("_student_kd"):
        return "KD"
    return variant


def metrics_path_for_checkpoint(
    checkpoint: Path,
    checkpoint_root: Path,
    results_root: Path,
) -> Path:
    relative = checkpoint.relative_to(checkpoint_root)
    return (results_root / relative).with_suffix(".csv")


def discover_targets(
    specs: Iterable[GroupSpec],
    checkpoint_root: Path,
    results_root: Path,
    seeds: tuple[int, ...],
    include_baselines: bool,
    include_alternating: bool,
    include_teachers: bool,
    allow_incomplete: bool,
) -> tuple[list[EvaluationTarget], list[str]]:
    targets: list[EvaluationTarget] = []
    warnings: list[str] = []
    expected_seeds = set(seeds)

    for spec in specs:
        if include_teachers:
            checkpoint = checkpoint_root / spec.teacher_checkpoint
            if checkpoint.exists():
                targets.append(
                    EvaluationTarget(
                        group=spec,
                        family="teacher",
                        method="Teacher",
                        variant=checkpoint.stem,
                        seed=None,
                        rho=None,
                        checkpoint_path=checkpoint,
                        config_path=repository_path(spec.teacher_config),
                        original_metrics_path=metrics_path_for_checkpoint(
                            checkpoint, checkpoint_root, results_root
                        ),
                    )
                )
            else:
                warnings.append(
                    f"{spec.label}: missing teacher checkpoint {display_path(checkpoint)}"
                )

        if include_baselines:
            for variant in spec.baseline_variants:
                available = {
                    seed: checkpoint_root / "seeds" / f"{variant}_seed{seed}.pt"
                    for seed in seeds
                }
                present = {seed for seed, path in available.items() if path.exists()}
                missing = sorted(expected_seeds - present)
                if missing:
                    warnings.append(
                        f"{spec.label}: baseline {variant} missing seeds {missing}"
                    )
                    if not allow_incomplete:
                        continue
                for seed in sorted(present):
                    checkpoint = available[seed]
                    targets.append(
                        EvaluationTarget(
                            group=spec,
                            family="baseline",
                            method=method_label(variant, "baseline"),
                            variant=variant,
                            seed=seed,
                            rho=None,
                            checkpoint_path=checkpoint,
                            config_path=repository_path(spec.student_config),
                            original_metrics_path=metrics_path_for_checkpoint(
                                checkpoint, checkpoint_root, results_root
                            ),
                        )
                    )

        if include_alternating and spec.key != "mnist":
            alternating_dir = checkpoint_root / "head_metric" / "alternating" / "seeds"
            variants: dict[str, dict[int, Path]] = {}
            pattern = f"{spec.checkpoint_prefix}*head_*_alternating_seed*.pt"
            for checkpoint in sorted(alternating_dir.glob(pattern)):
                variant, seed = parse_seed(checkpoint.stem)
                if seed is None or seed not in expected_seeds:
                    continue
                if not any(
                    token in variant
                    for token in (
                        "head_euclidean_alternating",
                        "head_student_fisher",
                        "head_teacher_fisher",
                    )
                ):
                    continue
                variants.setdefault(variant, {})[seed] = checkpoint

            if not variants:
                warnings.append(f"{spec.label}: no alternating checkpoints found")

            for variant, available in sorted(variants.items()):
                present = set(available)
                missing = sorted(expected_seeds - present)
                if missing:
                    warnings.append(
                        f"{spec.label}: alternating {variant} missing seeds {missing}; "
                        "treated as a sweep/partial variant"
                    )
                    if not allow_incomplete:
                        continue
                for seed in sorted(present):
                    checkpoint = available[seed]
                    targets.append(
                        EvaluationTarget(
                            group=spec,
                            family="alternating",
                            method=method_label(variant, "alternating"),
                            variant=variant,
                            seed=seed,
                            rho=parse_rho(variant),
                            checkpoint_path=checkpoint,
                            config_path=repository_path(spec.student_config),
                            original_metrics_path=metrics_path_for_checkpoint(
                                checkpoint, checkpoint_root, results_root
                            ),
                        )
                    )

    targets.sort(
        key=lambda target: (
            DEFAULT_GROUPS.index(target.group.key),
            target.family,
            target.variant,
            -1 if target.seed is None else target.seed,
        )
    )
    return targets, warnings


def load_yaml(path: Path) -> dict:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def load_state_dict(path: Path, device: torch.device) -> dict:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)

    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint contents in {path}")

    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def global_expected_calibration_error(
    confidences: torch.Tensor,
    correctness: torch.Tensor,
    n_bins: int,
) -> float:
    """Compute equal-width top-label ECE once over the full dataset."""
    if confidences.ndim != 1 or correctness.ndim != 1:
        raise ValueError("Confidences and correctness must be one-dimensional")
    if confidences.numel() != correctness.numel():
        raise ValueError("Confidences and correctness must have equal lengths")
    if confidences.numel() == 0:
        raise ValueError("Cannot compute ECE for an empty dataset")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    confidences = confidences.to(dtype=torch.float64, device="cpu")
    correctness = correctness.to(dtype=torch.float64, device="cpu")
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, dtype=torch.float64)
    ece = torch.zeros((), dtype=torch.float64)

    for index in range(n_bins):
        in_bin = (confidences > bin_edges[index]) & (
            confidences <= bin_edges[index + 1]
        )
        count = int(in_bin.sum().item())
        if count == 0:
            continue
        bin_accuracy = correctness[in_bin].mean()
        bin_confidence = confidences[in_bin].mean()
        ece += (count / confidences.numel()) * torch.abs(
            bin_accuracy - bin_confidence
        )
    return float(ece.item())


@torch.inference_mode()
def evaluate_checkpoint(
    target: EvaluationTarget,
    loader,
    device: torch.device,
    n_bins: int,
) -> dict[str, float | int]:
    cfg = load_yaml(target.config_path)
    model = make_model(cfg["model"]).to(device)
    state = load_state_dict(target.checkpoint_path, device)
    model.load_state_dict(state, strict=True)
    model.eval()

    correct_total = 0
    nll_total = 0.0
    example_total = 0
    confidences = []
    correctness = []

    progress = tqdm(
        loader,
        desc=f"{target.group.key}: {target.variant} seed={target.seed}",
        leave=False,
    )
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)

        predictions = logits.argmax(dim=1)
        batch_correct = predictions.eq(labels)
        batch_size = labels.numel()

        correct_total += int(batch_correct.sum().item())
        nll_total += float(
            F.cross_entropy(logits, labels, reduction="sum").item()
        )
        example_total += batch_size

        probabilities = F.softmax(logits, dim=1)
        batch_confidences = probabilities.max(dim=1).values
        confidences.append(batch_confidences.detach().cpu())
        correctness.append(batch_correct.detach().cpu())

    if example_total == 0:
        raise RuntimeError(f"Empty test loader for {target.group.label}")

    all_confidences = torch.cat(confidences)
    all_correctness = torch.cat(correctness)
    return {
        "n_test": example_total,
        "test_acc": correct_total / example_total,
        "test_nll": nll_total / example_total,
        "test_ece": global_expected_calibration_error(
            all_confidences,
            all_correctness,
            n_bins,
        ),
    }


def read_original_test_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", newline="") as handle:
        test_rows = [
            row for row in csv.DictReader(handle) if row.get("phase") == "test"
        ]
    return test_rows[-1] if test_rows else {}


def optional_float(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def difference(new_value: float, old_value: Optional[float]) -> object:
    return "" if old_value is None else new_value - old_value


def make_result_row(
    target: EvaluationTarget,
    metrics: dict[str, float | int],
    n_bins: int,
) -> dict[str, object]:
    old = read_original_test_row(target.original_metrics_path)
    old_acc = optional_float(old.get("test_acc"))
    old_nll = optional_float(old.get("test_nll"))
    old_ece = optional_float(old.get("test_ece"))
    checkpoint_stat = target.checkpoint_path.stat()

    return {
        "group": target.group.key,
        "dataset": target.group.dataset,
        "architecture": target.group.architecture,
        "family": target.family,
        "method": target.method,
        "variant": target.variant,
        "seed": "" if target.seed is None else target.seed,
        "rho": "" if target.rho is None else target.rho,
        "n_test": metrics["n_test"],
        "ece_bins": n_bins,
        "test_acc": metrics["test_acc"],
        "test_nll": metrics["test_nll"],
        "test_ece": metrics["test_ece"],
        "best_epoch": old.get("best_epoch", ""),
        "old_batch_test_acc": "" if old_acc is None else old_acc,
        "old_batch_test_nll": "" if old_nll is None else old_nll,
        "old_batch_test_ece": "" if old_ece is None else old_ece,
        "test_acc_delta": difference(float(metrics["test_acc"]), old_acc),
        "test_nll_delta": difference(float(metrics["test_nll"]), old_nll),
        "test_ece_delta": difference(float(metrics["test_ece"]), old_ece),
        "checkpoint_path": display_path(target.checkpoint_path),
        "original_metrics_path": display_path(target.original_metrics_path),
        "source_config_path": display_path(target.config_path),
        "checkpoint_size_bytes": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(path: Path, fieldnames: Iterable[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def result_is_current(
    row: dict[str, str],
    target: EvaluationTarget,
    n_bins: int,
) -> bool:
    stat = target.checkpoint_path.stat()
    try:
        return (
            int(row.get("ece_bins", "")) == n_bins
            and int(row.get("checkpoint_size_bytes", "")) == stat.st_size
            and int(row.get("checkpoint_mtime_ns", "")) == stat.st_mtime_ns
        )
    except (TypeError, ValueError):
        return False


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def build_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["family"]), str(row["variant"])), []).append(row)

    summaries = []
    for _, variant_rows in sorted(grouped.items()):
        first = variant_rows[0]
        acc_mean, acc_std = mean_std(
            [float(row["test_acc"]) for row in variant_rows]
        )
        nll_mean, nll_std = mean_std(
            [float(row["test_nll"]) for row in variant_rows]
        )
        ece_mean, ece_std = mean_std(
            [float(row["test_ece"]) for row in variant_rows]
        )
        summaries.append(
            {
                "group": first["group"],
                "dataset": first["dataset"],
                "architecture": first["architecture"],
                "family": first["family"],
                "method": first["method"],
                "variant": first["variant"],
                "rho": first["rho"],
                "n_seeds": len(variant_rows),
                "ece_bins": first["ece_bins"],
                "test_acc_mean": acc_mean,
                "test_acc_std": acc_std,
                "test_nll_mean": nll_mean,
                "test_nll_std": nll_std,
                "test_ece_mean": ece_mean,
                "test_ece_std": ece_std,
            }
        )
    return summaries


def print_inventory(targets: list[EvaluationTarget], warnings: list[str]) -> None:
    print("\nGlobal test re-evaluation inventory")
    print("=" * 80)
    for group_key in DEFAULT_GROUPS:
        group_targets = [target for target in targets if target.group.key == group_key]
        if not group_targets:
            continue
        variants: dict[str, list[EvaluationTarget]] = {}
        for target in group_targets:
            variants.setdefault(target.variant, []).append(target)
        print(f"{GROUPS[group_key].label}: {len(group_targets)} checkpoint(s)")
        for variant, variant_targets in sorted(variants.items()):
            seeds = [target.seed for target in variant_targets if target.seed is not None]
            seed_text = "teacher" if not seeds else "seeds " + ",".join(map(str, seeds))
            print(f"  - {variant}: {seed_text}")
    if warnings:
        print("\nInventory warnings")
        print("-" * 80)
        for warning in warnings:
            print(f"  - {warning}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute sample-weighted accuracy/NLL and dataset-level 15-bin "
            "ECE for final test checkpoints."
        )
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=("all",) + DEFAULT_GROUPS,
        default=("all",),
        help="Experiment groups to evaluate (default: all).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Required final seeds (default: 42 43 44 45 46).",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints",
        help="Checkpoint root relative to the repository (default: checkpoints).",
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Original results root relative to the repository (default: results).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/global_evaluation",
        help="Separate output directory (default: results/global_evaluation).",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override the data root from the YAML configs.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--include-baselines",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-alternating",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-teachers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Evaluate partial seed sets too; by default they are only reported.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Recompute rows already present in the new global CSVs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the checkpoint inventory without loading models or data.",
    )
    args = parser.parse_args()

    if "all" in args.groups and len(args.groups) != 1:
        parser.error("Use either --groups all or an explicit group list, not both")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must not contain duplicates")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    if args.ece_bins <= 0:
        parser.error("--ece-bins must be positive")
    return args


def main() -> None:
    args = parse_args()
    group_keys = (
        DEFAULT_GROUPS if tuple(args.groups) == ("all",) else tuple(args.groups)
    )
    specs = [GROUPS[key] for key in group_keys]
    checkpoint_root = repository_path(args.checkpoint_root)
    results_root = repository_path(args.results_root)
    output_dir = repository_path(args.output_dir)
    seeds = tuple(args.seeds)

    targets, warnings = discover_targets(
        specs=specs,
        checkpoint_root=checkpoint_root,
        results_root=results_root,
        seeds=seeds,
        include_baselines=args.include_baselines,
        include_alternating=args.include_alternating,
        include_teachers=args.include_teachers,
        allow_incomplete=args.allow_incomplete,
    )
    print_inventory(targets, warnings)
    if args.dry_run:
        print(f"\nDry run only: {len(targets)} checkpoint(s) would be evaluated.")
        return
    if not targets:
        raise SystemExit("No eligible checkpoints found. Run with --dry-run to inspect paths.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    print(f"\nEvaluation device: {device}")

    loaders = {}
    rows_by_group: dict[str, list[dict[str, object]]] = {}
    existing_by_group: dict[str, dict[str, dict[str, str]]] = {}
    target_keys_by_group = {
        spec.key: {
            display_path(target.checkpoint_path)
            for target in targets
            if target.group.key == spec.key
        }
        for spec in specs
    }
    for spec in specs:
        output_path = output_dir / f"{spec.key}_global_test_metrics.csv"
        existing_rows = [
            row
            for row in read_csv(output_path)
            if row.get("checkpoint_path") in target_keys_by_group[spec.key]
        ]
        existing_by_group[spec.key] = {
            row["checkpoint_path"]: row
            for row in existing_rows
            if row.get("checkpoint_path")
        }
        rows_by_group[spec.key] = list(existing_rows)

    for index, target in enumerate(targets, start=1):
        checkpoint_key = display_path(target.checkpoint_path)
        existing = existing_by_group[target.group.key].get(checkpoint_key)
        if existing is not None and not args.rerun and result_is_current(
            existing, target, args.ece_bins
        ):
            print(f"[{index}/{len(targets)}] SKIP current: {checkpoint_key}")
            continue

        cfg = load_yaml(target.config_path)
        data_cfg = cfg["data"]
        data_root = (
            repository_path(args.data_root)
            if args.data_root is not None
            else repository_path(data_cfg["root"])
        )
        batch_size = args.batch_size or int(data_cfg["batch_size"])
        num_workers = (
            args.num_workers
            if args.num_workers is not None
            else int(data_cfg["num_workers"])
        )
        loader_key = (
            target.group.dataset,
            str(data_root),
            batch_size,
            num_workers,
        )
        if loader_key not in loaders:
            loaders[loader_key] = get_image_test_loader(
                dataset_name=target.group.dataset,
                root=str(data_root),
                batch_size=batch_size,
                num_workers=num_workers,
            )

        print(f"[{index}/{len(targets)}] EVAL: {checkpoint_key}")
        metrics = evaluate_checkpoint(
            target=target,
            loader=loaders[loader_key],
            device=device,
            n_bins=args.ece_bins,
        )
        row = make_result_row(target, metrics, args.ece_bins)

        group_rows = rows_by_group[target.group.key]
        group_rows[:] = [
            old_row
            for old_row in group_rows
            if old_row.get("checkpoint_path") != checkpoint_key
        ]
        group_rows.append(row)
        group_rows.sort(
            key=lambda item: (
                str(item.get("family", "")),
                str(item.get("variant", "")),
                -1 if item.get("seed", "") == "" else int(item["seed"]),
            )
        )
        output_path = output_dir / f"{target.group.key}_global_test_metrics.csv"
        write_csv_atomic(output_path, PER_CHECKPOINT_FIELDS, group_rows)
        existing_by_group[target.group.key][checkpoint_key] = {
            key: str(value) for key, value in row.items()
        }
        print(
            f"    acc={float(metrics['test_acc']):.6f} | "
            f"nll={float(metrics['test_nll']):.6f} | "
            f"global ECE={float(metrics['test_ece']):.6f}"
        )

    print("\nGlobal test summaries")
    print("=" * 80)
    for spec in specs:
        group_rows = rows_by_group[spec.key]
        if not group_rows:
            continue
        summary = build_summary(group_rows)
        metrics_path = output_dir / f"{spec.key}_global_test_metrics.csv"
        summary_path = output_dir / f"{spec.key}_global_test_summary.csv"
        write_csv_atomic(metrics_path, PER_CHECKPOINT_FIELDS, group_rows)
        write_csv_atomic(summary_path, SUMMARY_FIELDS, summary)
        print(f"{spec.label}:")
        for row in summary:
            print(
                f"  {row['method']} | n={row['n_seeds']} | "
                f"acc={float(row['test_acc_mean']):.4f} +/- "
                f"{float(row['test_acc_std']):.4f} | "
                f"nll={float(row['test_nll_mean']):.4f} +/- "
                f"{float(row['test_nll_std']):.4f} | "
                f"ECE={float(row['test_ece_mean']):.4f} +/- "
                f"{float(row['test_ece_std']):.4f}"
            )
        print(f"  saved: {display_path(metrics_path)}")
        print(f"  saved: {display_path(summary_path)}")


if __name__ == "__main__":
    main()
