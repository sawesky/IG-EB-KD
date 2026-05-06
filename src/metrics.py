import torch
import torch.nn.functional as F


@torch.no_grad()
def accuracy(logits, labels):
    preds = logits.argmax(dim = 1)
    return (preds == labels).float().mean().item()


@torch.no_grad()
def nll(logits, labels):
    return F.cross_entropy(logits, labels).item()


@torch.no_grad()
def teacher_student_kl(student_logits, teacher_logits):
    log_p_student = F.log_softmax(student_logits, dim = 1)
    p_teacher = F.softmax(teacher_logits, dim = 1)
    return F.kl_div(log_p_student, p_teacher, reduction = "batchmean").item()


@torch.no_grad()
def expected_calibration_error(logits, labels, n_bins = 15):
    probs = F.softmax(logits, dim = 1)
    conf, preds = probs.max(dim = 1)
    correct = preds.eq(labels)

    ece = torch.zeros(1, device = logits.device)
    bin_edges = torch.linspace(0, 1, n_bins + 1, device = logits.device)

    for i in range(n_bins):
        in_bin = conf.gt(bin_edges[i]) * conf.le(bin_edges[i + 1])
        prop = in_bin.float().mean()
        if prop.item() > 0:
            acc_bin = correct[in_bin].float().mean()
            conf_bin = conf[in_bin].mean()
            ece += torch.abs(acc_bin - conf_bin) * prop

    return ece.item()
