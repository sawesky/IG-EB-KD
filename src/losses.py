import torch
import torch.nn.functional as F


def ce_loss(student_logits, labels):
    """
    Cross-entropy loss
    """
    return F.cross_entropy(student_logits, labels)


def kd_kl_loss(student_logits, teacher_logits, temp):
    """
    KL(teacher || student), computed on softened distributions.
    """
    log_p_student = F.log_softmax(student_logits / temp, dim = 1)
    p_teacher = F.softmax(teacher_logits / temp, dim = 1)
    return F.kl_div(log_p_student, p_teacher, reduction = "batchmean") * (temp * temp)


def kd_loss(student_logits, teacher_logits, labels, temperature=4.0, lambda_kd=0.7):
    ce = ce_loss(student_logits, labels)
    kl = kd_kl_loss(student_logits, teacher_logits, temperature)
    total = (1.0 - lambda_kd) * ce + lambda_kd * kl
    return total, {"ce": ce.item(), "kd_kl": kl.item()}

# output space fisher

def output_fisher_matrix(logits):
    p = F.softmax(logits, dim=1)
    diag_p = torch.diag_embed(p)
    outer_p = p.unsqueeze(2) * p.unsqueeze(1)
    return diag_p - outer_p

def output_fisher_loss(student_logits, teacher_logits):
    student_fisher = output_fisher_matrix(student_logits)
    teacher_fisher = output_fisher_matrix(teacher_logits)
    return torch.mean((student_fisher - teacher_fisher) ** 2)

# energy margin

def logit_margins(logits):
    num_classes = logits.shape[1]
    i, j = torch.triu_indices(num_classes, num_classes, offset=1, device=logits.device) # upper triangular without main diag (offset = 1) 
    return logits[:, i] - logits[:, j] # z_i - z_j for all i < j


def energy_margin_loss(student_logits, teacher_logits):
    student_margins = logit_margins(student_logits)
    teacher_margins = logit_margins(teacher_logits)

    return torch.mean((student_margins - teacher_margins) ** 2)


# ----------------------
# for later checkpoints
# ----------------------

def parameter_fisher_loss(student_model, teacher_model, batch):
    raise NotImplementedError("optional")


def grad_field_loss(student_model, teacher_model, images):
    raise NotImplementedError("optional")
