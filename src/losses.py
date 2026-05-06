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


# ----------------------
# for later checkpoints
# ----------------------

def output_fisher_loss(student_logits, teacher_logits):
    raise NotImplementedError("not yet")


def energy_margin_loss(student_logits, teacher_logits, top_k=3):
    raise NotImplementedError("not yet")


def parameter_fisher_loss(student_model, teacher_model, batch):
    raise NotImplementedError("optional")


def grad_field_loss(student_model, teacher_model, images):
    raise NotImplementedError("optional")
