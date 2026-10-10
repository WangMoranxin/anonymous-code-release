import torch
import torch.nn.functional as F


def compute_parallel_draft_loss(
    *,
    logits,
    target_local_ids,
    eval_mask,
    teacher_logits=None,
    ce_alpha: float = 1.0,
    ce_decay_gamma: float = 0.0,
    kl_alpha: float = 0.0,
    kl_temperature: float = 2.0,
    kl_decay_gamma: float = 4.0,
    compute_metrics: bool = True,
):
    ce_loss_num = None
    ce_loss_den = None
    loss_den = eval_mask.to(torch.float32).sum()
    metrics = {}
    ce_gamma = float(ce_decay_gamma)
    for step, step_logits in enumerate(logits):
        step_targets = target_local_ids[:, step]
        step_mask = eval_mask[:, step].to(torch.float32)
        step_loss = F.cross_entropy(step_logits.float(), step_targets, reduction="none")
        if ce_gamma > 0:
            step_weight = torch.exp(step_mask.new_tensor(-float(step) / ce_gamma))
        else:
            step_weight = step_mask.new_tensor(1.0)
        weighted = (step_loss * step_mask).sum() * step_weight
        weighted_den = step_mask.sum() * step_weight
        ce_loss_num = weighted if ce_loss_num is None else ce_loss_num + weighted
        ce_loss_den = weighted_den if ce_loss_den is None else ce_loss_den + weighted_den
        if compute_metrics:
            den = step_mask.sum().clamp_min(1.0)
            with torch.no_grad():
                pred = step_logits.argmax(dim=-1)
                acc = ((pred == step_targets).to(torch.float32) * step_mask).sum() / den
                metrics[f"acc_pos_{step}"] = float(acc.detach().cpu().item())
                metrics[f"loss_pos_{step}"] = float(
                    ((step_loss * step_mask).sum() / den).detach().cpu().item()
                )
                metrics[f"ce_weight_pos_{step}"] = float(step_weight.detach().cpu().item())

    if ce_loss_num is None:
        raise ValueError("No logits were produced; item_token_num must be positive")

    ce_loss = ce_loss_num / ce_loss_den.clamp_min(1.0)
    kl_loss = ce_loss.new_zeros(())
    if float(kl_alpha) > 0 and teacher_logits:
        kl_num = ce_loss.new_zeros(())
        kl_den = ce_loss.new_zeros(())
        temperature = float(kl_temperature)
        gamma = float(kl_decay_gamma)
        for step, step_teacher_logits in enumerate(teacher_logits):
            if step_teacher_logits is None:
                continue
            step_mask = eval_mask[:, step].to(torch.float32)
            if gamma > 0:
                step_weight = torch.exp(
                    step_mask.new_tensor(-float(step) / gamma)
                )
            else:
                step_weight = step_mask.new_tensor(1.0)
            student_log_probs = F.log_softmax(logits[step].float() / temperature, dim=-1)
            teacher_probs = F.softmax(
                step_teacher_logits.float().detach() / temperature,
                dim=-1,
            )
            step_kl = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="none",
            ).sum(dim=-1) * (temperature * temperature)
            weighted_kl = (step_kl * step_mask).sum() * step_weight
            weighted_den = step_mask.sum() * step_weight
            kl_num = kl_num + weighted_kl
            kl_den = kl_den + weighted_den
            if compute_metrics:
                den = step_mask.sum().clamp_min(1.0)
                with torch.no_grad():
                    metrics[f"kl_pos_{step}"] = float(
                        ((step_kl * step_mask).sum() / den).detach().cpu().item()
                    )
                    metrics[f"kl_weight_pos_{step}"] = float(
                        step_weight.detach().cpu().item()
                    )
        kl_loss = kl_num / kl_den.clamp_min(1.0)

    loss = float(ce_alpha) * ce_loss + float(kl_alpha) * kl_loss
    if compute_metrics:
        metrics["ce_loss"] = float(ce_loss.detach().cpu().item())
        metrics["kl_loss"] = float(kl_loss.detach().cpu().item())
        metrics["loss"] = float(loss.detach().cpu().item())
        metrics["tokens"] = float(loss_den.detach().cpu().item())
    return loss, metrics
