from collections import defaultdict

import torch


def compute_metrics(scores_flat, gt_idx, inverse_idx, loss_value: float) -> dict[str, float]:
    B = scores_flat.size(0)

    with torch.no_grad():
        probs = torch.nn.functional.softmax(scores_flat, dim=1)
        pred_idx = scores_flat.argmax(dim=1)

        return {
            "loss_sum": loss_value * B,
            "loss": loss_value,
            "correct": (pred_idx == gt_idx).sum().item(),
            "correct_inverse": (pred_idx == inverse_idx).sum().item(),
            "gt_score_sum": probs[range(B), gt_idx].sum().item(),
            "inverse_gt_score_sum": probs[range(B), inverse_idx].sum().item(),
            "pred_score_sum": probs[range(B), pred_idx].sum().item(),
            "count": B
        }
    
def aggregate_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    agg = defaultdict(float)
    total_count = 0

    for metrics in metrics_list:
        for k, v in metrics.items():
            agg[k] += v
        total_count += metrics["count"]

    # post-process into averaged metrics
    return {
        "loss": agg["loss_sum"] / total_count,
        "acc": agg["correct"] / total_count,
        "acc_incl_inverse": (agg["correct"] + agg["correct_inverse"]) / total_count,
        "avg_gt_move_score": agg["gt_score_sum"] / total_count,
        "avg_inverse_gt_move_score": agg["inverse_gt_score_sum"] / total_count,
        "avg_pred_move_score": agg["pred_score_sum"] / total_count,
    }
