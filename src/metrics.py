import torch


def uceis_argmax_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute UCEIS accuracy based on raw predictions and targets.
    :param pred: Shape [num_heads, bs, num_classes]
    :param target: Shape [num_heads, bs]
    :return: Accuracy
    """
    preds = []
    for p_mode in pred:
        preds.append(torch.max(p_mode, -1)[1])
    pred_uceis = torch.stack(preds, dim=1).sum(-1)
    real_uceis = target.sum(-1)
    return (pred_uceis == real_uceis).detach().float().mean().item()


def single_head_accuracy(pred, target):
    return (
        (torch.max(pred.detach(), dim=-1)[1] == target)
        .cpu()
        .float()
        .mean()
        .item()
    )


def __uceis_x_vs_y(uceis_pred, uceis_target, threshold=4):
    """
    Return accuracy on UCEIS with respect to classes uceis < ``threshold`` and classes
    uceis >= ``threshold``. As such, calling
    ``uceis_x_vs_y(p, t, threshold=1)`` yields the 0 vs. rest score.
    ``uceis_x_vs_y(p, t, threshold=4)`` yields [0;3] vs. [4;8].
    """
    return (
        ((uceis_pred < threshold) == (uceis_target < threshold))
        .astype(float)
        .mean()
    )


def uceis_0_vs_rest(uceis_pred, uceis_target):
    return __uceis_x_vs_y(uceis_pred, uceis_target, threshold=1)


def uceis_03_vs_48(uceis_pred, uceis_target):
    return __uceis_x_vs_y(uceis_pred, uceis_target, threshold=4)
