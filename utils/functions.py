import torch


def iou(pr, gt, eps=1e-7, threshold=None, activation='sigmoid'):
    """
    Args:
        pr(Tensor): A list of predicted elements
        gt(Tensor): A list of global truth elements
        eps(float): Epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
          float: IoU (Jaccard) score
    """
    if activation is None or activation.lower() == "none":
        activation_fn = lambda x: x
    elif activation.lower() == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation.lower() == 'softmax2d':
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pr = activation_fn(pr)
    if threshold is not None:
        pr = (pr > threshold).float()

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return (intersection + eps) / union

jaccard = iou


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, activation='sigmoid'):
    """
    Args:
        pr(Tensor): A list of predicted elements
        gt(Tensor): A list of global truth elements
        beta(float): positive constant
        eps(float): Epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    if activation is None or activation.lower() == "none":
        activation_fn = lambda x: x
    elif activation.lower() == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation.lower() == 'softmax2d':
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pr = activation_fn(pr)
    if threshold is not None:
        pr = (pr > threshold).float()

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp
    score = ((1 + beta ** 2) * tp + eps) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)
    return score
