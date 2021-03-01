import torch
import numpy as np
import torch.functional as F

DEFAULT_PADDING_INDEX = 0

# from https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/encoders/text/text_encoder.html#stack_and_pad_tensors #


def cross_entropy_with_probs(
    input,
    target,
    weight=None,
    reduction="mean",
):
    # From Snorkel library
    """Calculate cross-entropy loss when targets are probabilities (floats), not ints.
    PyTorch's F.cross_entropy() method requires integer labels; it does accept
    probabilistic labels. We can, however, simulate such functionality with a for loop,
    calculating the loss contributed by each class and accumulating the results.
    Libraries such as keras do not require this workaround, as methods like
    "categorical_crossentropy" accept float labels natively.
    Note that the method signature is intentionally very similar to F.cross_entropy()
    so that it can be used as a drop-in replacement when target labels are changed from
    from a 1D tensor of ints to a 2D tensor of probabilities.
    Parameters
    ----------
    input
        A [num_points, num_classes] tensor of logits
    target
        A [num_points, num_classes] tensor of probabilistic target labels
    weight
        An optional [num_classes] array of weights to multiply the loss by per class
    reduction
        One of "none", "mean", "sum", indicating whether to return one loss per data
        point, the mean loss, or the sum of losses
    Returns
    -------
    torch.Tensor
        The calculated loss
    Raises
    ------
    ValueError
        If an invalid reduction keyword is submitted
    """
    num_points, num_classes = input.shape
    # Note that t.new_zeros, t.new_full put tensor on same device as t
    cum_losses = input.new_zeros(num_points)
    for y in range(num_classes):
        target_temp = input.new_full((num_points,), y, dtype=torch.long)
        y_loss = F.cross_entropy(input, target_temp, reduction="none")
        if weight is not None:
            y_loss = y_loss * weight[y]
        cum_losses += target[:, y].float() * y_loss

    if reduction == "none":
        return cum_losses
    elif reduction == "mean":
        return cum_losses.mean()
    elif reduction == "sum":
        return cum_losses.sum()
    else:
        raise ValueError("Keyword 'reduction' must be one of ['none', 'mean', 'sum']")


def repeat_tensor(a, dim, n_rep):

    old_size = list(a.size())
    new_size = [1 if ix != dim else n_rep for ix, old in enumerate(old_size)]
    return a.repeat(*new_size)


def tile_tensor(a, dim, n_tile):

    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(
        np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
    )
    order_index = order_index.to(a.device)
    return torch.index_select(a, dim, order_index)


def pad_tensor(tensor, length, padding_index=DEFAULT_PADDING_INDEX):
    """Pad a ``tensor`` to ``length`` with ``padding_index``.

    Args:
        tensor (torch.Tensor [n, ...]): Tensor to pad.
        length (int): Pad the ``tensor`` up to ``length``.
        padding_index (int, optional): Index to pad tensor with.

    Returns
        (torch.Tensor [length, ...]) Padded Tensor.
    """
    n_padding = length - tensor.shape[0]
    assert n_padding >= 0
    if n_padding == 0:
        return tensor
    padding = tensor.new(n_padding, *tensor.shape[1:]).fill_(padding_index)
    return torch.cat((tensor, padding), dim=0)


def stack_and_pad_tensors(batch, padding_index=DEFAULT_PADDING_INDEX, dim=0):
    """Pad a :class:`list` of ``tensors`` (``batch``) with ``padding_index``.

    Args:
        batch (:class:`list` of :class:`torch.Tensor`): Batch of tensors to pad.
        padding_index (int, optional): Index to pad tensors with.
        dim (int, optional): Dimension on to which to concatenate the batch of tensors.

    Returns
        BatchedSequences(torch.Tensor, torch.Tensor): Padded tensors and original lengths of
            tensors.
    """
    lengths = [tensor.shape[0] for tensor in batch]
    max_len = max(lengths)
    padded = [pad_tensor(tensor, max_len, padding_index) for tensor in batch]
    lengths = torch.tensor(lengths)
    padded = torch.stack(padded, dim=dim).contiguous()
    for _ in range(dim):
        lengths = lengths.unsqueeze(0)

    return (padded, lengths)


################################################################################################################################
