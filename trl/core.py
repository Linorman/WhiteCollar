import numpy as np
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore.ops import functional as F
from mindspore.common.tensor import Tensor
from mindnlp import transforms
from mindnlp.transforms.tokenizers import BertTokenizer

WANDB_PADDING = -1


def flatten_dict(nested, sep='/'):
    """Flatten dictionary and concatenate nested keys with separator."""

    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, dict):
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v

    flat = {}
    rec(nested, '', flat)
    return flat


def stack_dicts(stats_dicts):
    """Stack the values of a dict."""
    results = dict()
    for k in stats_dicts[0]:
        stats_list = [F.flatten(d[k]) for d in stats_dicts]
        results[k] = ops.Pad(paddings=((0, 0), (0, 0)), constant_values=WANDB_PADDING)(ops.Stack(stats_list))
    return results


def add_suffix(input_dict, suffix):
    """Add suffix to dict keys."""
    return {k + suffix: v for k, v in input_dict.items()}


def pad_to_size(tensor, size, dim=1, padding=50256):
    """Pad tensor to size."""
    t_size = tensor.size()[dim]
    if t_size == size:
        return tensor
    else:
        return ops.Pad(paddings=((0, 0), (0, size - t_size)), constant_values=padding)(tensor)


def logprobs_from_logits(logits, labels):
    """Calculate log probabilities from logits."""
    logp = F.log_softmax(logits, axis=2)
    logpy = ops.GatherD()(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy


def whiten(values, shift_mean=True):
    """Whiten values."""
    mean, var = F.mean(values), F.var(values)
    whitened = (values - mean) * ops.Rsqrt()(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def clip_by_value(x, tensor_min, tensor_max):
    """
    Tensor extension to F.clamp
    """
    clipped = ops.Maximum()(ops.Minimum()(x, tensor_max), tensor_min)
    return clipped


def entropy_from_logits(logits):
    """Calculate entropy from logits."""
    pd = F.softmax(logits, axis=-1)
    entropy = Tensor(np.logsumexp(logits, axis=-1) - F.reduce_sum(pd * logits, axis=-1))
    return entropy


def average_mindspore_dicts(list_of_dicts):
    """Average values of a list of dicts with MindSpore tensors."""
    average_dict = dict()
    for key in list_of_dicts[0].keys():
        average_dict[key] = F.mean(ops.Stack([d[key] for d in list_of_dicts]), axis=0)
    return average_dict


def stats_to_np(stats_dict):
    """Cast all MindSpore tensors in dict to numpy arrays."""
    new_dict = dict()
    for k, v in stats_dict.items():
        if isinstance(v, Tensor):
            new_dict[k] = v.asnumpy()
        else:
            new_dict[k] = v
        if np.isscalar(new_dict[k]):
            new_dict[k] = float(new_dict[k])
    return new_dict


def listify_batch(tensor):
    """Turns the first dimension of a tensor into a list."""
    return [tensor[i] for i in range(tensor.shape[0])]


# class PadSequence(Cell):
#     def __init__(self, padding_value=WANDB_PADDING):
#         super(PadSequence, self).__init__()
#         self.padding_value = padding_value
#
#     def construct(self, x):
#         max_len = F.reduce_max(F.shape(x))
#         paddings = mnp.zeros([2, 2], np.int32)
#         paddings[1][1] = max_len - F.shape(x)[1]
#         return ops.Pad(paddings=paddings, constant_values=self.padding_value)(x)


def build_bert_batch_from_txt(text_list, tokenizer, device):
    """Create token id and attention mask tensors from text list for BERT classification."""

    # tokenize
    tensors = [Tensor([tokenizer.encode(text).ids]).to(device) for text in text_list]

    # find max length to pad to
    max_len = max([t.size()[1] for t in tensors])

    # get padded tensors and attention masks
    # (attention masks make bert ignore padding)
    padded_tensors = []
    attention_masks = []
    for t in tensors:
        attention_masks = ops.ones(t.size(), device)
        padded_tensors.append(pad_to_size(t, max_len, padding=0))
        attention_masks.append(pad_to_size(attention_masks, max_len, padding=0))

    # stack all tensors
    padded_tensors = ops.cat(padded_tensors)
    attention_masks = ops.cat(attention_masks)
    return padded_tensors, attention_masks
