"""Memory-bounded per-token scoring shared by every evaluation script.

A [batch, seq, 49152] fp32 logit tensor is several GB, which makes evaluation
impossible on a GPU that is already busy training. These helpers apply the
output projection over token chunks, so peak fp32 logit memory is
chunk_tokens x vocab regardless of batch size.
"""
import contextlib

import torch
import torch.nn.functional as F


def decoder_and_head(model):
    """Reach the transformer trunk and output projection of a CausalLM."""
    inner = getattr(model, "model", None)
    head = getattr(model, "lm_head", None)
    if inner is None or head is None:
        raise RuntimeError("need model.model + model.lm_head")
    return inner, head


@torch.no_grad()
def per_token_scores(model, x, y, chunk_tokens=1024, autocast=True):
    """Score y[i] under the distribution predicted at position i of x.

    Returns (ce, is_argmax), both shaped like y:
      ce        -- per-token cross-entropy, i.e. -log p(y[i])
      is_argmax -- whether y[i] was the model's top-1 prediction
    """
    inner, head = decoder_and_head(model)
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if autocast
           else contextlib.nullcontext())
    with ctx:
        hidden = inner(input_ids=x).last_hidden_state
    h = hidden.reshape(-1, hidden.size(-1))
    yy = y.reshape(-1)
    ces, greedy = [], []
    for i in range(0, h.size(0), chunk_tokens):
        logits = head(h[i:i + chunk_tokens]).float()
        tgt = yy[i:i + chunk_tokens]
        ces.append(F.cross_entropy(logits, tgt, reduction="none"))
        greedy.append(logits.argmax(-1) == tgt)
        del logits
    return torch.cat(ces).view_as(y), torch.cat(greedy).view_as(y)
