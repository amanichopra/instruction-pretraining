"""Build a from-scratch model by taking a pretrained checkpoint and randomizing its weights.

The architecture (and tokenizer) come from a real pretrained release; every
learnable weight is then re-initialized from the model's own init
distribution, so training starts from true random init -- no pretrained
knowledge survives. `verify_randomized` proves this numerically.
"""
import json, math, os
import torch
import torch.nn as nn

SIZE_PRESETS = {
    # name: architecture overrides applied on top of the base config
    "135m": {},
    "360m": dict(hidden_size=960, intermediate_size=2560, num_hidden_layers=32,
                 num_attention_heads=15, num_key_value_heads=5),
    "tiny": dict(hidden_size=192, intermediate_size=512, num_hidden_layers=6,
                 num_attention_heads=6, num_key_value_heads=2),
}


def load_config(model_id: str, size: str = "135m", max_position_embeddings=None, overrides=None):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    for k, v in SIZE_PRESETS.get(size, {}).items():
        setattr(cfg, k, v)
    if max_position_embeddings:
        cfg.max_position_embeddings = max_position_embeddings
    for k, v in (overrides or {}).items():
        setattr(cfg, k, v)
    cfg.use_cache = False
    return cfg


def param_counts(model):
    total = sum(p.numel() for p in model.parameters())
    emb = 0
    for name, p in model.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            emb += p.numel()
    tied = getattr(model.config, "tie_word_embeddings", False)
    return dict(total=total, embedding=emb, non_embedding=total - emb, tied_embeddings=bool(tied))


def build_model(model_id="HuggingFaceTB/SmolLM2-135M", size="135m", seq_len=2048,
                attn_impl="sdpa", dtype=torch.bfloat16, seed=0, from_pretrained_weights=False,
                overrides=None, verbose=True):
    """Return a randomly initialized model with the architecture of `model_id`.

    from_pretrained_weights=True keeps the pretrained weights (for reference /
    sanity checks only -- the experiments all train from scratch).
    """
    from transformers import AutoModelForCausalLM
    torch.manual_seed(seed)
    cfg = load_config(model_id, size, max_position_embeddings=max(seq_len, 2048), overrides=overrides)

    if from_pretrained_weights:
        model = AutoModelForCausalLM.from_pretrained(model_id, config=cfg, dtype=dtype,
                                                     attn_implementation=attn_impl)
    else:
        # instantiate from config == fresh random init, then re-init explicitly
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype, attn_implementation=attn_impl)
        randomize_(model, seed=seed)
    if verbose:
        pc = param_counts(model)
        print(f"model {model_id} size={size} | params total={pc['total']/1e6:.1f}M "
              f"non-embedding={pc['non_embedding']/1e6:.1f}M "
              f"(tied_emb={pc['tied_embeddings']}) | layers={cfg.num_hidden_layers} "
              f"hidden={cfg.hidden_size} heads={cfg.num_attention_heads}/{cfg.num_key_value_heads} "
              f"vocab={cfg.vocab_size} attn={attn_impl}", flush=True)
    return model, cfg


@torch.no_grad()
def randomize_(model, seed=0):
    """Re-initialize every parameter from scratch (normal(0, initializer_range), norms=1)."""
    torch.manual_seed(seed)
    std = getattr(model.config, "initializer_range", 0.02)
    n_lin, n_emb, n_norm = 0, 0, 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.weight.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.zero_()
            n_lin += 1
        elif isinstance(module, nn.Embedding):
            module.weight.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight[module.padding_idx].zero_()
            n_emb += 1
        elif "RMSNorm" in type(module).__name__ or isinstance(module, nn.LayerNorm):
            if getattr(module, "weight", None) is not None:
                module.weight.fill_(1.0)
            if getattr(module, "bias", None) is not None:
                module.bias.zero_()
            n_norm += 1
    model.tie_weights()
    return dict(linear=n_lin, embedding=n_emb, norm=n_norm, std=std)


@torch.no_grad()
def verify_randomized(model, model_id, max_tensors=24):
    """Prove the model is NOT carrying pretrained knowledge.

    Norm (RMSNorm/LayerNorm) weights are excluded from the verdict: initializing
    them to exactly 1.0 is the correct from-scratch init, and an all-ones vector
    trivially correlates with any pretrained near-ones norm vector. Only the
    tensors that actually store learned knowledge -- embeddings and the
    attention/MLP projection matrices -- determine `is_random_init`.
    """
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(model_id, "model.safetensors")
    except Exception as e:
        return {"checked": False, "reason": str(e)}
    ours = dict(model.named_parameters())
    weight_rows, norm_rows = [], []
    with safe_open(p, framework="pt") as f:
        keys = [k for k in f.keys() if k.endswith(".weight")]
        is_norm = lambda k: "norm" in k.lower()
        # sample across the whole depth, not just layer 0
        knowledge = [k for k in keys if not is_norm(k)]
        step = max(1, len(knowledge) // max_tensors)
        sel = knowledge[::step][:max_tensors] + [k for k in keys if is_norm(k)][:3]
        for k in sel:
            pre = f.get_tensor(k).float()
            cand = k if k in ours else ("model." + k if "model." + k in ours else None)
            if cand is None:
                continue
            cur = ours[cand].float().cpu()
            if cur.shape != pre.shape:
                continue
            cos = torch.nn.functional.cosine_similarity(
                cur.flatten().unsqueeze(0), pre.flatten().unsqueeze(0)).item()
            row = dict(tensor=k, pretrained_std=round(pre.std().item(), 5),
                       ours_std=round(cur.std().item(), 5), cosine_sim=round(cos, 6))
            (norm_rows if is_norm(k) else weight_rows).append(row)
    max_abs_cos = max((abs(r["cosine_sim"]) for r in weight_rows), default=1.0)
    return {"checked": True,
            "n_knowledge_tensors_checked": len(weight_rows),
            "max_abs_cosine_sim_knowledge_tensors": round(max_abs_cos, 6),
            "is_random_init": bool(max_abs_cos < 0.05),
            "knowledge_tensors": weight_rows,
            "norm_tensors_excluded_from_verdict": norm_rows}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--size", default="135m")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    m, cfg = build_model(a.model_id, a.size, dtype=torch.float32)
    print(json.dumps(param_counts(m), indent=2))
    if a.verify:
        print(json.dumps(verify_randomized(m, a.model_id), indent=2))
