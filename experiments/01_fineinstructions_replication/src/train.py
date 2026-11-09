"""From-scratch LM training for the FineInstructions format comparison.

One trainer serves every arm; the arm only changes which packed token stream is
fed in. Token budgets are matched exactly across arms, so differences in the
results are attributable to data *format*, not data *quantity*.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.loader import PackedDataset, SequenceSampler, make_batch
from src.model import build_model, param_counts, verify_randomized
from src.ce_utils import per_token_scores


def cosine_lr(step, total, peak, warmup, final_frac=0.1):
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    if step >= total:
        return peak * final_frac
    p = (step - warmup) / max(total - warmup, 1)
    return peak * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def build_optimizer(model, lr, wd, betas=(0.9, 0.95), eps=1e-8):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim < 2 or "norm" in n.lower()) else decay).append(p)
    groups = [{"params": decay, "weight_decay": wd},
              {"params": no_decay, "weight_decay": 0.0}]
    try:
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=True)
    except Exception:
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)


def _decoder_and_head(model):
    """Reach the transformer trunk and the output projection of a CausalLM."""
    inner = getattr(model, "model", None)
    head = getattr(model, "lm_head", None)
    if inner is None or head is None:
        raise RuntimeError("loss_fn needs model.model + model.lm_head")
    return inner, head


def loss_fn(model, x, y, mask=None, chunk_tokens=0):
    """Causal-LM loss.

    chunk_tokens=0 (default) is the fast path: one full-batch cross-entropy.
    chunk_tokens>0 keeps only the hidden states and applies lm_head +
    cross-entropy over token chunks under gradient checkpointing, so at most
    [chunk_tokens, vocab] fp32 exists at once. That trades ~7% throughput for a
    large drop in peak memory -- needed for bigger models/batches, unnecessary
    for the 135M config, where the 30-layer trunk activations dominate instead.
    Both paths are numerically identical (tests/test_loss.py).
    """
    if not chunk_tokens:
        logits = model(input_ids=x).logits.float()
        ls = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1), reduction="none")
        if mask is None:
            mean = ls.mean()
            return mean, mean.detach(), y.numel()
        m = mask.reshape(-1).to(ls.dtype)
        return (ls * m).sum() / m.sum().clamp(min=1.0), ls.mean().detach(), int(m.sum().item())

    inner, head = _decoder_and_head(model)
    hidden = inner(input_ids=x).last_hidden_state
    H = hidden.size(-1)
    h = hidden.reshape(-1, H)
    yy = y.reshape(-1)
    m = mask.reshape(-1).to(h.dtype) if mask is not None else None

    def chunk_sum(hc, yc, mc):
        logits = head(hc).float()
        ls = F.cross_entropy(logits, yc, reduction="none")
        if mc is None:
            return torch.stack([ls.sum(), ls.sum().detach()])
        return torch.stack([(ls * mc).sum(), ls.sum().detach()])

    tot = h.new_zeros(2)
    n = h.size(0)
    for i in range(0, n, chunk_tokens):
        hc, yc = h[i:i + chunk_tokens], yy[i:i + chunk_tokens]
        mc = m[i:i + chunk_tokens] if m is not None else None
        tot = tot + torch.utils.checkpoint.checkpoint(chunk_sum, hc, yc, mc,
                                                      use_reentrant=False)
    if m is None:
        denom = float(n)
        n_eff = n
    else:
        denom = m.sum().clamp(min=1.0)
        n_eff = int(m.sum().item())
    return tot[0] / denom, tot[1].detach() / n, n_eff


@torch.no_grad()
def evaluate(model, ds, seq_len, device, batch_size, max_batches=40, seed=1234, use_mask=False):
    """Mean token-level CE on held-out windows (deterministic subset)."""
    model.eval()
    sampler = SequenceSampler(len(ds), seed=seed, batch_size=batch_size)
    tot, n = 0.0, 0
    for _ in range(max_batches):
        idxs = sampler.next_batch()
        x, y, m = make_batch(ds, idxs, device, use_mask=use_mask)
        ls, _ = per_token_scores(model, x, y)
        if m is not None and use_mask:
            mm = m.to(ls.dtype)
            tot += (ls * mm).sum().item(); n += int(mm.sum().item())
        else:
            tot += ls.sum().item(); n += ls.numel()
    model.train()
    ce = tot / max(n, 1)
    return dict(ce=ce, ppl=math.exp(min(ce, 20)), tokens=n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--train-bin", required=True)
    p.add_argument("--train-mask", default=None)
    p.add_argument("--val-bins", nargs="*", default=[], help="name=path pairs for eval streams")
    p.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--size", default="135m")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--tokens-per-step", type=int, default=131072)
    p.add_argument("--train-tokens", type=float, default=1e9)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--warmup-frac", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--loss-chunk-tokens", type=int, default=0,
                   help="0=full-batch CE (fastest); >0 = chunked+checkpointed CE (less memory)")
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--loss-on-completion-only", action="store_true",
                   help="sft arm ablation: mask instruction/template tokens out of the loss")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=25)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--init-from", default=None,
                   help="branch from a checkpoint dir (weights only, fresh optimizer/schedule)")
    p.add_argument("--resume", action="store_true",
                   help="resume this run in place: restore weights, optimizer state, step and "
                        "data position from runs/<name>/ckpt")
    p.add_argument("--rewarm-steps", type=int, default=100,
                   help="on --resume from a checkpoint without optimizer state, ramp LR back up "
                        "over this many steps so fresh Adam moments don't cause a loss spike")
    p.add_argument("--bench-steps", type=int, default=0, help=">0: throughput benchmark only")
    a = p.parse_args()

    device = "cuda"
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    out = os.path.join(a.out_dir, a.run_name)
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "metrics.jsonl"), "a")

    def log(rec):
        rec["wall"] = round(time.time() - T0, 1)
        logf.write(json.dumps(rec) + "\n"); logf.flush()

    T0 = time.time()
    # ---- data ----
    train_ds = PackedDataset(a.train_bin, a.seq_len, mask_path=a.train_mask)
    print(f"train: {train_ds}", flush=True)
    vals = {}
    for spec in a.val_bins:
        name, path = spec.split("=", 1)
        mp = path.replace(".bin", ".mask")
        vals[name] = PackedDataset(path, a.seq_len, mask_path=mp if os.path.exists(mp) else None)
        print(f"val[{name}]: {vals[name]}", flush=True)

    # ---- model ----
    model, cfg = build_model(a.model_id, a.size, seq_len=a.seq_len, seed=a.seed,
                             dtype=torch.bfloat16)
    if a.init_from:
        sd = torch.load(os.path.join(a.init_from, "model.pt"), map_location="cpu")
        model.load_state_dict(sd["model"])
        print(f"initialized weights from {a.init_from} (step {sd.get('step')})", flush=True)
    model.to(device)
    model.gradient_checkpointing_disable()
    model.train()
    pc = param_counts(model)

    accum = max(1, a.tokens_per_step // (a.micro_batch * a.seq_len))
    eff_tokens_per_step = accum * a.micro_batch * a.seq_len
    total_steps = int(a.train_tokens // eff_tokens_per_step)
    warmup = max(10, int(total_steps * a.warmup_frac))

    opt = build_optimizer(model, a.lr, a.weight_decay)

    start_step = 0
    rewarm = 0
    if a.resume:
        ck = os.path.join(out, "ckpt", "model.pt")
        if os.path.exists(ck):
            sd = torch.load(ck, map_location="cpu")
            model.load_state_dict(sd["model"])
            if "optimizer" in sd:
                opt.load_state_dict(sd["optimizer"])
                print("restored optimizer state", flush=True)
            else:
                print("WARNING checkpoint has no optimizer state; "
                      f"Adam moments restart from zero; re-warming LR over {a.rewarm_steps} steps",
                      flush=True)
                rewarm = a.rewarm_steps
            start_step = int(sd.get("step", 0))
            model.to(device)
            print(f"resuming {a.run_name} at step {start_step} "
                  f"({sd.get('tokens', 0)/1e6:.0f}M tokens already seen)", flush=True)
        else:
            print(f"--resume given but no checkpoint at {ck}; starting fresh", flush=True)

    raw_model = model
    if a.compile:
        print("compiling...", flush=True)
        model = torch.compile(model)

    cfgdump = dict(vars(a), accum=accum, eff_tokens_per_step=eff_tokens_per_step,
                   total_steps=total_steps, warmup_steps=warmup, params=pc,
                   arch=dict(hidden=cfg.hidden_size, layers=cfg.num_hidden_layers,
                             heads=cfg.num_attention_heads, kv_heads=cfg.num_key_value_heads,
                             vocab=cfg.vocab_size, inter=cfg.intermediate_size),
                   train_meta=train_ds.meta)
    json.dump(cfgdump, open(os.path.join(out, "config.json"), "w"), indent=2, default=str)
    print(f"\n=== {a.run_name} ===\n"
          f"tokens/step={eff_tokens_per_step} (micro={a.micro_batch} x accum={accum} x L={a.seq_len})\n"
          f"total_steps={total_steps} warmup={warmup} target_tokens={a.train_tokens/1e9:.3f}B\n"
          f"lr={a.lr} wd={a.weight_decay} completion_only_loss={a.loss_on_completion_only}\n", flush=True)

    use_mask = a.loss_on_completion_only and train_ds.mask is not None
    # each optimizer step consumes `accum` micro-batches, so fast-forward by that much
    sampler = SequenceSampler(len(train_ds), seed=a.seed, batch_size=a.micro_batch,
                              start_step=start_step * accum)

    if a.bench_steps:
        total_steps = a.bench_steps

    if start_step >= total_steps:
        print(f"run already at step {start_step}/{total_steps}; nothing to do", flush=True)
        if not os.path.isdir(os.path.join(out, "hf")):  # crashed between final save and export
            raw_model.save_pretrained(os.path.join(out, "hf"), safe_serialization=True)
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(a.model_id).save_pretrained(os.path.join(out, "hf"))
            print(f"exported HF model -> {os.path.join(out, 'hf')}", flush=True)
        return

    t_log = time.time(); tok_since = 0
    N = pc["total"]
    for step in range(start_step, total_steps):
        lr = cosine_lr(step, total_steps, a.lr, warmup, a.min_lr_frac)
        if rewarm and step - start_step < rewarm:
            lr *= (step - start_step + 1) / rewarm
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        acc_loss, acc_raw = 0.0, 0.0
        for micro in range(accum):
            idxs = sampler.next_batch()
            x, y, m = make_batch(train_ds, idxs, device, use_mask=use_mask)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, raw, _ = loss_fn(model, x, y, m if use_mask else None,
                                       chunk_tokens=a.loss_chunk_tokens)
            (loss / accum).backward()
            acc_loss += loss.item() / accum
            acc_raw += raw.item() / accum
        gn = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), a.grad_clip)
        opt.step()
        tok_since += eff_tokens_per_step

        if (step + 1) % a.log_every == 0 or step == 0:
            dt = time.time() - t_log
            tps = tok_since / max(dt, 1e-6)
            mfu = 6 * N * tps / 71e12  # 3090 bf16 peak ~71 TFLOPS
            done = (step + 1) * eff_tokens_per_step
            eta = (total_steps - step - 1) * eff_tokens_per_step / max(tps, 1) / 3600
            print(f"step {step+1}/{total_steps} loss {acc_loss:.4f} raw {acc_raw:.4f} "
                  f"lr {lr:.2e} gn {float(gn):.2f} {tps/1e3:.1f}k tok/s mfu {mfu*100:.1f}% "
                  f"tok {done/1e6:.0f}M eta {eta:.1f}h", flush=True)
            log(dict(step=step + 1, loss=acc_loss, raw_loss=acc_raw, lr=lr,
                     grad_norm=float(gn), tokens=done, tok_per_s=tps, mfu=mfu))
            t_log = time.time(); tok_since = 0

        if vals and ((step + 1) % a.eval_every == 0 or step + 1 == total_steps):
            ev = {}
            for name, ds in vals.items():
                r = evaluate(model, ds, a.seq_len, device, a.eval_batch_size, a.eval_batches)
                ev[name] = r
                print(f"  eval[{name}] ce {r['ce']:.4f} ppl {r['ppl']:.2f}", flush=True)
            log(dict(step=step + 1, tokens=(step + 1) * eff_tokens_per_step, eval=ev))
            t_log = time.time(); tok_since = 0

        if a.save_every and ((step + 1) % a.save_every == 0 or step + 1 == total_steps) \
                and not a.bench_steps:
            ck = os.path.join(out, "ckpt")
            os.makedirs(ck, exist_ok=True)
            tmp_ck = os.path.join(ck, "model.pt.tmp")
            torch.save({"model": raw_model.state_dict(), "optimizer": opt.state_dict(),
                        "step": step + 1,
                        "tokens": (step + 1) * eff_tokens_per_step}, tmp_ck)
            os.replace(tmp_ck, os.path.join(ck, "model.pt"))  # atomic: never a torn checkpoint
            raw_model.config.save_pretrained(ck)
            print(f"  saved {ck} @ step {step+1}", flush=True)

    if a.bench_steps:
        print("BENCH DONE", flush=True)
        return
    # final HF-format export for easy generation/eval
    final = os.path.join(out, "hf")
    raw_model.save_pretrained(final, safe_serialization=True)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(a.model_id).save_pretrained(final)
    print(f"\nexported HF model -> {final}", flush=True)
    log(dict(done=True, total_tokens=total_steps * eff_tokens_per_step,
             wall_hours=(time.time() - T0) / 3600))


if __name__ == "__main__":
    main()
