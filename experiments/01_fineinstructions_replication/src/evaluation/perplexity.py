"""Cross-format held-out perplexity matrix: every model x every held-out stream.

The diagonal is each model's own distribution (always favourable); the
off-diagonal is the interesting part -- how well a model trained on one format
explains text in the other.
"""
import argparse, json, math, os, sys
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.loader import PackedDataset, SequenceSampler, make_batch
from src.ce_utils import per_token_scores


@torch.no_grad()
def stream_ce(model, ds, seq_len, device, batch_size=8, max_batches=60, seed=1234,
              answer_tokens_only=False):
    sampler = SequenceSampler(len(ds), seed=seed, batch_size=batch_size)
    tot, n = 0.0, 0
    for _ in range(max_batches):
        idxs = sampler.next_batch()
        x, y, m = make_batch(ds, idxs, device, use_mask=answer_tokens_only)
        ls, _ = per_token_scores(model, x, y)
        if answer_tokens_only and m is not None:
            mm = m.to(ls.dtype)
            tot += (ls * mm).sum().item(); n += int(mm.sum().item())
        else:
            tot += ls.sum().item(); n += ls.numel()
    ce = tot / max(n, 1)
    return dict(ce=ce, ppl=math.exp(min(ce, 20)), bits_per_token=ce / math.log(2), n_tokens=n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="label=path pairs")
    p.add_argument("--streams", nargs="+", required=True, help="label=path.bin pairs")
    p.add_argument("--seq-len", type=int, default=1024)  # = training context; longer windows score untrained positions
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=60)
    p.add_argument("--out", default="results/perplexity_matrix.json")
    a = p.parse_args()
    from transformers import AutoModelForCausalLM
    device = "cuda"

    streams = {}
    for s in a.streams:
        name, path = s.split("=", 1)
        mp = path.replace(".bin", ".mask")
        streams[name] = PackedDataset(path, a.seq_len, mask_path=mp if os.path.exists(mp) else None)
        print(f"stream[{name}] {streams[name]}", flush=True)

    rows = []
    for ms in a.models:
        label, path = ms.split("=", 1)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
        for sname, ds in streams.items():
            r = stream_ce(model, ds, a.seq_len, device, a.batch_size, a.max_batches)
            r.update(model=label, stream=sname, scope="all_tokens")
            rows.append(r)
            print(f"  {label:28s} -> {sname:16s} ce {r['ce']:.4f} ppl {r['ppl']:8.2f}", flush=True)
            if ds.mask is not None:
                r2 = stream_ce(model, ds, a.seq_len, device, a.batch_size, a.max_batches,
                               answer_tokens_only=True)
                r2.update(model=label, stream=sname, scope="answer_tokens_only")
                rows.append(r2)
                print(f"  {label:28s} -> {sname:16s} ce {r2['ce']:.4f} ppl {r2['ppl']:8.2f} "
                      f"[answer tokens only]", flush=True)
        del model
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
