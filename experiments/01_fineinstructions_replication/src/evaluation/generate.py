"""Free-form instruction-following evaluation on held-out FineInstructions prompts.

This is the evaluation the paper cares about (it uses MixEval / MT-Bench-101 /
AlpacaEval, all judge-based). Here we compute reference-based and
reference-free automatic metrics, and emit side-by-side generations that an
LLM judge (src/evaluation/judge.py) or a human can score.
"""
import argparse, json, os, re, sys, time
from collections import Counter
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.prepare_data import TEMPLATE_PREFIX, TEMPLATE_MIDDLE, judge_path_for


def load_heldout_prompts(shard, n=200, min_score=4, seed=0, max_instr_chars=1200):
    import pyarrow.parquet as pq
    scores = json.load(open(judge_path_for(shard)))
    pf = pq.ParquetFile(shard)
    out, off = [], 0
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=["warc_record_id", "instantiated_instruction",
                                           "answer"]).to_pydict()
        for w, instr, ans in zip(t["warc_record_id"], t["instantiated_instruction"], t["answer"]):
            sc = scores[off]; off += 1
            if sc < min_score or not instr or not ans:
                continue
            if len(instr) > max_instr_chars:
                continue
            out.append(dict(warc_record_id=w, instruction=instr.strip(), reference=ans.strip(),
                            judge_score=sc))
        if len(out) >= n * 4:
            break
    import random
    random.Random(seed).shuffle(out)
    return out[:n]


def _toks(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def token_f1(pred, ref):
    p, r = Counter(_toks(pred)), Counter(_toks(ref))
    if not p or not r:
        return 0.0
    ov = sum((p & r).values())
    if ov == 0:
        return 0.0
    prec, rec = ov / sum(p.values()), ov / sum(r.values())
    return 2 * prec * rec / (prec + rec)


def distinct_ngram_ratio(s, n=3):
    t = _toks(s)
    if len(t) < n:
        return 1.0
    g = [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
    return len(set(g)) / len(g)


def rouge_l(pred, ref):
    try:
        from rouge_score import rouge_scorer
        sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return sc.score(ref, pred)["rougeL"].fmeasure
    except Exception:
        return float("nan")


@torch.no_grad()
def generate_batch(model, tok, prompts, device, max_new_tokens=256, temperature=0.0,
                   batch_size=8, context_len=1024):
    outs = []
    tok.padding_side = "left"
    tok.truncation_side = "left"  # a long prompt keeps its end, i.e. the "Answer:" cue
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=context_len - max_new_tokens,  # prompt + generation fit the trained context
                  add_special_tokens=False).to(device)
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id,
                  eos_token_id=tok.eos_token_id)
        if temperature and temperature > 0:
            kw.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kw.update(do_sample=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            g = model.generate(**enc, **kw)
        for j in range(len(chunk)):
            new = g[j, enc["input_ids"].shape[1]:]
            hit_eos = bool((new == tok.eos_token_id).any().item())
            txt = tok.decode(new, skip_special_tokens=True)
            # count up to and including the first EOS; the rest is batch padding
            n_new = int((new == tok.eos_token_id).nonzero()[0, 0].item()) + 1 if hit_eos \
                else int(new.shape[0])
            outs.append(dict(text=txt, terminated=hit_eos, n_new_tokens=n_new))
    return outs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="label=path pairs")
    p.add_argument("--val-shard", required=True)
    p.add_argument("--n-prompts", type=int, default=150)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--prompt-style", default="template", choices=["template", "bare"],
                   help="template = FineInstructions 'Instruction:/Answer:' wrapper")
    p.add_argument("--out", default="results/generations.json")
    a = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda"
    items = load_heldout_prompts(a.val_shard, a.n_prompts)
    print(f"{len(items)} held-out instructions (judge>=4) from {os.path.basename(a.val_shard)}",
          flush=True)

    if a.prompt_style == "template":
        prompts = [TEMPLATE_PREFIX + it["instruction"] + TEMPLATE_MIDDLE for it in items]
    else:
        prompts = [it["instruction"] for it in items]

    results = {"prompt_style": a.prompt_style, "items": items, "models": {}}
    for ms in a.models:
        label, path = ms.split("=", 1)
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
        t0 = time.time()
        gens = generate_batch(model, tok, prompts, device, a.max_new_tokens, a.temperature,
                              a.batch_size)
        agg = dict(
            token_f1=sum(token_f1(g["text"], it["reference"]) for g, it in zip(gens, items)) / len(items),
            rouge_l=sum(rouge_l(g["text"], it["reference"]) for g, it in zip(gens, items)) / len(items),
            terminated_frac=sum(g["terminated"] for g in gens) / len(gens),
            mean_new_tokens=sum(g["n_new_tokens"] for g in gens) / len(gens),
            distinct_3gram=sum(distinct_ngram_ratio(g["text"]) for g in gens) / len(gens),
            echoes_instruction_frac=sum(1 for g in gens if "Instruction:" in g["text"]) / len(gens),
            empty_frac=sum(1 for g in gens if not g["text"].strip()) / len(gens),
            seconds=round(time.time() - t0, 1),
        )
        results["models"][label] = dict(path=path, aggregate=agg, generations=gens)
        print(f"\n== {label} ==")
        for k, v in agg.items():
            print(f"   {k:26s} {v:.4f}" if isinstance(v, float) else f"   {k:26s} {v}")
        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=2)
    # human-readable side-by-side
    md = a.out.replace(".json", ".md")
    with open(md, "w") as f:
        f.write(f"# Side-by-side generations (prompt_style={a.prompt_style})\n\n")
        for i, it in enumerate(items[:40]):
            f.write(f"## {i+1}. {it['instruction'][:300]}\n\n")
            f.write(f"**Reference (judge {it['judge_score']}):** {it['reference'][:700]}\n\n")
            for label in results["models"]:
                g = results["models"][label]["generations"][i]
                f.write(f"**{label}** (eos={g['terminated']}, {g['n_new_tokens']} tok): "
                        f"{g['text'][:700]}\n\n")
            f.write("---\n\n")
    print(f"\nwrote {a.out} and {md}", flush=True)


if __name__ == "__main__":
    main()
