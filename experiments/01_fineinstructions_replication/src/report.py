"""Aggregate every result artifact into one markdown report."""
import argparse, glob, json, math, os, sys
from collections import defaultdict

ARM_ORDER = ["arm_a_pretrain", "arm_b_sft_from_scratch", "arm_c_pretrain_then_sft"]
ARM_DESC = {
    "arm_a_pretrain": "A - standard pretraining on raw documents",
    "arm_b_sft_from_scratch": "B - SFT-format pretraining from scratch (FineInstructions)",
    "arm_c_pretrain_then_sft": "C - pretrain then short SFT stage (classic pipeline)",
}


def read_metrics(run_dir):
    f = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(f):
        return [], []
    train, evals = [], []
    for line in open(f):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "eval" in r:
            evals.append(r)
        elif "loss" in r:
            train.append(r)
    return train, evals


def fmt_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out", default="results/REPORT.md")
    a = p.parse_args()
    L = []
    W = L.append

    W("# FineInstructions reproduction: pretraining format comparison\n")
    W("Re-implementation of the controlled, token-for-token experiment from "
      "*FineInstructions: Scaling Synthetic Instructions to Pre-Training Scale* "
      "(Patel, Raffel & Callison-Burch, [arXiv:2601.22146](https://arxiv.org/abs/2601.22146)), "
      "at single-GPU scale.\n")

    # ---------------- data ----------------
    W("## 1. Data\n")
    rows = []
    for arm in ("pretrain", "sft"):
        for split in ("train", "val"):
            f = f"data/tokens/{arm}.{split}.meta.json"
            if not os.path.exists(f):
                continue
            m = json.load(open(f))
            rows.append([arm, split, f"{m['total_tokens']/1e6:.1f}M", f"{m['n_docs']:,}",
                         f"{m['n_pairs']:,}", m["min_judge_score"], len(m["shards"])])
    if rows:
        W(fmt_table(["arm", "split", "tokens", "source docs", "instr pairs",
                     "judge>=", "shards"], rows))
        W("")
        pt = [r for r in rows if r[0] == "pretrain" and r[1] == "train"]
        sf = [r for r in rows if r[0] == "sft" and r[1] == "train"]
        if pt and sf:
            W(f"Both arms derive from the **same {pt[0][3]} source documents** "
              f"(a document is included iff it has at least one instruction pair with "
              f"Flow-Judge score >= 4, the paper's filter). Only the *format* differs.\n")

    # ---------------- runs ----------------
    W("## 2. Training runs\n")
    rows = []
    for arm in ARM_ORDER:
        rd = os.path.join(a.runs_dir, arm)
        cf = os.path.join(rd, "config.json")
        if not os.path.exists(cf):
            continue
        c = json.load(open(cf))
        train, evals = read_metrics(rd)
        last = train[-1] if train else {}
        done = [json.loads(l) for l in open(os.path.join(rd, "metrics.jsonl"))
                if '"done"' in l]
        # `wall` restarts at 0 in each process, so sum the segments of a resumed run
        walls = [r["wall"] for r in train if "wall" in r]
        wall_h = sum(prev for prev, cur in zip(walls, walls[1:] + [0]) if cur < prev) / 3600
        rows.append([
            ARM_DESC.get(arm, arm),
            f"{c['params']['total']/1e6:.0f}M",
            f"{c['train_tokens']/1e9:.2f}B",
            c["total_steps"],
            f"{last.get('step','-')}",
            f"{last.get('loss', float('nan')):.3f}" if last else "-",
            f"{last.get('tok_per_s',0)/1e3:.1f}k",
            f"{wall_h:.1f}h" if done else "running",
        ])
    if rows:
        W(fmt_table(["run", "params", "token budget", "steps", "step reached",
                     "last train loss", "tok/s", "wall"], rows))
        W("")

    # final in-training eval
    rows = []
    for arm in ARM_ORDER:
        rd = os.path.join(a.runs_dir, arm)
        _, evals = read_metrics(rd)
        if not evals:
            continue
        e = evals[-1]["eval"]
        rows.append([arm, evals[-1]["step"]] +
                    [f"{e[k]['ppl']:.2f}" for k in ("pretrain_val", "sft_val") if k in e])
    if rows:
        W("### Held-out perplexity during training (last eval)\n")
        W(fmt_table(["run", "step", "pretrain_val ppl", "sft_val ppl"], rows))
        W("")

    # ---------------- perplexity matrix ----------------
    pf = os.path.join(a.results_dir, "perplexity_matrix.json")
    if os.path.exists(pf):
        W("## 3. Cross-format held-out perplexity\n")
        data = json.load(open(pf))
        for scope in ("all_tokens", "answer_tokens_only"):
            sub = [r for r in data if r.get("scope") == scope]
            if not sub:
                continue
            models = sorted({r["model"] for r in sub})
            streams = sorted({r["stream"] for r in sub})
            rows = []
            for m in models:
                row = [m]
                for s in streams:
                    hit = [r for r in sub if r["model"] == m and r["stream"] == s]
                    row.append(f"{hit[0]['ppl']:.2f}" if hit else "-")
                rows.append(row)
            W(f"**scope: {scope}**\n")
            W(fmt_table(["model"] + streams, rows))
            W("")

    # ---------------- benchmarks ----------------
    bfs = sorted(glob.glob(os.path.join(a.results_dir, "benchmarks_*.json")))
    if bfs:
        W("## 4. Zero-shot benchmarks (length-normalized log-likelihood)\n")
        allrows = []
        for f in bfs:
            allrows += json.load(open(f))
        tasks = sorted({r["task"] for r in allrows})
        for fmt in sorted({r["prompt_format"] for r in allrows}):
            sub = [r for r in allrows if r["prompt_format"] == fmt]
            if not sub:
                continue
            models = sorted({r["model"] for r in sub})
            rows = []
            for m in models:
                row = [m]
                accs = []
                for t in tasks:
                    hit = [r for r in sub if r["model"] == m and r["task"] == t]
                    if hit:
                        row.append(f"{hit[0]['acc_norm']*100:.1f}")
                        accs.append(hit[0]["acc_norm"])
                    else:
                        row.append("-")
                row.append(f"**{sum(accs)/len(accs)*100:.1f}**" if accs else "-")
                rows.append(row)
            W(f"**prompt format: {fmt}** (acc_norm %)\n")
            W(fmt_table(["model"] + tasks + ["mean"], rows))
            W("")

    # ---------------- generations ----------------
    gfs = sorted(glob.glob(os.path.join(a.results_dir, "generations*.json")))
    for gi, f in enumerate(gfs):
        d = json.load(open(f))
        W(f"## 5{'abcdefgh'[gi]}. Free-form instruction following (`{os.path.basename(f)}`, "
          f"prompt_style={d.get('prompt_style')})\n")
        keys = ["token_f1", "rouge_l", "terminated_frac", "mean_new_tokens",
                "distinct_3gram", "echoes_instruction_frac", "empty_frac"]
        rows = []
        for label, m in d["models"].items():
            ag = m["aggregate"]
            rows.append([label] + [f"{ag[k]:.3f}" if isinstance(ag.get(k), float) else "-"
                                   for k in keys])
        W(fmt_table(["model"] + keys, rows))
        W("")
        W("`terminated_frac` = fraction of generations that emitted EOS instead of "
          "running to the token cap (an instruction-following model knows when to stop). "
          "`echoes_instruction_frac` = fraction that emitted a new `Instruction:` header "
          "(a raw LM continuing the document rather than answering).\n")

    # ---------------- judge ----------------
    jf = os.path.join(a.results_dir, "judge.json")
    if os.path.exists(jf):
        d = json.load(open(jf))
        W(f"## 6. LLM-as-judge (model: {d.get('judge_model')})\n")
        if "pairwise" in d:
            pw = d["pairwise"]
            A, B = pw["arm_a"], pw["arm_b"]
            W(f"**Pairwise win rate** ({A} vs {B}, both orientations, n={pw['n_judged']}):\n")
            W(fmt_table([A, B, "tie", "orientation agreement"],
                        [[f"{pw[f'win_rate_{A}']*100:.1f}%", f"{pw[f'win_rate_{B}']*100:.1f}%",
                          f"{pw['tie_rate']*100:.1f}%",
                          f"{pw['agreement_both_orientations']*100:.0f}%"]]))
            W("")
        if "absolute" in d:
            W("**Absolute helpfulness (1-10):**\n")
            W(fmt_table(["model", "mean score", "n"],
                        [[k, f"{v['mean_score']:.2f}", v["n"]]
                         for k, v in d["absolute"].items()]))
            W("")
    else:
        W("## 6. LLM-as-judge\n")
        W("Not run (no Anthropic credential available in this environment). "
          "`src/evaluation/judge.py` implements the paper's judge protocols "
          "(AlpacaEval-style pairwise win rate + MT-Bench-style 1-10 scoring) over the "
          "Batch API; set `ANTHROPIC_API_KEY` and run the command in the README.\n")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[wrote {a.out}]", file=sys.stderr)


if __name__ == "__main__":
    main()
