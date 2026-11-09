"""Build two token-matched training corpora from fineinstructions_nemotron.

Both arms are derived from the SAME pool of source documents:

  arm "pretrain" : the raw human-written Nemotron-CC document  (`text`)
  arm "sft"      : the synthetic instruction/answer pairs derived from that
                   document, rendered with the FineInstructions template
                   "Instruction: {instruction}\n\nAnswer: {answer}"

A document enters the pool iff it has >=1 instruction pair whose Flow-Judge
score is >= --min-judge-score (4 in the paper). That guarantees both arms see
exactly the same documents, so the only variable is the *format* of the text.

Output per arm: a flat uint16/uint32 token stream (.bin memmap) + meta json.
For the sft arm a parallel uint8 mask marks answer tokens (1) vs
instruction/template tokens (0), enabling a completion-only-loss ablation.
"""
import argparse, json, os, sys, time
import multiprocessing as mp
import numpy as np
import pyarrow.parquet as pq

TEMPLATE_PREFIX = "Instruction: "
TEMPLATE_MIDDLE = "\n\nAnswer: "


def judge_path_for(shard_pq: str) -> str:
    base = os.path.basename(shard_pq)
    idx = base.split("-")[1]
    return os.path.join(os.path.dirname(shard_pq), f"judge-{idx}-_-02733.json")


def _worker(task):
    (shard_pq, arm, tokenizer_id, min_score, out_bin, out_mask, dtype_str, max_doc_tokens) = task
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    eos = tok.eos_token_id
    assert eos is not None, "tokenizer has no eos_token_id"
    dtype = np.dtype(dtype_str)

    scores = json.load(open(judge_path_for(shard_pq)))
    pf = pq.ParquetFile(shard_pq)
    assert pf.metadata.num_rows == len(scores), (
        f"judge/parquet misalignment for {shard_pq}: {pf.metadata.num_rows} vs {len(scores)}")

    fb = open(out_bin, "wb")
    fm = open(out_mask, "wb") if out_mask else None
    n_tokens = 0
    n_docs = 0
    n_pairs = 0
    n_rows = 0
    row_off = 0
    cur_warc, cur_text = None, None
    # buffers: batch encode for tokenizer throughput
    buf_text, buf_meta = [], []

    def flush():
        nonlocal n_tokens, buf_text, buf_meta
        if not buf_text:
            return
        need_offsets = fm is not None
        enc = tok(buf_text, add_special_tokens=False,
                  return_offsets_mapping=need_offsets)
        encs = enc["input_ids"]
        offs = enc["offset_mapping"] if need_offsets else [None] * len(encs)
        out_ids, out_msk = [], []
        for ids, off, prompt_chars in zip(encs, offs, buf_meta):
            if max_doc_tokens and len(ids) > max_doc_tokens:
                ids = ids[:max_doc_tokens]
                if off is not None:
                    off = off[:max_doc_tokens]
            ids = ids + [eos]
            out_ids.append(np.asarray(ids, dtype=dtype))
            if fm is not None:
                # A token is an answer token iff any of its characters lie at or
                # past the end of the prompt. Using offsets avoids the
                # boundary-merge off-by-one that comparing token counts causes
                # (e.g. "Answer: " + "Yes" merging into "Answer:" + " Yes").
                m = np.ones(len(ids), dtype=np.uint8)   # trailing eos stays loss-on
                for ti, (s, e) in enumerate(off):
                    if e <= prompt_chars:
                        m[ti] = 0
                out_msk.append(m)
        arr = np.concatenate(out_ids)
        arr.tofile(fb)
        n_tokens += arr.size
        if fm is not None:
            np.concatenate(out_msk).tofile(fm)
        buf_text, buf_meta = [], []

    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=["warc_record_id", "text",
                                           "instantiated_instruction", "answer"]).to_pydict()
        for w, txt, instr, ans in zip(t["warc_record_id"], t["text"],
                                      t["instantiated_instruction"], t["answer"]):
            sc = scores[row_off]
            row_off += 1
            n_rows += 1
            if w != cur_warc:          # start of a new document group
                cur_warc = w
                cur_text = txt
                doc_emitted = False
            if sc < min_score:
                continue
            n_pairs += 1
            if arm == "pretrain":
                if not doc_emitted and cur_text:
                    buf_text.append(cur_text)
                    buf_meta.append(0)
                    doc_emitted = True
                    n_docs += 1
            else:
                if not doc_emitted:
                    doc_emitted = True
                    n_docs += 1
                prompt = TEMPLATE_PREFIX + (instr or "") + TEMPLATE_MIDDLE
                buf_text.append(prompt + (ans or ""))
                buf_meta.append(len(prompt))   # char offset where the answer starts
            if len(buf_text) >= 256:
                flush()
    flush()
    fb.close()
    if fm:
        fm.close()
    return dict(shard=os.path.basename(shard_pq), arm=arm, tokens=int(n_tokens),
                docs=int(n_docs), pairs=int(n_pairs), rows=int(n_rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["pretrain", "sft"], required=True)
    p.add_argument("--shards", nargs="+", required=True, help="parquet paths")
    p.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--min-judge-score", type=int, default=4)
    p.add_argument("--max-doc-tokens", type=int, default=8192,
                   help="truncate pathologically long docs/answers (0=off)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--workers", type=int, default=max(1, min(8, mp.cpu_count() - 1)))
    a = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    vocab = len(tok)
    dtype_str = "uint16" if vocab < 65536 else "uint32"
    os.makedirs(a.out_dir, exist_ok=True)
    tmp = os.path.join(a.out_dir, "tmp")
    os.makedirs(tmp, exist_ok=True)

    tasks = []
    for s in a.shards:
        base = os.path.basename(s).replace(".parquet", "")
        ob = os.path.join(tmp, f"{a.arm}.{a.split}.{base}.bin")
        om = os.path.join(tmp, f"{a.arm}.{a.split}.{base}.mask") if a.arm == "sft" else None
        tasks.append((s, a.arm, a.tokenizer, a.min_judge_score, ob, om, dtype_str, a.max_doc_tokens))

    print(f"arm={a.arm} split={a.split} shards={len(tasks)} workers={a.workers} "
          f"vocab={vocab} dtype={dtype_str}", flush=True)
    t0 = time.time()
    results = []
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for r in pool.imap(_worker, tasks):
            results.append(r)
            el = time.time() - t0
            tot = sum(x["tokens"] for x in results)
            print(f"  [{len(results)}/{len(tasks)}] {r['shard']} "
                  f"{r['tokens']/1e6:.1f}M tok  docs={r['docs']}  pairs={r['pairs']}  "
                  f"| total {tot/1e6:.1f}M  {el/60:.1f}min", flush=True)

    # concatenate temp shards in deterministic shard order
    order = [os.path.basename(s).replace(".parquet", "") for s in a.shards]
    final_bin = os.path.join(a.out_dir, f"{a.arm}.{a.split}.bin")
    final_mask = os.path.join(a.out_dir, f"{a.arm}.{a.split}.mask")
    with open(final_bin, "wb") as fo:
        for b in order:
            fp = os.path.join(tmp, f"{a.arm}.{a.split}.{b}.bin")
            with open(fp, "rb") as fi:
                while True:
                    c = fi.read(1 << 24)
                    if not c:
                        break
                    fo.write(c)
            os.remove(fp)
    if a.arm == "sft":
        with open(final_mask, "wb") as fo:
            for b in order:
                fp = os.path.join(tmp, f"{a.arm}.{a.split}.{b}.mask")
                with open(fp, "rb") as fi:
                    while True:
                        c = fi.read(1 << 24)
                        if not c:
                            break
                        fo.write(c)
                os.remove(fp)

    total_tokens = os.path.getsize(final_bin) // np.dtype(dtype_str).itemsize
    meta = dict(arm=a.arm, split=a.split, tokenizer=a.tokenizer, vocab_size=vocab,
                dtype=dtype_str, total_tokens=int(total_tokens),
                min_judge_score=a.min_judge_score, max_doc_tokens=a.max_doc_tokens,
                template=dict(prefix=TEMPLATE_PREFIX, middle=TEMPLATE_MIDDLE),
                n_docs=sum(r["docs"] for r in results),
                n_pairs=sum(r["pairs"] for r in results),
                n_rows_scanned=sum(r["rows"] for r in results),
                shards=order, per_shard=results,
                has_mask=(a.arm == "sft"), bin=os.path.basename(final_bin),
                build_seconds=round(time.time() - t0, 1))
    with open(os.path.join(a.out_dir, f"{a.arm}.{a.split}.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWROTE {final_bin}: {total_tokens/1e6:.1f}M tokens "
          f"({os.path.getsize(final_bin)/1e9:.2f}GB) docs={meta['n_docs']} pairs={meta['n_pairs']}", flush=True)


if __name__ == "__main__":
    main()
