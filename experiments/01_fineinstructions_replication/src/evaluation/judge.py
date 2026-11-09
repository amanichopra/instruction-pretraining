"""LLM-as-judge scoring of free-form generations, mirroring the paper's evaluation.

The FineInstructions paper measures free-form response quality with MixEval,
MT-Bench-101 and AlpacaEval -- all judge-based. This implements the two judge
protocols that matter for the comparison:

  pairwise : AlpacaEval-style win rate between two arms. Each item is judged
             twice with the arms swapped, so position bias cancels instead of
             silently favouring whichever arm is shown first.
  absolute : MT-Bench-style 1-10 helpfulness score per response.

Requests go through the Batch API (50% cheaper, and judging is not
latency-sensitive). Scoring is grounded on the held-out reference answer.
"""
import argparse, json, os, sys, time
from collections import defaultdict

PAIRWISE_SYSTEM = """You are grading two AI assistants' answers to the same instruction.
You are given the instruction and a reference answer written from a trusted source document.
Judge which response is more helpful, accurate and better-formed as a direct answer to the instruction.
Ignore length unless it harms quality. Ignore the order the responses are shown in.
These are very small models trained from scratch; expect low quality and grade relatively.
A response that merely continues or restates the instruction without answering it is poor.
Respond with a verdict of "A", "B", or "tie"."""

ABSOLUTE_SYSTEM = """You are grading an AI assistant's answer to an instruction.
You are given the instruction and a reference answer written from a trusted source document.
Rate the response's helpfulness and accuracy as a direct answer to the instruction, 1-10.
These are very small models trained from scratch; expect low quality. Use the full scale:
1-2 = incoherent or unrelated; 3-4 = on-topic but does not answer; 5-6 = partially answers;
7-8 = answers reasonably; 9-10 = answers well and accurately.
A response that merely continues or restates the instruction without answering it scores 1-3."""

PAIRWISE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "description": "One or two sentences of justification."},
        "verdict": {"type": "string", "enum": ["A", "B", "tie"]},
    },
    "required": ["reasoning", "verdict"],
    "additionalProperties": False,
}
ABSOLUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "description": "One or two sentences of justification."},
        # structured outputs reject minimum/maximum on integers; range is enforced when parsing
        "score": {"type": "integer", "description": "Overall score from 1 (worst) to 10 (best)."},
    },
    "required": ["reasoning", "score"],
    "additionalProperties": False,
}


def _clip(s, n=2000):
    s = (s or "").strip()
    return s[:n] if len(s) <= n else s[:n] + " ...[truncated]"


def build_pairwise_prompt(instruction, reference, resp_a, resp_b):
    return (f"# Instruction\n{_clip(instruction)}\n\n"
            f"# Reference answer\n{_clip(reference)}\n\n"
            f"# Response A\n{_clip(resp_a) or '(empty)'}\n\n"
            f"# Response B\n{_clip(resp_b) or '(empty)'}\n\n"
            f"Which response better answers the instruction?")


def build_absolute_prompt(instruction, reference, resp):
    return (f"# Instruction\n{_clip(instruction)}\n\n"
            f"# Reference answer\n{_clip(reference)}\n\n"
            f"# Response\n{_clip(resp) or '(empty)'}\n\n"
            f"Rate the response 1-10.")


def submit_and_collect(client, requests, poll_seconds=30, verbose=True, rebuild_plain=None):
    """Submit a batch and return {custom_id: parsed_json}.

    rebuild_plain: optional zero-arg callable returning the same requests built
    without output_config, used if the API rejects structured outputs.
    """
    try:
        batch = client.messages.batches.create(requests=requests)
    except Exception as e:
        if rebuild_plain is None or "output_config" not in str(e):
            raise
        print(f"  structured outputs rejected ({e}); retrying with plain-text JSON",
              flush=True)
        batch = client.messages.batches.create(requests=rebuild_plain())
    if verbose:
        print(f"batch {batch.id} submitted with {len(requests)} requests", flush=True)
    t0 = time.time()
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        if verbose:
            print(f"  status={b.processing_status} "
                  f"processing={b.request_counts.processing} "
                  f"succeeded={b.request_counts.succeeded} "
                  f"errored={b.request_counts.errored} "
                  f"({(time.time()-t0)/60:.1f}min)", flush=True)
        time.sleep(poll_seconds)
    out, n_err = {}, 0
    for result in client.messages.batches.results(batch.id):
        rt = result.result.type
        if rt == "succeeded":
            msg = result.result.message
            if msg.stop_reason == "refusal":
                n_err += 1
                continue
            text = next((blk.text for blk in msg.content if blk.type == "text"), None)
            if not text:
                n_err += 1
                continue
            parsed = _extract_json(text)
            if parsed is None:
                n_err += 1
            else:
                out[result.custom_id] = parsed
        else:
            n_err += 1
    if verbose:
        print(f"batch done: {len(out)} parsed, {n_err} unusable "
              f"({(time.time()-t0)/60:.1f}min)", flush=True)
    return out


EFFORT = "medium"  # set from --effort in main()

JSON_FALLBACK_SUFFIX = ("\n\nReply with ONLY a JSON object matching this schema, "
                        "no prose and no code fence: ")


def make_request(custom_id, model, system, prompt, schema, max_tokens, structured=True):
    """One batch request. structured=True uses output_config json_schema (guaranteed
    valid JSON); structured=False asks for JSON in plain text, as a fallback for
    deployments where output_config is unavailable."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    kw = dict(model=model, max_tokens=max_tokens, system=system,
              messages=[{"role": "user", "content": prompt}],
              output_config={"effort": EFFORT})
    if structured:
        kw["output_config"]["format"] = {"type": "json_schema", "schema": schema}
    else:
        kw["messages"] = [{"role": "user",
                           "content": prompt + JSON_FALLBACK_SUFFIX + json.dumps(schema)}]
    return Request(custom_id=custom_id, params=MessageCreateParamsNonStreaming(**kw))


def _extract_json(text):
    """Parse a JSON object out of model text (handles stray prose/code fences)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    i, j = text.find("{"), text.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(text[i:j + 1])
        except json.JSONDecodeError:
            return None
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generations", required=True, help="results/generations*.json from generate.py")
    p.add_argument("--mode", choices=["pairwise", "absolute", "both"], default="both")
    p.add_argument("--a", default=None, help="label of arm A (pairwise)")
    p.add_argument("--b", default=None, help="label of arm B (pairwise)")
    p.add_argument("--judge-model", default="claude-opus-5-5")
    p.add_argument("--limit", type=int, default=150)
    p.add_argument("--max-tokens", type=int, default=16000,
                   help="thinking is always on for current models and counts toward this cap")
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--out", default="results/judge.json")
    a = p.parse_args()
    global EFFORT
    EFFORT = a.effort

    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic  (needed for judge-based evaluation)")

    data = json.load(open(a.generations))
    items = data["items"][:a.limit]
    labels = list(data["models"].keys())
    gens = {k: data["models"][k]["generations"] for k in labels}
    A = a.a or labels[0]
    B = a.b or (labels[1] if len(labels) > 1 else labels[0])

    client = anthropic.Anthropic()
    report = {"judge_model": a.judge_model, "n_items": len(items),
              "generations_file": a.generations}

    if a.mode in ("pairwise", "both"):
        assert A != B, "pairwise needs two distinct arms"
        reqs = []
        for i, it in enumerate(items):
            ga, gb = gens[A][i]["text"], gens[B][i]["text"]
            # orientation 0: A shown first; orientation 1: swapped. Averaging the
            # two removes judge position bias.
            reqs.append(make_request(f"pw-{i}-0", a.judge_model, PAIRWISE_SYSTEM,
                                     build_pairwise_prompt(it["instruction"], it["reference"], ga, gb),
                                     PAIRWISE_SCHEMA, a.max_tokens))
            reqs.append(make_request(f"pw-{i}-1", a.judge_model, PAIRWISE_SYSTEM,
                                     build_pairwise_prompt(it["instruction"], it["reference"], gb, ga),
                                     PAIRWISE_SCHEMA, a.max_tokens))
        print(f"\n== pairwise: {A} vs {B} ({len(reqs)} judgements, both orientations) ==", flush=True)
        res = submit_and_collect(client, reqs, rebuild_plain=lambda: [
            make_request(r["custom_id"], a.judge_model, PAIRWISE_SYSTEM,
                         r["params"]["messages"][0]["content"], PAIRWISE_SCHEMA,
                         a.max_tokens, structured=False) for r in reqs])
        wins = defaultdict(float)
        per_item, n = [], 0
        for i in range(len(items)):
            votes = []
            for o in (0, 1):
                r = res.get(f"pw-{i}-{o}")
                if not r:
                    continue
                v = r["verdict"]
                if v == "tie":
                    votes.append("tie")
                elif o == 0:
                    votes.append(A if v == "A" else B)
                else:
                    votes.append(B if v == "A" else A)   # swapped
            if not votes:
                continue
            n += 1
            for v in votes:
                wins[v] += 1.0 / len(votes)
            per_item.append(dict(index=i, votes=votes,
                                 instruction=items[i]["instruction"][:200]))
        tot = max(n, 1)
        pw = {"arm_a": A, "arm_b": B, "n_judged": n,
              f"win_rate_{A}": wins[A] / tot, f"win_rate_{B}": wins[B] / tot,
              "tie_rate": wins["tie"] / tot,
              "agreement_both_orientations": sum(
                  1 for x in per_item if len(x["votes"]) == 2 and x["votes"][0] == x["votes"][1]
              ) / tot,
              "per_item": per_item}
        report["pairwise"] = pw
        print(f"  {A}: {pw[f'win_rate_{A}']*100:.1f}%  |  {B}: {pw[f'win_rate_{B}']*100:.1f}%  "
              f"|  tie: {pw['tie_rate']*100:.1f}%  (n={n}, "
              f"orientation agreement {pw['agreement_both_orientations']*100:.0f}%)", flush=True)

    if a.mode in ("absolute", "both"):
        reqs = []
        for label in labels:
            for i, it in enumerate(items):
                reqs.append(make_request(
                    f"ab-{label}-{i}", a.judge_model, ABSOLUTE_SYSTEM,
                    build_absolute_prompt(it["instruction"], it["reference"], gens[label][i]["text"]),
                    ABSOLUTE_SCHEMA, a.max_tokens))
        print(f"\n== absolute scoring: {len(labels)} arms x {len(items)} items ==", flush=True)
        res = submit_and_collect(client, reqs, rebuild_plain=lambda: [
            make_request(r["custom_id"], a.judge_model, ABSOLUTE_SYSTEM,
                         r["params"]["messages"][0]["content"], ABSOLUTE_SCHEMA,
                         a.max_tokens, structured=False) for r in reqs])
        report["absolute"] = {}
        for label in labels:
            sc = [res[f"ab-{label}-{i}"]["score"] for i in range(len(items))
                  if f"ab-{label}-{i}" in res
                  and isinstance(res[f"ab-{label}-{i}"].get("score"), int)
                  and 1 <= res[f"ab-{label}-{i}"]["score"] <= 10]
            if not sc:
                continue
            report["absolute"][label] = dict(
                mean_score=sum(sc) / len(sc), n=len(sc),
                histogram={str(k): sc.count(k) for k in range(1, 11)})
            print(f"  {label:28s} mean {sum(sc)/len(sc):.2f}/10  (n={len(sc)})", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
