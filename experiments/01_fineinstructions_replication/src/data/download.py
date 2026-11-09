"""Download parquet shards + judge score files from fineinstructions/fineinstructions_nemotron."""
import argparse, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import urllib.request

REPO = "fineinstructions/fineinstructions_nemotron"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/data"
N_SHARDS = 2755          # train-NNNNN-of-02755.parquet
JUDGE_SUFFIX = "_-02733" # judge files use a different suffix than the parquet shards


def _fetch(url: str, dest: str, retries: int = 4) -> int:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return os.path.getsize(dest)
    tmp = dest + ".part"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pretrain-from-scratch/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, dest)
            return os.path.getsize(dest)
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt == retries - 1:
                print(f"  FAILED {url}: {e}", flush=True)
                return 0
            time.sleep(5 * (attempt + 1))
    return 0


def shard_names(idx: int):
    return (f"train-{idx:05d}-of-{N_SHARDS:05d}.parquet", f"judge-{idx:05d}-{JUDGE_SUFFIX}.json")


def download_shards(indices, out_dir: str, workers: int = 3):
    os.makedirs(out_dir, exist_ok=True)
    jobs = []
    for i in indices:
        pqf, jsf = shard_names(i)
        jobs.append((f"{BASE}/{pqf}", os.path.join(out_dir, pqf)))
        jobs.append((f"{BASE}/{jsf}", os.path.join(out_dir, jsf)))
    t0 = time.time()
    total = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch, u, d): d for u, d in jobs}
        done = 0
        for f in futs:
            pass
        for fut in list(futs):
            n = fut.result()
            total += n
            done += 1
            print(f"[{done}/{len(jobs)}] {os.path.basename(futs[fut])} {n/1e6:.1f}MB "
                  f"({total/1e9:.2f}GB, {total/1e6/max(time.time()-t0,1):.1f}MB/s)", flush=True)
    print(f"DONE {total/1e9:.2f}GB in {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-shards", type=int, default=12, help="number of shards starting at --start")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--val-shards", type=int, nargs="*", default=[2700],
                   help="shard indices held out for validation (disjoint docs from train)")
    p.add_argument("--out", default="data/raw")
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args()
    idx = list(range(a.start, a.start + a.train_shards)) + list(a.val_shards)
    print(f"downloading {len(idx)} shards -> {a.out}: {idx}", flush=True)
    download_shards(idx, a.out, a.workers)
