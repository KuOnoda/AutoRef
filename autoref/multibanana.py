"""Run a harness on MultiBanana and score it with the benchmark's official judge.

    python -m autoref.multibanana --split test                               # FLUX.2 [klein] 4B
    python -m autoref.multibanana --split test --generator qwen-image-edit-2511

Splits are fixed lists of item ids in data/splits/multibanana_<split>.json:

    train   48 four-reference items the search scores candidates on (D_train)
    val     48 four-reference items the search selects candidates on (D_val)
    test    133 four-reference items held out from the search
    ref3    96 three-reference items
    ref5    96 five-reference items

The dataset keeps each item in `<n>_<kind>/` as `<id>_0..<n-1>.jpg` (references) and
`<id>_prompt.txt`. The official judge looks for the output as `<id>_generated.png` next to the
references, so a run directory links them in and writes the output beside them. A run directory also
holds a trace of every model call per item (`trace/`), every generated image (`rounds/`), the judge's
answers (`judge_results.json`) and the summary (`harness_result.json`).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from autoref.generators import GENERATORS, get_generator, release
from autoref.harness import Context, load_harness
from autoref.paths import MULTIBANANA, OUTPUTS, SPLITS

TASK_NAME = "multibanana"
DEFAULT_GENERATOR = "flux-klein-4b"
KINDS = ["back", "global", "local", "object"]
METRIC, METRIC_LABEL = "overall", "MultiBanana overall"
METRICS = ["Instruction Alignment", "Reference Consistency", "Background-Subject Match",
           "Physical Realism", "Visual Quality"]
SPLIT_NAMES = ["train", "val", "test", "ref3", "ref5"]
TRACE_MODES = ("text", "visual")


# ---- data -----------------------------------------------------------------------------------
def split_ids(split):
    path = os.path.join(SPLITS, f"multibanana_{split}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no split file {path}")
    return json.load(open(path))["ids"]


def load_item(item_id):
    category, stem = item_id.split("/")
    d = os.path.join(MULTIBANANA, category)
    n_refs = int(category.split("_")[0])
    refs = sorted(os.path.join(d, p) for p in os.listdir(d)
                  if p.startswith(stem + "_") and not p.endswith("_prompt.txt"))
    if len(refs) != n_refs:
        raise RuntimeError(f"{item_id}: expected {n_refs} references, found {len(refs)}")
    with open(os.path.join(d, f"{stem}_prompt.txt")) as fh:
        prompt = fh.read().strip()
    return {"id": item_id, "category": category, "stem": stem, "n_refs": n_refs,
            "kind": category.split("_")[1], "refs": refs, "prompt": prompt}


def load_split(split, n=0):
    """-> the split's items, in order; `n` > 0 keeps the first n (still balanced across kinds)."""
    ids = split_ids(split)
    return [load_item(i) for i in (ids[:n] if n else ids)]


def prepare_run_dir(base, item):
    """Create <base>/<category>/ with the references and the prompt linked in; -> (directory, path
    the generated image must be written to, named the way the official judge expects)."""
    d = os.path.join(base, item["category"])
    os.makedirs(d, exist_ok=True)
    for src in item["refs"]:
        dst = os.path.join(d, os.path.basename(src))
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
    p_src = os.path.join(MULTIBANANA, item["category"], f"{item['stem']}_prompt.txt")
    p_dst = os.path.join(d, f"{item['stem']}_prompt.txt")
    if not os.path.exists(p_dst):
        shutil.copyfile(p_src, p_dst)
    return d, os.path.join(d, f"{item['stem']}_generated.png")


# ---- the judge --------------------------------------------------------------------------------
def judge_gpus(min_free_gib=22, at_most=2):
    """`AUTOREF_JUDGE_GPUS` if set; otherwise the emptiest visible GPUs after cuda:0 (where the
    generator loads), at most two. A card another job already fills is skipped rather than shared."""
    env = os.environ.get("AUTOREF_JUDGE_GPUS")
    if env:
        return env.split()
    import torch
    free = {i: torch.cuda.mem_get_info(i)[0] for i in range(1, torch.cuda.device_count())}
    fit = sorted((i for i in free if free[i] >= min_free_gib * 2 ** 30), key=lambda i: -free[i])
    return [str(i) for i in sorted(fit[:at_most])] or ["0"]


def run_judge(out_dir, gpus=None):
    """The official judge over every generated image in `out_dir`; -> path of judge_results.json.

    Greedy decoding occasionally loops and never reaches the score block; the judge's recovery pass
    (a longer budget, then a repetition penalty) re-scores only those answers.
    """
    gpus = gpus or judge_gpus()
    res = os.path.join(out_dir, "judge_results.json")
    if os.path.isfile(res):
        os.remove(res)
    cmd = [sys.executable, "-m", "autoref.judge", "--run-dir", out_dir,
           "--gpus", *gpus]
    print("running the official judge on GPUs", " ".join(gpus), flush=True)
    subprocess.run(cmd, check=True)
    rejudge_unparsed(out_dir, gpus)
    return res


def rejudge_unparsed(out_dir, gpus=None):
    """Re-score the answers in `out_dir` that did not parse; -> how many are still unparsed."""
    res = os.path.join(out_dir, "judge_results.json")
    s = json.load(open(res))["summary"]
    if s["n_parsed"] < s["n"]:
        cmd = [sys.executable, "-m", "autoref.judge", "--run-dir", out_dir,
               "--gpus", *(gpus or judge_gpus()), "--skip-existing",
               "--retry-unparsed", "2048", "--rep-penalty", "1.1"]
        print(f"re-judging {s['n'] - s['n_parsed']} unparsed answer(s)", flush=True)
        subprocess.run(cmd, check=True)
        s = json.load(open(res))["summary"]
    return s["n"] - s["n_parsed"]


def per_item(out_dir):
    """-> {id: {metric: float, ..., "overall": float}} from judge_results.json."""
    p = os.path.join(out_dir, "judge_results.json")
    out = {}
    if not os.path.isfile(p):
        return out
    for r in json.load(open(p)).get("results", []):
        s = r.get("scores") or {}
        if len(s) < len(METRICS):
            continue
        d = {m: float(s[m]) for m in METRICS}
        d["overall"] = sum(d.values()) / len(METRICS)
        out[f"{r['category']}/{r['stem']}"] = d
    return out


def _merge_verdicts_into_traces(out_dir):
    """Fold the judge's scores into each item's trace file."""
    scored = per_item(out_dir)
    tdir = os.path.join(out_dir, "trace")
    for f in os.listdir(tdir) if os.path.isdir(tdir) else []:
        p = os.path.join(tdir, f)
        t = json.load(open(p))
        v = scored.get(t["id"])
        if v is not None:
            t["verdict"] = v
            with open(p, "w") as fh:
                json.dump(t, fh, indent=2, ensure_ascii=False)


# ---- running a harness ----------------------------------------------------------------------
def _trace_name(row):
    return f"{row['category']}__{row['stem']}"


def _strip_image_paths(rec):
    out = []
    for r in rec:
        r = dict(r)
        r.pop("image", None)
        out.append(r)
    return out


def default_out(harness, split, generator=DEFAULT_GENERATOR, mllm=None):
    tag = harness + ("" if generator == DEFAULT_GENERATOR else f"__{generator}") \
        + (f"__{mllm}" if mllm else "")
    return os.path.join(OUTPUTS, "runs", "multibanana", split, tag)


def evaluate(name, split="train", n=0, generator=DEFAULT_GENERATOR, cache="on", workers=3, out=None,
             trace_mode="visual", mllm=None, judge=True, harness_dir=None):
    """Run harness `name` over a split and score it with the official judge.

    `judge=False` stops after the images are written (score later with `--judge-only`).
    """
    assert trace_mode in TRACE_MODES, trace_mode
    t0 = time.time()
    harness = load_harness(name, harness_dir)
    rows = load_split(split, n)
    out_dir = out or default_out(harness.name, split, generator, mllm)
    os.makedirs(os.path.join(out_dir, "trace"), exist_ok=True)
    gen = get_generator(generator)

    def work(row):
        t1 = time.time()
        ctx = Context(generator=gen, task_id=row["id"], cache=cache,
                      save_dir=os.path.join(out_dir, "rounds", _trace_name(row)),
                      **({"mllm": mllm} if mllm else {}))
        err, dst = None, None
        task = {"id": row["id"], "prompt": row["prompt"], "refs": list(row["refs"]),
                "n_refs": row["n_refs"]}
        try:
            img = harness.run(task, ctx)
            _, dst = prepare_run_dir(out_dir, row)
            with open(dst, "wb") as f:
                f.write(img)
        except Exception as e:                  # noqa: BLE001 - recorded, not hidden
            err = f"{e!r}\n{traceback.format_exc()}"
            dst = None
        rec = {"id": row["id"], "prompt": row["prompt"], "n_refs": row["n_refs"], "error": err,
               "calls": dict(ctx.calls), "sec": round(time.time() - t1, 1),
               "record": ctx.record if trace_mode == "visual" else _strip_image_paths(ctx.record)}
        if trace_mode == "visual":
            rec["image"] = os.path.abspath(dst) if dst else None
            rec["refs"] = [os.path.abspath(p) for p in row["refs"]]
        with open(os.path.join(out_dir, "trace", _trace_name(row) + ".json"), "w") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        recs = list(ex.map(work, rows))
    release(gen)
    ok = [r for r in recs if not r["error"]]
    if not ok:
        return {METRIC: 0.0, "n": len(recs), "n_ok": 0, "error": "no images produced",
                "errors": [r["error"].splitlines()[0] for r in recs][:5], "out_dir": out_dir}
    if not judge:
        return {"n": len(recs), "n_ok": len(ok), "out_dir": out_dir, "judged": False,
                "sec": round(time.time() - t0, 1)}
    run_judge(out_dir)
    _merge_verdicts_into_traces(out_dir)
    return summarise(out_dir, recs, round(time.time() - t0, 1))


def summarise(out_dir, recs, wall_sec):
    """-> the result of a judged run: the judge's means over the items it scored."""
    ok = [r for r in recs if not r["error"]]
    scored = per_item(out_dir)
    vals = [v["overall"] for v in scored.values()]
    mean = sum(vals) / len(vals) if vals else 0.0
    sem = ((sum((x - mean) ** 2 for x in vals) / max(len(vals) - 1, 1)) / max(len(vals), 1)) ** 0.5
    by_id = {r["id"]: r for r in recs}
    worst = sorted(scored.items(), key=lambda kv: kv[1]["overall"])[:4]
    result = {METRIC: round(mean, 3), "n": len(recs), "n_ok": len(ok), "n_scored": len(scored),
              "sem": round(sem, 3),
              "per_score": {m: round(sum(v[m] for v in scored.values()) / len(scored), 3)
                            for m in METRICS} if scored else {},
              "mean_generations": round(sum(r["calls"]["generate"] for r in recs) / len(recs), 2),
              "mean_sec": round(sum(r["sec"] for r in ok) / len(ok), 1) if ok else None,
              "calls": {k: sum(r["calls"][k] for r in recs) for k in recs[0]["calls"]},
              "worst": [{"id": i, "score": round(v["overall"], 2),
                         "note": " ".join(f"{m.split()[0][:5]}={v[m]:.0f}" for m in METRICS)
                                 + f" calls={by_id.get(i, {}).get('calls', {})}"}
                        for i, v in worst],
              "errors": [r["error"].splitlines()[0] for r in recs if r["error"]][:5],
              "wall_sec": wall_sec, "out_dir": out_dir}
    with open(os.path.join(out_dir, "harness_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def judge_only(out_dir):
    """Score a run directory written with `judge=False`, from the environment the judge lives in."""
    recs = [json.load(open(os.path.join(out_dir, "trace", f)))
            for f in sorted(os.listdir(os.path.join(out_dir, "trace")))]
    run_judge(out_dir)
    _merge_verdicts_into_traces(out_dir)
    return summarise(out_dir, recs, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--harness", default="autoref_harness",
                    help="a harness in harnesses/ (default autoref_harness) or a path to a .py file")
    ap.add_argument("--split", default="test", choices=SPLIT_NAMES)
    ap.add_argument("--n", type=int, default=0, help="first n items of the split (0 = all)")
    ap.add_argument("--generator", default=DEFAULT_GENERATOR, choices=sorted(GENERATORS))
    ap.add_argument("--mllm", default=None, help="the reasoning model (default gpt-5.5-2026-04-23)")
    ap.add_argument("--cache", default="on", choices=("on", "write", "off"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--trace-mode", default="visual", choices=TRACE_MODES)
    ap.add_argument("--no-judge", action="store_true", help="write the images and stop")
    ap.add_argument("--judge-only", metavar="RUN_DIR", help="score an existing run directory")
    args = ap.parse_args()
    if args.judge_only:
        print(json.dumps(judge_only(args.judge_only), indent=2))
        return
    print(json.dumps(evaluate(args.harness, args.split, args.n, args.generator, args.cache,
                              args.workers, args.out, args.trace_mode, args.mllm,
                              judge=not args.no_judge), indent=2))


if __name__ == "__main__":
    main()
