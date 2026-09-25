"""Score MultiBanana outputs with the official Qwen3-VL-8B judge.

The benchmark's `qwenvl_judge.py` hardcodes its base directory and loads the model at import, so it
is not imported. Instead the same prompt goes to the same model here, sharded across GPUs, and the
scores are parsed with the official regexes. The prompt is read out of the official `judge.py`
(external/multibanana, fetched by scripts/setup_data.sh) at run time, so the wording cannot drift
from the published one. Decoding is greedy, so judging is deterministic on a given GPU type.

    python -m autoref.judge --run-dir outputs/runs/multibanana/test/autoref_harness --gpus 0
"""
import argparse
import base64
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

# transformers' lazy module is not thread-safe: importing these inside the shard threads makes
# concurrent shards fail with "cannot import name AutoProcessor". Import once, up front.
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from autoref.models import JUDGE_MODEL, revision
from autoref.paths import MULTIBANANA_OFFICIAL

OFFICIAL = os.path.join(MULTIBANANA_OFFICIAL, "judge.py")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
METRICS = ["Instruction Alignment", "Reference Consistency", "Background-Subject Match",
           "Physical Realism", "Visual Quality"]
PATTERNS = {m: re.compile(rf"{re.escape(m)}:\s*(\d+)") for m in METRICS}


def official_prompts(instruction):
    """create_evaluation_prompts() taken straight from the official judge.py."""
    src = open(OFFICIAL).read()
    ns = {}
    start = src.index("def create_evaluation_prompts")
    end = src.index("\ndef ", start + 1)
    exec(src[start:end], ns)                      # noqa: S102 - the official prompt builder
    return ns["create_evaluation_prompts"](instruction)


def parse_scores(text):
    out = {}
    for m, pat in PATTERNS.items():
        hit = pat.search(text)
        if hit:
            out[m] = int(hit.group(1))
    return out if len(out) == len(METRICS) else None


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# `device_map` streams the weights straight onto the target GPU instead of materialising the full
# checkpoint in host memory first, and the lock keeps two shards from loading at the same moment.
_LOAD_LOCK = threading.Lock()


class Judge:
    def __init__(self, device):
        with _LOAD_LOCK:
            rev = revision(JUDGE_MODEL)
            self.proc = AutoProcessor.from_pretrained(JUDGE_MODEL, revision=rev)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                JUDGE_MODEL, revision=rev, dtype="auto", device_map={"": device}).eval()
        self.device = device

    def score(self, refs, generated, instruction, max_new_tokens=1536, repetition_penalty=None):
        """The judge reasons at length before the score block, so the token budget is generous."""
        a, b, c = official_prompts(instruction)
        content = [{"type": "text", "text": a}]
        for r in refs:
            content.append({"type": "image", "image": f"data:image/jpeg;base64,{b64(r)}"})
        content.append({"type": "text", "text": b})
        content.append({"type": "image", "image": f"data:image/png;base64,{b64(generated)}"})
        content.append({"type": "text", "text": c})
        inputs = self.proc.apply_chat_template([{"role": "user", "content": content}],
                                               tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt").to(self.device)
        kw = {"repetition_penalty": repetition_penalty} if repetition_penalty else {}
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, **kw)
        text = self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)[0]
        return text, parse_scores(text)


def collect(run_dir):
    """-> [(category, stem, [refs], generated, prompt)] for every generated image present."""
    jobs = []
    for cat in sorted(os.listdir(run_dir)):
        d = os.path.join(run_dir, cat)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith("_generated.png"):
                continue
            stem = f[: -len("_generated.png")]
            # only reference images: the judge writes <stem>_qwenvl_judge.txt into the same
            # directory, and a re-run must not feed it back in as a reference
            refs = sorted(os.path.join(d, p) for p in os.listdir(d)
                          if p.startswith(stem + "_") and p.lower().endswith(IMAGE_EXTS)
                          and not p.endswith("_generated.png"))
            prompt_path = os.path.join(d, f"{stem}_prompt.txt")
            if refs and os.path.isfile(prompt_path):
                jobs.append((cat, stem, refs, os.path.join(d, f),
                             open(prompt_path).read().strip()))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", nargs="+", required=True)
    ap.add_argument("--gpus", nargs="+", default=["0"])
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--retry-unparsed", type=int, default=0,
                    help="re-judge cached answers that did not parse, with this token budget")
    ap.add_argument("--rep-penalty", type=float, default=0.0,
                    help="last-resort decoding penalty for answers stuck in a repetition loop")
    args = ap.parse_args()

    for run_dir in args.run_dir:
        out_path = os.path.join(run_dir, "judge_results.json")
        if args.skip_existing and os.path.isfile(out_path):
            # only a fully parsed directory is finished; a truncated one is re-entered, reusing the
            # cached answers and re-judging only the gaps
            s = json.load(open(out_path))["summary"]
            if s["n_parsed"] == s["n"]:
                print(f"skip {run_dir}")
                continue
            print(f"re-entering {run_dir}: {s['n'] - s['n_parsed']} unparsed", flush=True)
        jobs = collect(run_dir)
        print(f"\n=== {run_dir}: {len(jobs)} images, {len(args.gpus)} GPUs", flush=True)
        shards = [jobs[i::len(args.gpus)] for i in range(len(args.gpus))]
        resume, retry_budget, rep_penalty = args.skip_existing, args.retry_unparsed, args.rep_penalty

        def run_shard(pair):
            gpu, shard = pair
            if not shard:
                return []

            def _parsed(g, st):
                c = os.path.join(os.path.dirname(g), f"{st}_qwenvl_judge.txt")
                return os.path.isfile(c) and parse_scores(open(c).read()) is not None

            judge = None if resume and all(_parsed(g, st) for _, st, _, g, _ in shard) \
                else Judge(f"cuda:{gpu}")
            res = []
            for cat, stem, refs, gen, prompt in shard:
                cached = os.path.join(os.path.dirname(gen), f"{stem}_qwenvl_judge.txt")
                text = scores = None
                if resume and os.path.isfile(cached):
                    text = open(cached).read()          # judging is deterministic; reuse it
                    scores = parse_scores(text)
                if scores is None:
                    budget = retry_budget if (text is not None and retry_budget) else 1536
                    text, scores = judge.score(refs, gen, prompt, max_new_tokens=budget)
                    if scores is None and rep_penalty:
                        # greedy decoding sometimes loops and never reaches the score block; same
                        # model, prompt and greedy search, with only the loop penalised
                        text, scores = judge.score(refs, gen, prompt, max_new_tokens=budget,
                                                   repetition_penalty=rep_penalty)
                    with open(cached, "w") as f:
                        f.write(text)
                res.append({"category": cat, "stem": stem, "n_refs": len(refs),
                            "kind": cat.split("_")[1], "scores": scores})
                print(f"  {cat}/{stem}: {scores}", flush=True)
            return res

        with ThreadPoolExecutor(max_workers=len(args.gpus)) as ex:
            results = [r for shard in ex.map(run_shard, zip(args.gpus, shards)) for r in shard]

        ok = [r for r in results if r["scores"]]
        summary = {"n": len(results), "n_parsed": len(ok)}
        if ok:
            summary["mean"] = {m: round(sum(r["scores"][m] for r in ok) / len(ok), 3)
                               for m in METRICS}
            summary["overall"] = round(sum(sum(r["scores"].values()) / len(METRICS)
                                           for r in ok) / len(ok), 3)
            by_n = {}
            for r in ok:
                by_n.setdefault(r["n_refs"], []).append(sum(r["scores"].values()) / len(METRICS))
            summary["by_n_refs"] = {k: round(sum(v) / len(v), 3) for k, v in sorted(by_n.items())}
        with open(out_path, "w") as f:
            json.dump({"judge": JUDGE_MODEL, "summary": summary, "results": results}, f, indent=2)
        print(json.dumps(summary, indent=2))
        print("saved", out_path)


if __name__ == "__main__":
    main()
