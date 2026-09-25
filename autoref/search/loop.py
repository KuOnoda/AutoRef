"""The AutoRef search: a coding agent proposes harnesses, a hidden validation split selects them.

    python -m autoref.search --run-name autoref               # B=2, K=4, T=5 on 48 + 48 tasks
    python -m autoref.search --run-name autoref --finalize    # score the final harness on test

Each iteration follows Algorithm 1 of the paper:

    propose    the proposer (Claude Code with the skill in .claude/skills/autoref-proposer) reads
               the search history and writes K candidate harnesses and pending_eval.json
    validate   each candidate must import and define a Harness whose name is its file name
    train      each candidate runs on D_train; its code, scores, judge rationales, traces and
               images join the history the proposer reads
    select     each candidate is scored on D_val, and the B best replace the beam; the history
               records which candidates were kept, never their D_val scores

The search starts from two seed harnesses (autoref/search/seeds/): the generator alone and GEMS.

A run lives in outputs/search/<run-name>/:

    harnesses/   the two seed harnesses and every candidate          (the proposer reads and writes)
    logs/        evolution_summary.jsonl, frontier.json, beam.json, adoption.jsonl, reports/,
                 runs/ (every D_train run), claude_sessions/          (the proposer reads)
    private/     D_val and test scores and runs, selection history    (never mounted)

The loop is adapted from the Meta-Harness reference implementation
(github.com/stanford-iris-lab/meta-harness, reference_examples/text_classification/meta_harness.py):
Phase 0 evaluates the seeds, then propose -> validate -> benchmark -> summarise, one stateless
proposer session per iteration, with memory kept in files.
"""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime

from autoref import multibanana as TASK
from autoref.models import PROPOSER_MODEL
from autoref.paths import MULTIBANANA, ROOT, SEARCH_RUNS, SKILLS, SPLITS

METRIC, METRIC_LABEL = TASK.METRIC, TASK.METRIC_LABEL
SKILL = os.path.join(SKILLS, "autoref-proposer")
SEEDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seeds")
SANDBOX = os.path.join(ROOT, "sandbox")
WRAPPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_wrapper.py")
# Agent is absent on purpose: the reference skill forbids delegating, because constraints get lost
# across a subagent boundary and candidates come back as parameter tweaks
PROPOSER_TOOLS = ["Read", "Glob", "Grep", "Write", "Edit", "Bash"]

def _ts():
    return datetime.now().strftime("%H:%M:%S")


def _c(code, t):
    return f"\033[{code}m{t}\033[0m" if sys.stdout.isatty() else t


bold, dim, green, red, cyan = (lambda t: _c("1", t), lambda t: _c("2", t),
                               lambda t: _c("32", t), lambda t: _c("31", t), lambda t: _c("36", t))


class Run:
    """The files of one search run."""

    def __init__(self, name):
        self.name = name
        self.root = os.path.join(SEARCH_RUNS, name)
        self.harnesses = os.path.join(self.root, "harnesses")
        self.logs = os.path.join(self.root, "logs")
        self.summary = os.path.join(self.logs, "evolution_summary.jsonl")
        self.frontier = os.path.join(self.logs, "frontier.json")
        self.pending = os.path.join(self.logs, "pending_eval.json")
        self.reports = os.path.join(self.logs, "reports")
        self.runs = os.path.join(self.logs, "runs")
        self.beam = os.path.join(self.logs, "beam.json")               # names only
        self.adoption = os.path.join(self.logs, "adoption.jsonl")      # names only
        self.private = os.path.join(self.root, "private")
        self.val_summary = os.path.join(self.private, "val_summary.jsonl")
        self.val_runs = os.path.join(self.private, "val_runs")
        self.selection = os.path.join(self.private, "selection.jsonl")  # with scores
        self.test_summary = os.path.join(self.private, "test_summary.jsonl")
        self.config = os.path.join(self.root, "config.json")

    def create(self, args):
        for d in (self.harnesses, self.logs, self.reports, self.runs, self.private, self.val_runs):
            os.makedirs(d, exist_ok=True)
        for f in sorted(os.listdir(SEEDS)):
            if f.startswith("baseline_") and f.endswith(".py"):
                dst = os.path.join(self.harnesses, f)
                if not os.path.exists(dst):
                    shutil.copyfile(os.path.join(SEEDS, f), dst)
        if not os.path.isfile(self.config):
            with open(self.config, "w") as f:
                json.dump({k: v for k, v in vars(args).items() if k != "func"}, f, indent=2)
        self.build_proposer_view()

    def build_proposer_view(self):
        """The dataset as the proposer's container sees it: the D_train items and the train split
        file, nothing else. Files are hard-linked (copied across filesystems), not symlinked, since
        the view is mounted where the dataset normally is."""
        view = os.path.join(self.root, "proposer_view")
        splits = os.path.join(view, "splits")
        os.makedirs(splits, exist_ok=True)
        shutil.copyfile(os.path.join(SPLITS, "multibanana_train.json"),
                        os.path.join(splits, "multibanana_train.json"))
        for item_id in TASK.split_ids("train"):
            category, stem = item_id.split("/")
            src_dir = os.path.join(MULTIBANANA, category)
            dst_dir = os.path.join(view, "MultiBanana", category)
            os.makedirs(dst_dir, exist_ok=True)
            for f in os.listdir(src_dir):
                if f.startswith(stem + "_"):
                    dst = os.path.join(dst_dir, f)
                    if not os.path.exists(dst):
                        try:
                            os.link(os.path.join(src_dir, f), dst)
                        except OSError:
                            shutil.copyfile(os.path.join(src_dir, f), dst)

    @property
    def seeds(self):
        return sorted(f[:-3] for f in os.listdir(self.harnesses)
                      if f.startswith("baseline_") and f.endswith(".py"))


def read_jsonl(path):
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def ranked(rows):
    return sorted([r for r in rows if r.get(METRIC) is not None], key=lambda r: -r[METRIC])


def write_frontier(run):
    best = ranked(read_jsonl(run.summary))
    with open(run.frontier, "w") as f:
        json.dump({"metric": METRIC, "best": best[:8],
                   "updated": datetime.now().isoformat(timespec="seconds")}, f, indent=2)
    return best[0] if best else None


# ---- the iteration prompt: short, because the instructions live in the skill ----------------
def render_task_prompt(run, iteration, n, beam=None, n_candidates=4):
    text = (
        f"Run iteration {iteration} of the harness evolution loop for task '{TASK.TASK_NAME}'. "
        f"Each candidate will be benchmarked on {n} tasks of the 'train' split.\n\n"
        f"## Run directories\n"
        f"All logs and results for this run are under `{run.logs}/`.\n"
        f"- `{run.summary}` — past results\n"
        f"- `{run.frontier}` — frontier\n"
        f"- `{run.runs}/<harness>.json/harness_result.json` — per-harness scores, tool counts, worst tasks\n"
        f"- `{run.runs}/<harness>.json/trace/<id>.json` — every tool call, per task\n"
        f"- `{run.reports}/` — post-eval reports\n"
        f"- Harness files live in `{run.harnesses}/`\n"
        f"- Write pending_eval.json to: `{run.pending}`")
    if beam:
        bases = beam["bases"]
        per = n_candidates // len(bases)
        listed = ", ".join(f"`{b}`" for b in bases)
        text += (
            f"\n- `{run.adoption}` — which candidates were kept as base harnesses, and which were not"
            f"\n- `{run.beam}` — the current base harnesses"
            f"\n\n## Base harnesses\n"
            f"The current base harnesses are {listed} (files in `{run.harnesses}/`). ")
        if iteration <= 1:
            text += (f"Write {n_candidates} candidates that build on them, in whatever mix you judge "
                     f"best; each candidate's `base_harness` names the one it builds on.")
        else:
            text += (f"Write exactly {n_candidates} candidates: {per} that build on each base. "
                     f"Each candidate's `base_harness` is the exact name of its base.")
        text += (f" Which candidates are kept as bases for the next iteration is decided on tasks "
                 f"outside the 'train' split that this session cannot see: the {len(bases)} "
                 f"candidates that do best there replace the current bases, so a gain has to hold "
                 f"on unseen tasks to be kept.")
    return text


# ---- proposers ------------------------------------------------------------------------------
def clean_tree(run):
    """Refuse to start the proposer if the code it will read carries guidance (sandbox/inspect_contents.py)."""
    checker = os.path.join(SANDBOX, "inspect_contents.py")
    r = subprocess.run([sys.executable, checker], capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "AUTOREF_SEARCH_RUN": run.name})
    if r.returncode == 0:
        print(f"  {dim(r.stdout.strip().splitlines()[-1])}")
        return True
    print(f"  {red('the files the proposer reads carry guidance; not starting it')}")
    print(r.stdout.rstrip())
    return False


def _load_wrapper():
    spec = importlib.util.spec_from_file_location("claude_wrapper", WRAPPER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def propose_claude(run, task_prompt, iteration, args):
    """One stateless `claude -p` session per iteration; its memory is the filesystem."""
    if not clean_tree(run):
        return False
    env_path = os.environ.get("PATH", "")
    if not args.no_sandbox:
        # sandbox/claude runs the CLI in a container that sees only what a proposer may read
        os.environ["PATH"] = SANDBOX + os.pathsep + env_path
        os.environ["AUTOREF_SEARCH_RUN"] = run.name
    if not shutil.which("claude"):
        print(f"  {red('claude CLI not on PATH')}; install Claude Code, or use --proposer manual")
        os.environ["PATH"] = env_path
        return False
    wrapper = _load_wrapper()
    # like the reference, the API key is set aside so the CLI uses subscription auth, unless
    # AUTOREF_PROPOSER_AUTH=api_key asks for the key
    use_key = os.environ.get("AUTOREF_PROPOSER_AUTH") == "api_key"
    saved = None if use_key else os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ.pop("CLAUDECODE", None)
    try:
        r = wrapper.run(prompt=task_prompt, model=args.proposer_model,
                        allowed_tools=PROPOSER_TOOLS, tools=PROPOSER_TOOLS, skills=[SKILL], cwd=ROOT,
                        log_dir=os.path.join(run.logs, "claude_sessions"), name=f"iter{iteration}",
                        timeout_seconds=args.proposer_timeout, effort="max")
    finally:
        if saved:
            os.environ["ANTHROPIC_API_KEY"] = saved
        os.environ["PATH"] = env_path
    if r.exit_code != 0:
        print(f"  {red('proposer failed')} exit={r.exit_code}  {(r.stderr or '').strip()[-300:]}")
        return False
    r.show()
    return os.path.isfile(run.pending)


def propose_manual(run, task_prompt, iteration, args):
    """Hand the iteration to an agent (or a person) outside the loop and wait for pending_eval.json."""
    print(f"\n{bold('PROPOSER (manual)')}: write {run.pending} following {SKILL}/SKILL.md\n")
    print(task_prompt)
    print(f"\n{dim('waiting for pending_eval.json ... (ctrl-c to abort)')}", flush=True)
    while not os.path.isfile(run.pending):
        time.sleep(5)
    return True


PROPOSERS = {"claude": propose_claude, "manual": propose_manual}


# ---- evaluation -----------------------------------------------------------------------------
def validate_candidates(run, candidates):
    """Import-check each candidate before spending compute, as the reference does."""
    valid = []
    for c in candidates:
        try:
            h = TASK.load_harness(c["name"], run.harnesses)
            assert h.name == c["name"], f"name mismatch: {h.name} != {c['name']}"
            valid.append(c)
        except Exception as e:                          # noqa: BLE001 - reported, then skipped
            print(f"    {red('invalid')} {c['name']}: {type(e).__name__}: {e}")
    return valid


def _row(iteration, name, rep, split, args, t0, meta=None):
    rec = {"iteration": iteration, "harness": name, METRIC: rep.get(METRIC, 0.0),
           "sem": rep.get("sem"), "n": rep.get("n"), "n_ok": rep.get("n_ok"),
           "n_scored": rep.get("n_scored"), "per_score": rep.get("per_score"),
           "calls": rep.get("calls"), "wall_sec": round(time.time() - t0, 1),
           "errors": rep.get("errors", []), "split": split, "model": args.generator}
    for k in ("axis", "hypothesis", "components", "base_harness", "mechanism"):
        if (meta or {}).get(k):
            rec[k] = meta[k]
    return rec


def _evaluate(run, name, split, n, out, args):
    try:
        return TASK.evaluate(name, split=split, n=n, generator=args.generator, cache=args.cache,
                             workers=args.workers, out=out, trace_mode="visual",
                             harness_dir=run.harnesses)
    except Exception as e:                              # noqa: BLE001 - recorded as a failed run
        return {METRIC: 0.0, "n": n, "n_ok": 0, "errors": [f"{type(e).__name__}: {e}"],
                "trace": traceback.format_exc()[-1000:]}


def benchmark_train(run, name, args, iteration, meta):
    """Score a candidate on D_train; the run joins the proposer's history."""
    t0 = time.time()
    rep = _evaluate(run, name, "train", args.n, os.path.join(run.runs, f"{name}.json"), args)
    rec = _row(iteration, name, rep, "train", args, t0, meta)
    append_jsonl(run.summary, rec)
    flag = red(f"  {rec['n'] - rec['n_ok']} failed") if (rec.get("n_ok") or 0) < (rec["n"] or 0) else ""
    print(f"    {bold(name):<44} {METRIC_LABEL} {rec[METRIC]:.2f}{flag}", flush=True)
    return rec


def evaluate_val(run, name, args, iteration):
    """Score a candidate on D_val; the row and the run stay in private/. Idempotent."""
    for r in read_jsonl(run.val_summary):
        if r["harness"] == name and r.get("iteration") == iteration and r.get("n_ok") == r.get("n"):
            return r
    t0 = time.time()
    rep = _evaluate(run, name, "val", args.n, os.path.join(run.val_runs, f"VAL_{name}.json"), args)
    rec = _row(iteration, name, rep, "val", args, t0)
    append_jsonl(run.val_summary, rec)
    print(f"    VAL {bold(name):<40} {METRIC_LABEL} {rec[METRIC]:.2f}", flush=True)
    return rec


def val_score(run, name):
    rows = [r for r in read_jsonl(run.val_summary) if r["harness"] == name]
    return rows[-1][METRIC] if rows else None


def read_beam(run):
    return json.load(open(run.beam)) if os.path.isfile(run.beam) else None


def write_beam(run, bases, iteration):
    with open(run.beam, "w") as f:
        json.dump({"bases": list(bases), "since_iteration": iteration}, f, indent=2)


def select_beam(run, iteration, names, args):
    """The B candidates of this iteration with the best D_val score become the next beam.

    Generational: the previous bases do not compete again, so a lucky earlier score never blocks
    the search. The proposer's copy of the decision (beam.json, adoption.jsonl) is names only.
    """
    scored = [(evaluate_val(run, n, args, iteration)[METRIC], n) for n in names]
    if not scored:
        return read_beam(run)
    order = sorted(scored, key=lambda t: -t[0])
    bases = [n for _, n in order[:args.beam]]
    write_beam(run, bases, iteration)
    append_jsonl(run.selection, {"iteration": iteration, "bases": bases,
                                 "ranked": [{"harness": n, METRIC: sc} for sc, n in order]})
    for _, n in sorted(scored, key=lambda t: names.index(t[1])):
        append_jsonl(run.adoption, {"iteration": iteration, "harness": n, "selected": n in bases,
                                    "bases_after": bases})
    print(f"  {green('bases for the next iteration')}: "
          + ", ".join(f"{bold(n)} {sc:.2f}" for sc, n in order[:args.beam]) + "  on val; dropped: "
          + ", ".join(f"{n} {sc:.2f}" for sc, n in order[args.beam:]), flush=True)
    return read_beam(run)


def check_beam_shape(cands, beam, n_candidates, iteration):
    """Warn when the proposer did not write the asked-for number per base; nothing is dropped."""
    bases = beam["bases"]
    if len(cands) != n_candidates:
        print(f"  {red('warning')}: {len(cands)} candidates, {n_candidates} asked for")
    if iteration > 1:
        per = n_candidates // len(bases)
        got = {b: sum(1 for c in cands if c.get("base_harness") == b) for b in bases}
        if any(v != per for v in got.values()):
            print(f"  {red('warning')}: candidates per base {got}, {per} asked for")


def write_report_stub(run, iteration, recs, best_before):
    """A machine-written skeleton; the proposer replaces it with a real reading (skill, Step 0)."""
    path = os.path.join(run.reports, f"iteration_{iteration}.md")
    if os.path.isfile(path):
        return path
    with open(path, "w") as f:
        f.write(f"# Iteration {iteration}\n\nfrontier before: "
                f"{best_before['harness'] if best_before else 'none'} "
                f"{(best_before or {}).get(METRIC, 0):.2f}\n\n")
        for r in recs:
            f.write(f"- **{r['harness']}** {r[METRIC]:.2f} (n_ok {r.get('n_ok')}/{r.get('n')})\n"
                    f"  - hypothesis: {r.get('hypothesis', '')[:200]}\n"
                    f"  - calls: {r.get('calls')}\n")
    return path


# ---- the loop ---------------------------------------------------------------------------------
def run_search(args):
    run = Run(args.run_name)
    run.create(args)
    rows = read_jsonl(run.summary)
    print(f"\n{_ts()} {bold('AutoRef search')}  run={run.name} B={args.beam} K={args.candidates} "
          f"n={args.n} proposer={args.proposer} generator={args.generator}")

    print(f"\n{_ts()} {bold('Phase 0: seeds')}  {run.seeds}")
    done = {r["harness"] for r in rows}
    for s in run.seeds:
        if s not in done:
            benchmark_train(run, s, args, 0, {"axis": "baseline"})
    write_frontier(run)
    for s in run.seeds:
        evaluate_val(run, s, args, 0)
    if read_beam(run) is None:
        write_beam(run, run.seeds, 0)
        for s in run.seeds:
            append_jsonl(run.adoption, {"iteration": 0, "harness": s, "selected": True,
                                        "bases_after": run.seeds})
    print("  bases: " + ", ".join(f"{bold(b)} {val_score(run, b) or 0:.2f}"
                                  for b in read_beam(run)["bases"]) + "  on val")

    start = max([r.get("iteration", 0) for r in read_jsonl(run.summary)
                 if isinstance(r.get("iteration"), int)] + [0]) + 1
    if start > 1:
        finish_interrupted_iteration(run, start - 1, args)
    for it in range(start, start + args.iterations):
        best_before = write_frontier(run)
        print(f"\n{_ts()} {bold(f'Iteration {it}')}  frontier: "
              f"{best_before['harness'] if best_before else 'none'} {(best_before or {}).get(METRIC, 0):.2f}")
        if os.path.isfile(run.pending):
            os.remove(run.pending)
        beam = read_beam(run)
        prompt = render_task_prompt(run, it, args.n, beam, args.candidates)
        if not PROPOSERS[args.proposer](run, prompt, it, args) or not os.path.isfile(run.pending):
            print(f"  {red('no candidates produced; stopping')}")
            break
        cands = json.load(open(run.pending)).get("candidates", [])
        for i, c in enumerate(cands, 1):
            print(f"    {i}. {bold(c['name'])}: {c.get('hypothesis', '')[:90]}")
        check_beam_shape(cands, beam, args.candidates, it)
        valid = validate_candidates(run, cands)
        if not valid:
            print(f"  {red('no candidate passed the import check')}")
            break
        print(f"  {_ts()} {cyan('benchmarking')} {len(valid)} candidate(s) x {args.n} train tasks")
        recs = [benchmark_train(run, c["name"], args, it, c) for c in valid]
        write_report_stub(run, it, recs, best_before)
        best = write_frontier(run)
        delta = best[METRIC] - (best_before or {}).get(METRIC, 0) if best else 0
        print(f"  frontier after iteration {it}: {bold(best['harness'])} {best[METRIC]:.2f} "
              + (green(f"(+{delta:.2f})") if delta > 0 else dim("(unchanged)")))
        select_beam(run, it, [r["harness"] for r in recs if r.get("n_ok") == r.get("n")], args)
    print(f"\n{_ts()} {bold('Search complete.')}  Next: python -m autoref.search --run-name "
          f"{run.name} --finalize")


def finish_interrupted_iteration(run, it, args):
    """On resume, benchmark the candidates of iteration `it` that a crash left unscored, and run its
    selection if it never ran; otherwise the next iteration would start from a stale beam."""
    pending = json.load(open(run.pending)) if os.path.isfile(run.pending) else {}
    if pending.get("iteration") == it:
        done = {r["harness"] for r in read_jsonl(run.summary) if r.get("iteration") == it}
        todo = [c for c in pending.get("candidates", []) if c["name"] not in done]
        for c in validate_candidates(run, todo):
            benchmark_train(run, c["name"], args, it, c)
    if not any(r.get("iteration") == it for r in read_jsonl(run.selection)):
        select_beam(run, it, [r["harness"] for r in read_jsonl(run.summary)
                              if r.get("iteration") == it and r.get("n_ok") == r.get("n")], args)


def final_harness(run):
    """The member of the final beam with the best D_val score."""
    return max(read_beam(run)["bases"], key=lambda b: val_score(run, b) or float("-inf"))


def run_finalize(args):
    run = Run(args.run_name)
    if not os.path.isfile(run.config):
        sys.exit(f"no search run at {run.root}")
    name = args.harness or final_harness(run)
    print(f"\n{_ts()} {bold('Final harness')}: {name}  -> test ({args.test_n or 'all'} items)")
    t0 = time.time()
    rep = TASK.evaluate(name, split="test", n=args.test_n, generator=args.generator, cache=args.cache,
                        workers=args.workers, out=os.path.join(run.private, "test_runs", f"TEST_{name}.json"),
                        trace_mode="visual", harness_dir=run.harnesses)
    rec = _row("test", name, rep, "test", args, t0)
    append_jsonl(run.test_summary, rec)
    print(f"  TEST {bold(name)} {METRIC_LABEL} {rep.get(METRIC, 0):.3f}  (n={rep.get('n')})")


def main():
    ap = argparse.ArgumentParser(description="The AutoRef harness search.")
    ap.add_argument("--run-name", required=True, help="outputs/search/<run-name>/")
    ap.add_argument("--beam", type=int, default=2, help="B, the beam width")
    ap.add_argument("--candidates", type=int, default=4, help="K, candidates per iteration")
    ap.add_argument("--iterations", type=int, default=5, help="T (resumes after the last finished one)")
    ap.add_argument("--n", type=int, default=48, help="tasks of D_train and D_val per evaluation")
    ap.add_argument("--generator", default=TASK.DEFAULT_GENERATOR, choices=sorted(TASK.GENERATORS))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--cache", default="write", choices=("on", "write", "off"),
                    help="'write' does not read the cache, so every candidate pays for its own calls "
                         "and their timings compare (the paper's setting)")
    ap.add_argument("--proposer", default="claude", choices=sorted(PROPOSERS))
    ap.add_argument("--proposer-model", default=PROPOSER_MODEL)
    ap.add_argument("--proposer-timeout", type=int, default=10800,
                    help="seconds per proposer session; prototyping generates and judges images")
    ap.add_argument("--no-sandbox", action="store_true",
                    help="run the proposer on the host instead of in the container (not recommended: "
                         "it runs with --dangerously-skip-permissions)")
    ap.add_argument("--init-only", action="store_true",
                    help="create the run directory (seeds, the proposer's view of the data) and stop")
    ap.add_argument("--finalize", action="store_true", help="score the final harness on the test split")
    ap.add_argument("--harness", default=None, help="with --finalize: score this harness instead")
    ap.add_argument("--test-n", type=int, default=0, help="with --finalize: first n test items (0 = all 133)")
    args = ap.parse_args()
    if args.init_only:
        run = Run(args.run_name)
        run.create(args)
        print(f"created {run.root}")
        return
    (run_finalize if args.finalize else run_search)(args)


if __name__ == "__main__":
    main()
