---
name: autoref-proposer
description: Run one iteration of harness evolution for multi-reference image generation. Called by autoref/search/loop.py.
---

# Meta-Harness (Multi-Reference Image Generation)

Run ONE iteration of harness evolution. Do all work in the main session — do NOT delegate to subagents. Constraints get lost when you delegate, leading to parameter-only changes and skipped prototyping.

**You do NOT run the benchmark.** You analyse results and traces, prototype changes, and implement new harnesses. The outer loop (`autoref/search/loop.py`) handles benchmarking separately.

## CRITICAL CONSTRAINTS

- You MUST implement 3 new harnesses every iteration.
- Do NOT write "the frontier is optimal" or "stop iterating", or abort early.
- ALWAYS complete all steps including prototyping.
- Design as many candidates as the task prompt asks for (3 when it does not say): mix of exploitation and exploration.

## The task being harnessed

A harness is the Python around two FROZEN models that turns one task — a text prompt plus four reference images — into ONE image. The image model, the LLM and the judge are fixed and NOT yours to change.

The references show the subjects, objects, backgrounds or styles the prompt refers to; the output has to follow the prompt while staying consistent with them.

Score = the official judge (a vision-language model shown the references, the prompt and the output) on five metrics, each 1–10: Instruction Alignment, Reference Consistency, Background-Subject Match, Physical Realism, Visual Quality. The score is their mean.

## Harness interface

```python
from autoref.harness import Harness

class MyHarness(Harness):
    name = "my_harness"          # must equal the file name (no .py)
    description = "one line"
    def run(self, task, ctx):
        # task = {"id": str, "prompt": str, "refs": [image paths], "n_refs": int}
        return ctx.generate(task["prompt"], images=task["refs"])   # -> image bytes
```

`ctx` gives you, and nothing else:
- `ctx.think(prompt, images=[...]) -> str` — the frozen LLM (gpt-5.5). `images` are bytes or paths; `<image>` placeholders in the prompt take them in order.
- `ctx.generate(prompt, images=[...]) -> bytes` — the frozen image model. `images` are bytes or paths the model conditions on. Call it as many times as your mechanism needs; every call is counted and reported as `mean_generations`.
- `ctx.calls` — the running count of every call this run has made.

Every call is cached on disk by its arguments, so repeating another harness's question is free.

`autoref/harness.py` also has `load_skills()`, which returns the skill instructions under `external/GEMS/agent/skills/` as a dict.

### Anti-parameter-tuning rules

The most common failure mode is creating harnesses that are just parameter variants of existing ones. Check `evolution_summary.jsonl` for what's been tried — sweeps (how many rounds, how many questions, how many references to pass) almost always regress or tie.

**Good candidates change a fundamental mechanism:**
- A new verification design (e.g. graded checks instead of yes/no, or checks that compare against a specific reference)
- A new refinement architecture (e.g. separate what must be kept from what must change)
- A new control flow (e.g. let the first verification decide how many rounds to run)
- A new rule for what to return

**Bad candidates just tune numbers.** If `run()` is identical to an existing harness except for constants, it is a parameter variant. Rewrite with a genuinely novel mechanism.

**Combining harnesses is valid.** Take the verification from A and the refinement from B.

### Anti-overfitting rules

- **No task-specific hints.** Do not hardcode knowledge about particular prompts or subjects.
- **Never mention the benchmark name** in harness code, prompts, or comments.
- **General patterns are OK.** "Check the subject before the background" applies broadly.
- Never read the validation or test splits. `ctx` is your only access to the models.

## WORKFLOW

**Do ALL steps yourself in the main session.**

### Step 0: Post-eval reports (write if missing)

For each past iteration that has results in `evolution_summary.jsonl` but NO report in `reports/`, write one. Each report is **<=30 lines** covering: what changed, which tasks improved/regressed and why, and a takeaway for future iterations.

### Step 1: Analyse

1. **Read all state files:**
   - `evolution_summary.jsonl` — what's been tried (one JSON per candidate)
   - `frontier.json` — current best
   - `runs/<harness>.json/harness_result.json` — scores, call counts, `worst` tasks
   - `runs/<harness>.json/trace/<id>.json` — per task: the prompt, every call in order,
     the judge's five scores, and `image` / `refs` / each generated round under `rounds/`

2. Formulate 3 hypotheses — each must be falsifiable and target a different mechanism.

### Step 2: Prototype — MANDATORY

**You MUST prototype your mechanism before writing the final harness.** Do NOT skip this. Candidates that skip prototyping tend to have bugs or produce no improvement.

Run everything from the repository root, with the virtualenv and the API keys loaded. A prototype run loads the image model and the judge, so allow a few minutes:

```bash
set -a && . ./.env && set +a           # OPENAI_API_KEY
.venv/bin/python -m autoref.multibanana --harness <harnesses>/<name>.py --split train --n 2 \
    --trace-mode visual --out /tmp/proto_<name>
```

`<harnesses>` is the harness directory named in the task prompt.

For each candidate:
1. Run it against 1-2 real tasks with the command above.
2. Read `/tmp/proto_<name>/harness_result.json`: `n_ok` must equal `n`, `errors` must be empty,
   and `calls` must show what you intended actually fired.
3. Try 2-3 variants and compare before picking one.
4. Delete `/tmp/proto_*` when done.

### Step 3: Implement

1. Copy a top-performing harness to `<harnesses>/<name>.py`, then modify. Copy-then-edit keeps the imports and proven patterns correct.
2. Implement the new mechanism according to your hypothesis.
3. **Self-critique (mandatory):** re-read the file. Does this introduce a genuinely NEW mechanism, or is it a parameter variant? If `run()` differs from the base only in numbers, REWRITE.
4. Validate: `.venv/bin/python -c "from autoref.harness import load_harness; print(load_harness('<harnesses>/<name>.py').name)"`

### Step 4: Write pending_eval.json

Write to the path given in the task prompt:

```json
{
  "iteration": 2,
  "candidates": [
    {
      "name": "snake_case_name",
      "file": "<harnesses>/snake_case_name.py",
      "hypothesis": "falsifiable claim about why the judge's score rises",
      "axis": "exploitation|exploration",
      "base_harness": "what it builds on",
      "components": ["tag1", "tag2"]
    }
  ]
}
```

Output: `CANDIDATES: <name1>, <name2>, <name3>`

## evolution_summary.jsonl format

One JSON object per line, one per evaluated candidate:

```json
{"iteration": 1, "harness": "example_harness", "overall": 6.0, "sem": 0.4,
 "n_ok": 48, "axis": "exploitation", "hypothesis": "...", "delta": +0.3,
 "components": ["tag1", "tag2"]}
```
