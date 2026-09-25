# The proposer's container

The proposer is the Claude Code CLI with `--dangerously-skip-permissions`, so it runs in a container
that holds only what it may read. `sandbox/claude` stands in for `claude` on PATH (the search loop
puts it first); the flags, prompt and streamed output pass through unchanged. Everything is mounted
at its host path; what is not mounted does not exist inside.

| Mounted | |
|---|---|
| read-only | the `claude` binary, `.venv/`, the package files a harness imports (`autoref/{paths,models,harness,generators,multibanana,judge}.py`), the skill, `.env`, the judge's prompt, the GEMS prompts and skills, **the D<sub>train</sub> items and the train split file only** |
| writable | the run's `harnesses/` and `logs/`, its own model-call cache, `~/.cache/huggingface/`, the Claude credentials in an otherwise empty `~/.claude` |

Not in the container: the validation and test items and their split files, the run's `private/`
directory (D<sub>val</sub> and test scores), other runs, the shared model-call cache, `harnesses/`,
the search loop, the documentation and `.git`. The proposer can read every key in `.env`, so keep only
`OPENAI_API_KEY` there for a search.

```bash
docker build -t autoref-proposer:latest sandbox/
python -m autoref.search --run-name <run> --init-only      # create a run without starting it
bash sandbox/verify_isolation.sh <run>                      # try every route out of the container
```

`sandbox/inspect_contents.py` reads every mounted source file and fails on past results, conclusions
about which mechanism wins, or text addressed to the proposer; the search loop runs it before every
proposer session.

| Variable | Default | |
|---|---|---|
| `AUTOREF_SANDBOX_IMAGE` | `autoref-proposer:latest` | |
| `AUTOREF_SANDBOX_GPUS` | `--gpus all` | or e.g. `--device nvidia.com/gpu=0 --device nvidia.com/gpu=1` (CDI) |
| `AUTOREF_PROPOSER_AUTH` | `subscription` | `api_key` passes `ANTHROPIC_API_KEY` into the container |

A virtualenv built on the host's system Python must match the image's Python (3.12 in the provided
Dockerfile); one built with uv, pyenv or conda has its interpreter mounted in instead.
