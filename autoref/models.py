"""Model identifiers, and the revisions they are pinned to."""

# the reasoning model inside every harness (a dated snapshot, not a moving alias)
REASONING_MODEL = "gpt-5.5-2026-04-23"

# the coding agent that writes harnesses during a search (Claude Code CLI)
PROPOSER_MODEL = "claude-fable-5-1"

# the MultiBanana judge
JUDGE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# Hugging Face repository -> commit
REVISIONS = {
    "black-forest-labs/FLUX.2-klein-4B": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
    "black-forest-labs/FLUX.2-klein-9B": "92196c8e11f7b6cf2b7493e037d8c5345c559216",
    "Qwen/Qwen-Image-Edit-2511": "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9",
    "Qwen/Qwen3-VL-8B-Instruct": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
}

# the dataset and the third-party code scripts/setup_data.sh fetches
DATASET_REVISION = ("kohsei/MultiBanana-Benchmark", "6c682f06c8c8d4e6760d7804901c074866dc1da1")
CODE_REVISIONS = {
    "matsuolab/multibanana": "04b7c454be2967b422b06080f5250ab2a1ec63ee",     # the judge's prompt
    "lcqysl/GEMS": "fec33ff64b65efa55787616d504468c801776579",               # the search's seed harness
}


def revision(repo):
    """The pinned commit for `repo`, or None for an unpinned repository."""
    return REVISIONS.get(repo)
