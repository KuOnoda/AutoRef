"""The files mounted into the proposer's container carry no past results or guidance, and the GEMS
prompts fetched at run time are the ones the paper used."""
import os
import subprocess
import sys

import pytest

from autoref.paths import GEMS, ROOT


def test_mounted_sources_pass_the_content_check():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "sandbox", "inspect_contents.py")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout


def test_autoref_harness_and_the_search_loop_are_not_mounted():
    r = subprocess.run([os.path.join(ROOT, "sandbox", "claude")], capture_output=True, text=True,
                       env={**os.environ, "AUTOREF_SEARCH_RUN": "t", "AUTOREF_PRINT_MOUNTS": "1"})
    mounted = r.stdout.split()
    assert mounted and not any("/harnesses" in p or "/search" in p or p.endswith("README.md")
                               for p in mounted)


@pytest.mark.skipif(not os.path.isdir(os.path.join(GEMS, "agent")), reason="GEMS not fetched")
def test_gems_prompts_match_the_paper():
    import importlib.util
    from autoref.search.loop import SEEDS
    spec = importlib.util.spec_from_file_location("seed_gems", os.path.join(SEEDS, "baseline_gems.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert set(mod.gems_prompts()) == set(mod.GEMS_PROMPTS)       # raises if a hash differs
