"""AutoRef-Harness and the search's seed harnesses, run against a fake context (no GPU or API key)."""
import json
import os

from autoref.harness import load_harness
from autoref.search.loop import SEEDS


class FakeContext:
    """Answers each reasoning-model prompt by its kind and numbers the generated images."""

    def __init__(self, check=None, winner=("B", "A"), complaints=("the hat from image 2 is missing",)):
        self.calls = {"think": 0, "generate": 0}
        self.samples, self.prompts, self.record = [], [], []
        self.check = list(check or [])
        self.winner = list(winner)
        self.complaints = list(complaints)

    def generate(self, prompt, images=(), sample=None):
        self.calls["generate"] += 1
        self.samples.append(sample)
        self.prompts.append(prompt)
        return f"draft-{self.calls['generate']}".encode()

    def think(self, prompt, images=()):
        self.calls["think"] += 1
        if "HARD failures" in prompt:
            return json.dumps(self.check.pop(0) if self.check else {"missing": [], "extra": [],
                                                                     "wrong_background": False})
        if "Candidate A: <image>" in prompt:
            w = self.winner.pop(0) if self.winner else "A"
            return json.dumps({"differences": ["..."], "winner": w, "margin": 3})
        if "strict rater will grade" in prompt:
            return json.dumps({"complaints": self.complaints})
        if "as the BACKGROUND, SETTING or LOCATION" in prompt:
            return json.dumps({"background": 3, "style": 0})
        return ("A sunny beach at noon, light from the upper left, a photograph. The dog from image 1 "
                "wears the hat from image 2 and sits on the sand of image 3; one light, contact shadows.")


TASK = {"id": "t", "prompt": "The dog from image 1 wears the hat from image 2 on the beach of image 3.",
        "refs": [b"ref1", b"ref2", b"ref3"], "n_refs": 3}


def test_autoref_harness_loads():
    assert load_harness("autoref_harness").name == "autoref_harness"


def test_three_images_and_the_canvas_draft_wins_in_both_orders():
    # drafts A (scene) and B (canvas) pass the check and B wins in both orders; the revised draft C
    # then loses one order, so draft B is returned
    ctx = FakeContext(winner=["B", "A", "B", "B"])
    out = load_harness("autoref_harness").run(TASK, ctx)
    assert ctx.calls["generate"] == 3
    assert out == b"draft-2"
    assert "hat" in ctx.prompts[2]                              # the revision answers the complaint


def test_hard_failures_decide_before_the_pairwise_rater():
    missing = {"missing": ["image 2"], "extra": [], "wrong_background": False}
    clean = {"missing": [], "extra": [], "wrong_background": False}
    ctx = FakeContext(check=[missing, clean, clean], winner=["A", "A"])
    assert load_harness("autoref_harness").run(TASK, ctx) == b"draft-2"


def test_seed_harnesses_load():
    for name in ("baseline_plain", "baseline_gems"):
        assert load_harness(name, SEEDS).name == name


def test_the_generator_alone():
    ctx = FakeContext()
    assert load_harness("baseline_plain", SEEDS).run(TASK, ctx) == b"draft-1"
    assert ctx.prompts == [TASK["prompt"]]


def test_a_harness_can_be_loaded_from_a_file():
    path = os.path.join(SEEDS, "baseline_plain.py")
    assert load_harness(path).name == "baseline_plain"
