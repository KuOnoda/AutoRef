"""The search loop's bookkeeping, with evaluation stubbed out: what the proposer is told, and how the
beam is selected. No GPU, API key or proposer is needed."""
import json
import os
import types

import pytest

from autoref.search import loop


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "SEARCH_RUNS", str(tmp_path))
    r = loop.Run("t")
    for d in (r.harnesses, r.logs, r.reports, r.runs, r.private, r.val_runs):
        os.makedirs(d, exist_ok=True)
    return r


def fake_evaluate(scores):
    def evaluate(name, split, n, **_):
        return {"overall": scores[(name, split)], "n": n, "n_ok": n, "n_scored": n}
    return evaluate


ARGS = types.SimpleNamespace(beam=2, n=48, generator="flux-klein-4b", cache="write", workers=1)


def test_the_beam_is_the_top_b_of_this_iteration_and_the_proposer_sees_names_only(run, monkeypatch):
    scores = {("a", "val"): 6.1, ("b", "val"): 7.0, ("c", "val"): 6.9, ("d", "val"): 5.0}
    monkeypatch.setattr(loop.TASK, "evaluate", fake_evaluate(scores))
    loop.write_beam(run, ["old1", "old2"], 0)
    beam = loop.select_beam(run, 1, ["a", "b", "c", "d"], ARGS)
    assert beam["bases"] == ["b", "c"]                     # generational: old bases do not compete
    adoption = loop.read_jsonl(run.adoption)
    assert {r["harness"]: r["selected"] for r in adoption} == {"a": False, "b": True, "c": True, "d": False}
    for r in adoption:                                     # the proposer's copy carries no score
        assert "overall" not in r and all("overall" not in str(v) for v in r.values())
    private = loop.read_jsonl(run.selection)[0]
    assert [x["harness"] for x in private["ranked"]] == ["b", "c", "a", "d"]


def test_validation_rows_stay_private_and_are_not_recomputed(run, monkeypatch):
    calls = []

    def evaluate(name, split, n, **_):
        calls.append(name)
        return {"overall": 6.0, "n": n, "n_ok": n}
    monkeypatch.setattr(loop.TASK, "evaluate", evaluate)
    loop.evaluate_val(run, "a", ARGS, 1)
    loop.evaluate_val(run, "a", ARGS, 1)
    assert calls == ["a"]
    assert os.path.isfile(run.val_summary) and run.val_summary.startswith(run.private)
    assert not os.path.exists(os.path.join(run.logs, "val_summary.jsonl"))


def test_the_iteration_prompt_matches_the_protocol(run):
    beam = {"bases": ["complaint_directed_pick", "framing_ladder"]}
    text = loop.render_task_prompt(run, 5, 48, beam, 4)
    assert "benchmarked on 48 tasks of the 'train' split" in text
    assert "Write exactly 4 candidates: 2 that build on each base." in text
    assert "outside the 'train' split that this session cannot see" in text
    assert run.private not in text and "val_summary" not in text   # nothing private is named


def test_the_final_harness_is_the_best_beam_member_on_validation(run):
    loop.write_beam(run, ["x", "y"], 5)
    for name, s in (("x", 7.1), ("y", 7.3), ("z", 7.9)):
        loop.append_jsonl(run.val_summary, {"harness": name, "overall": s, "iteration": 5})
    assert loop.final_harness(run) == "y"


def test_the_beam_shape_check_only_warns(capsys):
    loop.check_beam_shape([{"name": "a", "base_harness": "x"}], {"bases": ["x", "y"]}, 4, 2)
    out = capsys.readouterr().out
    assert "1 candidates, 4 asked for" in out


def test_a_resumed_run_selects_for_an_iteration_a_crash_interrupted(run, monkeypatch):
    scores = {("a", "val"): 6.0, ("b", "val"): 7.0}
    monkeypatch.setattr(loop.TASK, "evaluate", fake_evaluate(scores))
    for name in ("a", "b"):
        loop.append_jsonl(run.summary, {"iteration": 1, "harness": name, "overall": 6.0, "n": 48, "n_ok": 48})
    loop.finish_interrupted_iteration(run, 1, ARGS)
    assert loop.read_beam(run)["bases"] == ["b", "a"]
    loop.finish_interrupted_iteration(run, 1, ARGS)                      # idempotent
    assert len(loop.read_jsonl(run.selection)) == 1
