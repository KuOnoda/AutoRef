"""The split files: sizes, balance across task kinds, and no item in two MultiBanana splits."""
import collections
import json
import os

from autoref.paths import SPLITS

SIZES = {"train": 48, "val": 48, "test": 133, "ref3": 96, "ref5": 96}


def load(name):
    return json.load(open(os.path.join(SPLITS, name + ".json")))


def test_sizes_and_reference_counts():
    for split, n in SIZES.items():
        d = load(f"multibanana_{split}")
        assert d["n"] == n == len(d["ids"]) == len(set(d["ids"]))
        expected = {"ref3": 3, "ref5": 5}.get(split, 4)
        assert all(i.split("/")[0].startswith(f"{expected}_") for i in d["ids"]), split
        assert d["revision"] == "6c682f06c8c8d4e6760d7804901c074866dc1da1"


def test_multibanana_splits_are_disjoint():
    seen = collections.Counter()
    for split in SIZES:
        seen.update(load(f"multibanana_{split}")["ids"])
    assert max(seen.values()) == 1


def test_search_splits_are_balanced_across_kinds():
    for split in ("train", "val", "ref3", "ref5"):
        kinds = collections.Counter(i.split("/")[0].split("_")[1] for i in load(f"multibanana_{split}")["ids"])
        assert len(set(kinds.values())) == 1, (split, kinds)


def test_prefixes_cycle_through_the_kinds():
    ids = load("multibanana_train")["ids"]
    assert [i.split("/")[0] for i in ids[:4]] == ["4_back", "4_global", "4_local", "4_object"]
