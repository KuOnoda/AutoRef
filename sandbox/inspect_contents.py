"""Read what the proposer's container mounts, not just which paths it can reach.

A file the proposer opens to check an import is read in full, docstrings and comments included, so
guidance can arrive without ever being an instruction: a past result quoted in a docstring, a
conclusion about which mechanism wins, a note explaining the experiment to the agent it is run on.
This scans the source files `sandbox/claude` mounts (it asks the shim for the list) and exits
non-zero if any of them carries such text. The search loop runs it before every proposer session.

    AUTOREF_SEARCH_RUN=<run> python sandbox/inspect_contents.py
"""
import collections
import os
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
RUN = os.environ.get("AUTOREF_SEARCH_RUN", "")
OUTPUTS = pathlib.Path(os.environ.get("AUTOREF_OUTPUTS", REPO / "outputs"))


def mounted_sources():
    out = subprocess.run([str(HERE / "claude")], env={**os.environ, "AUTOREF_SEARCH_RUN": RUN or "_",
                                                      "AUTOREF_PRINT_MOUNTS": "1"},
                         capture_output=True, text=True, check=True).stdout.split()
    return out


# The run's own harness directory and logs are not scanned: the candidates and reports quote each
# other's scores, and that is the search's own history, which the proposer is meant to read.
RULES = [
    ("a candidate named outside the harness directory", None),
    ("a conclusion rather than a fact about the code",
     r"\b(we|I) (found|measured|saw|observed)\b|\bin practice\b|\bit turns out\b"),
    ("a claim about which harness wins", r"\b(better|worse|higher|lower) than\b|\boutperform"),
    ("a tally from a run", r"\b\d+\s*(of|/)\s*(48|96|133)\b"),
    ("a tally written in words",
     r"\b(one|two|three|four|five|six|seven|eight|nine|ten) of (one|two|three|four|five|six|seven"
     r"|eight|nine|ten|forty-eight)\b"),
    ("a score from a run", r"(?<![\w.-])[4-9]\.\d{2,3}(?![\w.])"),
    ("addressed at or about the proposer",
     r"\bthe proposer\b(?!'s container)|\ba proposer\b|\bwhoever reads\b"),
    ("a reference to the paper or its results",
     r"\bthe paper\b|\bAutoRef-Harness\b|\bTable \d+\b|\bFigure \d+\b|\bheadline\b"
     r"|\bcanvas_pair_pick\b|\brestyle_gate_rank\b"),
    ("a hint about how the task fails",
     r"\boverrun by\b|\bcrowd(s|ed)? out\b|\bdrowns?\b|\bcompete for\b|\bpasted\b|\bcomposited look\b"),
    ("a narrative about what happened here",
     r"\bwhich is what happened\b|\b(was|were) archived\b|\bthe first search\b"
     r"|\b(candidates?|harnesses|iterations?) (over|across) \w+ (iterations?|rounds?)\b"
     r"|\b(left|lost|crashed|regressed|collapsed) (that|the|on|every) \b"),
]


def candidate_names():
    d = OUTPUTS / "search" / RUN / "harnesses"
    if not RUN or not d.is_dir():
        return []
    return sorted(f.stem for f in d.glob("*.py") if not f.stem.startswith("baseline_"))


def main():
    names = candidate_names()
    rules = []
    for why, pat in RULES:
        if pat is None:
            pat = r"\b(" + "|".join(re.escape(n) for n in names) + r")\b" if names else None
        if pat:
            rules.append((why, re.compile(pat, re.I)))
    files = []
    for r in mounted_sources():
        p = pathlib.Path(r)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files += [f for f in p.rglob("*") if f.is_file() and f.suffix in (".py", ".md")
                      and "__pycache__" not in f.parts]
    bad = collections.Counter()
    for f in sorted(set(files)):
        txt = f.read_text(errors="ignore")
        lines = txt.splitlines()
        for why, pat in rules:
            for m in pat.finditer(txt):
                n = txt[:m.start()].count("\n") + 1
                print(f"  FAIL  {f.relative_to(REPO) if REPO in f.parents else f}:{n}  {why}")
                print(f"        {' '.join(lines[n - 1].split())[:96]}")
                bad[why] += 1
    verdict = "PASS" if not bad else "FAIL"
    print(f"  {verdict}  {len(set(files))} mounted files carry no past results, conclusions or guidance")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
