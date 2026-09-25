"""Where things live. Everything derives from the repository root unless an environment variable
points elsewhere:

    AUTOREF_OUTPUTS   runs, search logs and the model-call cache   (default: <repo>/outputs)
    AUTOREF_EXTERNAL  the dataset and third-party checkouts         (default: <repo>/external)
    AUTOREF_CACHE     the model-call cache                          (default: <outputs>/cache)
"""
import os

# physical paths: the proposer's container mounts everything at its real path
ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

OUTPUTS = os.path.realpath(os.environ.get("AUTOREF_OUTPUTS", os.path.join(ROOT, "outputs")))
CACHE = os.path.realpath(os.environ.get("AUTOREF_CACHE", os.path.join(OUTPUTS, "cache")))
SEARCH_RUNS = os.path.join(OUTPUTS, "search")

# fetched by scripts/setup_data.sh
EXTERNAL = os.path.realpath(os.environ.get("AUTOREF_EXTERNAL", os.path.join(ROOT, "external")))
MULTIBANANA = os.path.join(EXTERNAL, "MultiBanana")                # kohsei/MultiBanana-Benchmark
MULTIBANANA_OFFICIAL = os.path.join(EXTERNAL, "multibanana")       # matsuolab/multibanana
GEMS = os.path.join(EXTERNAL, "GEMS")                              # lcqysl/GEMS

# committed
SPLITS = os.path.join(ROOT, "data", "splits")
HARNESSES = os.path.join(ROOT, "harnesses")
SKILLS = os.path.join(ROOT, ".claude", "skills")
