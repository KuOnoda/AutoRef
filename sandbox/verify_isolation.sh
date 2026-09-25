#!/bin/bash
# What the proposer can and cannot reach, checked rather than asserted.
#
#   bash sandbox/verify_isolation.sh <run-name>
#
# Starts a container with exactly the mounts sandbox/claude gives the proposer for that run (create
# the run first: `python -m autoref.search --run-name <run-name> --init-only`), and tries each route
# to something it should not have. A route closed
# by the filesystem stays closed whatever the model decides to do.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd -P)
RUN=${1:?usage: sandbox/verify_isolation.sh <run-name>}
export AUTOREF_SEARCH_RUN=$RUN AUTOREF_SHELL=1
OUTPUTS=$(readlink -m "${AUTOREF_OUTPUTS:-$REPO/outputs}")
EXTERNAL=$(readlink -m "${AUTOREF_EXTERNAL:-$REPO/external}")
RUN_DIR=$OUTPUTS/search/$RUN
[ -d "$RUN_DIR/proposer_view" ] || { echo "no run at $RUN_DIR (create it first)"; exit 2; }
PY=".venv/bin/python"

# the container must start before any check means anything
if ! err=$(sandbox/claude "true" 2>&1); then
    echo "the container does not start: $err"; exit 2
fi

pass=0; fail=0
check () {          # check <description> <absent|present> <shell command>
    local what=$1 mode=$2 cmd=$3 out found=1
    out=$(sandbox/claude "$cmd" 2>&1)
    [ -z "$out" ] && found=0
    if { [ "$mode" = absent ] && [ "$found" = 0 ]; } || { [ "$mode" = present ] && [ "$found" = 1 ]; }; then
        printf '  PASS  %s\n' "$what"; pass=$((pass + 1))
    else
        printf '  FAIL  %s\n        -> %s\n' "$what" "$(echo "$out" | head -2 | tr '\n' ' ')"; fail=$((fail + 1))
    fi
}

# an item of each held-out split, as a file the proposer must not be able to open
VAL_ITEM=$(python3 -c "import json;print(json.load(open('data/splits/multibanana_val.json'))['ids'][0])")
TEST_ITEM=$(python3 -c "import json;print(json.load(open('data/splits/multibanana_test.json'))['ids'][0])")
TRAIN_ITEM=$(python3 -c "import json;print(json.load(open('data/splits/multibanana_train.json'))['ids'][0])")
prompt_file () { echo "$EXTERNAL/MultiBanana/${1%/*}/${1#*/}_prompt.txt"; }

echo "== things the proposer must not reach =="
check "a validation item"                    absent "cat $(prompt_file "$VAL_ITEM") 2>/dev/null"
check "a test item"                          absent "cat $(prompt_file "$TEST_ITEM") 2>/dev/null"
check "the validation and test split files"  absent "ls data/splits/multibanana_val.json data/splits/multibanana_test.json 2>/dev/null"
check "this run's validation and test scores" absent "ls $RUN_DIR/private 2>/dev/null"
check "other search runs"                    absent "ls $OUTPUTS/search | grep -vx $RUN"
check "AutoRef-Harness (harnesses/)"          absent "ls harnesses 2>/dev/null"
check "the README and documentation"         absent "cat README.md 2>/dev/null; ls docs 2>/dev/null"
check "the search loop and its seeds"        absent "ls autoref/search 2>/dev/null"
check "git history"                          absent "ls .git 2>/dev/null"
check "the shared model-call cache"          absent "ls $OUTPUTS/cache 2>/dev/null"
check "the CLI's home beyond credentials"    absent "ls -A \$HOME/.claude | grep -v '^.credentials.json$'"
check "earlier session transcripts"          absent "ls \$HOME/.claude/projects 2>/dev/null"
check "python can read the private scores"   absent \
      "$PY -c \"print(open('$RUN_DIR/private/val_summary.jsonl').read())\" 2>/dev/null"

echo
echo "== things the proposer needs =="
check "the training items"                   present "cat $(prompt_file "$TRAIN_ITEM")"
check "the training split file"              present "ls data/splits/multibanana_train.json"
check "the seed harnesses"                   present "ls $RUN_DIR/harnesses/baseline_plain.py $RUN_DIR/harnesses/baseline_gems.py"
check "the harness interface"                present "$PY -c 'from autoref.harness import Harness; print(Harness.__name__)'"
check "the benchmark runner"                 present "$PY -c 'import autoref.multibanana as m; print(m.TASK_NAME)'"
check "somewhere to write candidates"        present "touch $RUN_DIR/harnesses/.w && rm $RUN_DIR/harnesses/.w && echo ok"
check "this run's logs"                      present "ls -d $RUN_DIR/logs"
check "the judge's prompt"                   present "ls $EXTERNAL/multibanana/judge.py"
check "a GPU"                                present "$PY -c 'import torch; print(torch.cuda.is_available() or \"\")'"

echo
echo "== what the mounted files say =="
python3 sandbox/inspect_contents.py && pass=$((pass + 1)) || fail=$((fail + 1))

echo
echo "  $pass passed, $fail failed"
[ "$fail" = 0 ]
