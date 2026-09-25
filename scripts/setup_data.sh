#!/bin/bash
# Fetch MultiBanana, the benchmark's judge prompt, and GEMS (the search's seed harness) into
# external/ (or $AUTOREF_EXTERNAL), at the revisions the paper used. Nothing fetched here is
# redistributed by this repository; see THIRD_PARTY_NOTICES.md.
#
#   bash scripts/setup_data.sh
#
# MultiBanana's 3-, 4- and 5-reference pools are about 3 GB. Set HF_TOKEN to avoid Hugging Face's
# rate limit on anonymous downloads.
set -euo pipefail
cd "$(dirname "$0")/.."
EXT=${AUTOREF_EXTERNAL:-external}
mkdir -p "$EXT"

# a repository at one commit, fetched shallowly
clone_at () {
    local repo=$1 dir=$2 rev=$3
    if [ "$(git -C "$dir" rev-parse HEAD 2>/dev/null || true)" = "$rev" ]; then
        echo "  $repo @ ${rev:0:7} (present)"; return
    fi
    mkdir -p "$dir"
    git -C "$dir" init -q
    git -C "$dir" remote remove origin 2>/dev/null || true
    git -C "$dir" remote add origin "https://github.com/$repo.git"
    git -C "$dir" fetch -q --depth 1 origin "$rev"
    git -C "$dir" checkout -q FETCH_HEAD
    echo "  $repo @ ${rev:0:7}"
}

echo "== the judge's prompt and GEMS"
clone_at matsuolab/multibanana "$EXT/multibanana" 04b7c454be2967b422b06080f5250ab2a1ec63ee
clone_at lcqysl/GEMS           "$EXT/GEMS"        fec33ff64b65efa55787616d504468c801776579

echo "== MultiBanana"
python - "$EXT" <<'PY'
import sys, time
from huggingface_hub import snapshot_download
from autoref.models import DATASET_REVISION
repo, rev = DATASET_REVISION
for attempt in range(10):
    try:
        snapshot_download(repo, repo_type="dataset", revision=rev,
                          allow_patterns=["3_*/*", "4_*/*", "5_*/*"],
                          local_dir=f"{sys.argv[1]}/MultiBanana", max_workers=4)
        break
    except Exception as e:                       # rate limits: wait and resume
        print(f"  retrying after {type(e).__name__}: {str(e)[:120]}", flush=True)
        time.sleep(60)
else:
    sys.exit("MultiBanana download failed")
print("  done")
PY
