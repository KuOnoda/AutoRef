# Third-party notices

## Meta-Harness (MIT)

[stanford-iris-lab/meta-harness](https://github.com/stanford-iris-lab/meta-harness), commit `0cbc31e`.
`autoref/search/claude_wrapper.py` is `reference_examples/text_classification/claude_wrapper.py`,
copied unmodified apart from a notice at the top. The search loop (`autoref/search/loop.py`) and the
proposer's skill (`.claude/skills/autoref-proposer/SKILL.md`) are adapted from that example's
`meta_harness.py` and `.claude/skills/meta-harness/SKILL.md`.

```
MIT License

Copyright (c) 2026 Yoonho Lee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Fetched at run time, not redistributed

`scripts/setup_data.sh` downloads these into `external/`.

| What | Source (pinned) | License | Used for |
|---|---|---|---|
| MultiBanana | [kohsei/MultiBanana-Benchmark](https://huggingface.co/datasets/kohsei/MultiBanana-Benchmark) `6c682f0` | CC BY-NC 4.0 | the benchmark |
| MultiBanana judge prompt | [matsuolab/multibanana](https://github.com/matsuolab/multibanana) `04b7c45` | CC BY-NC 4.0 | `autoref/judge.py` reads `create_evaluation_prompts` from `judge.py` |
| GEMS | [lcqysl/GEMS](https://github.com/lcqysl/GEMS) `fec33ff` | none published | the search's seed harness (`autoref/search/seeds/baseline_gems.py`) reads GEMS's prompt templates and skills from the checkout and checks each template's hash; the loop's control flow is re-implemented |

The reference images shown in `assets/qualitative.jpg` and `assets/overview.png` are from
MultiBanana (CC BY-NC 4.0).

## Models

Weights are downloaded from Hugging Face and keep their own licenses: FLUX.2 [klein] 4B,
Qwen-Image-Edit-2511 and Qwen3-VL-8B-Instruct (Apache-2.0); FLUX.2 [klein] 9B (FLUX Non-Commercial
License, gated). GPT-5.5 and Claude are used under their providers' terms.
