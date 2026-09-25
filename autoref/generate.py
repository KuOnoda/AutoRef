"""Run AutoRef-Harness on your own reference images.

    python -m autoref.generate --refs dog.png hat.png beach.png \
        --prompt "The dog from image 1 wears the hat from image 2 on the beach from image 3." \
        --out out.png

The instruction refers to the references as "image 1", "image 2", ... in the order given. Every
draft is kept in <out>_rounds/, and the calls the harness made are written to <out>.trace.json.
"""
import argparse
import json
import os

from autoref.generators import GENERATORS, get_generator, release
from autoref.harness import Context, load_harness


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--refs", nargs="+", required=True, help="reference images, image 1 first")
    ap.add_argument("--prompt", required=True, help="the instruction, naming references as 'image N'")
    ap.add_argument("--out", default="out.png")
    ap.add_argument("--harness", default="autoref_harness", help=argparse.SUPPRESS)
    ap.add_argument("--generator", default="flux-klein-4b", choices=sorted(GENERATORS))
    ap.add_argument("--mllm", default=None, help="the reasoning model (default gpt-5.5-2026-04-23)")
    ap.add_argument("--cache", default="on", choices=("on", "write", "off"))
    args = ap.parse_args()

    harness = load_harness(args.harness)
    gen = get_generator(args.generator)
    stem = os.path.splitext(args.out)[0]
    ctx = Context(generator=gen, save_dir=stem + "_rounds", cache=args.cache,
                  **({"mllm": args.mllm} if args.mllm else {}))
    task = {"id": os.path.basename(stem), "prompt": args.prompt, "refs": list(args.refs),
            "n_refs": len(args.refs)}
    img = harness.run(task, ctx)
    release(gen)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(img)
    with open(stem + ".trace.json", "w") as f:
        json.dump({"harness": harness.name, "generator": args.generator, "task": task,
                   "calls": ctx.calls, "record": ctx.record}, f, indent=2, ensure_ascii=False)
    print(f"{args.out}  ({ctx.calls['generate']} images drawn, {ctx.calls['think']} reasoning calls; "
          f"drafts in {stem}_rounds/)")


if __name__ == "__main__":
    main()
