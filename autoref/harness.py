"""The harness interface.

A harness is the Python around two frozen models: an image generator that draws from a prompt and
the images it is shown, and a reasoning model that reads text and images. It turns one task -- an
instruction plus reference images -- into one output image.

    from autoref.harness import Harness

    class MyHarness(Harness):
        name = "my_harness"
        def run(self, task, ctx):
            plan = ctx.think("Describe the subject in <image>", images=task["refs"][:1])
            return ctx.generate(plan, images=task["refs"])

`task` is {"id": str, "prompt": str, "refs": [image paths], "n_refs": int}; `run` returns the output
image as bytes. Calls are counted, not limited (`ctx.calls`).

Every call is cached on disk by its arguments. `cache="on"` reads and writes; `cache="write"` only
writes, so every harness pays for its own calls; `cache="off"` does neither.
"""
import base64
import hashlib
import importlib.util
import io
import json
import os
import re
import sys
import threading
import time

from autoref.models import REASONING_MODEL
from autoref.paths import CACHE, GEMS, HARNESSES

CACHE_DIR = os.path.join(CACHE, "calls")
_RETRYABLE = ("429", "rate limit", "timed out", "timeout", "overloaded", "502", "503")


class Harness:
    """Base class. Subclasses set `name` (equal to the file name) and implement `run`."""
    name = "base"
    description = ""

    def run(self, task, ctx):
        raise NotImplementedError


def load_harness(name="autoref_harness", harness_dir=None):
    """-> an instance of the harness in `harnesses/<name>.py`, in `<harness_dir>/<name>.py`, or in the
    .py file `name`. A harness may import other files of its own directory by file name.

    Bytecode is not written: a .pyc outlives the .py it was compiled from.
    """
    sys.dont_write_bytecode = True
    path = name if name.endswith(".py") else os.path.join(harness_dir or HARNESSES, name + ".py")
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no harness at {path}")
    folder, stem = os.path.dirname(path), os.path.basename(path)[:-3]
    if folder not in sys.path:
        sys.path.insert(0, folder)
    spec = importlib.util.spec_from_file_location(f"harness_{stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    classes = [obj for obj in vars(mod).values()
               if isinstance(obj, type) and issubclass(obj, Harness) and obj is not Harness]
    classes.sort(key=lambda c: (c.__module__ != mod.__name__, getattr(c, "name", "") != stem))
    if not classes:
        raise RuntimeError(f"{path} defines no Harness subclass")
    return classes[0]()


def load_skills(skills_dir=None):
    """-> {skill_id: {"id", "description", "instructions"}} from GEMS's `<dir>/<id>/SKILL.md` files
    (fetched by scripts/setup_data.sh; used by the search's GEMS seed harness)."""
    skills_dir = skills_dir or os.path.join(GEMS, "agent", "skills")
    out = {}
    if not os.path.isdir(skills_dir):
        return out
    for sid in sorted(os.listdir(skills_dir)):
        p = os.path.join(skills_dir, sid, "SKILL.md")
        if not os.path.isfile(p):
            continue
        text = open(p, encoding="utf-8").read()
        desc = re.search(r"## Description\n(.*?)\n##", text, re.S)
        instr = re.search(r"## Instructions\n(.*)", text, re.S)
        out[sid] = {"id": sid,
                    "description": desc.group(1).strip() if desc else "",
                    "instructions": instr.group(1).strip() if instr else ""}
    return out


def _key(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
        h.update(b"|")
    return h.hexdigest()


def _bytes(x):
    """Image bytes from bytes, a path, or a file-like object."""
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    if hasattr(x, "getvalue"):
        return x.getvalue()
    with open(x, "rb") as f:
        return f.read()


class Context:
    """The frozen models, and a record of every call made through them.

    generator:  an image generator from `autoref.generators`
    mllm:       the reasoning model, an OpenAI model id (OPENAI_API_KEY)
    save_dir:   every generated image is also written here as round_<n>.png
    """

    def __init__(self, *, generator, mllm=REASONING_MODEL, task_id="", save_dir="rounds",
                 cache="on", max_tokens=16384):
        self.backend = generator
        self.mllm = mllm
        self.task_id = task_id
        self.save_dir = save_dir
        self.max_tokens = max_tokens
        self.cache = {True: "on", False: "off"}.get(cache, cache)
        self.calls = {"think": 0, "generate": 0}
        self.record = []
        self._lock = threading.Lock()
        self._client = None
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

    def _cached(self, kind, key, produce, binary=False):
        path = os.path.join(CACHE_DIR, f"{kind}_{key}" + (".bin" if binary else ".json"))
        if self.cache == "on" and os.path.isfile(path):
            with open(path, "rb" if binary else "r") as f:
                return f.read() if binary else json.load(f)
        val = produce()
        if self.cache in ("on", "write"):
            with open(path, "wb" if binary else "w") as f:
                f.write(val) if binary else json.dump(val, f)
        return val

    def think(self, prompt, images=()):
        """One turn of the reasoning model. `<image>` placeholders take `images` in order."""
        blobs = [_bytes(x) for x in images]
        with self._lock:
            self.calls["think"] += 1
        segments = prompt.split("<image>")
        content = []
        for i, seg in enumerate(segments):
            if seg:
                content.append({"type": "text", "text": seg})
            if i < len(blobs):
                b64 = base64.b64encode(blobs[i]).decode()
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"}})

        def produce():
            if self._client is None:
                from openai import OpenAI
                self._client = OpenAI()
            last = None
            for attempt in range(6):
                try:
                    try:
                        r = self._client.chat.completions.create(
                            model=self.mllm, messages=[{"role": "user", "content": content}],
                            max_completion_tokens=self.max_tokens)
                    except TypeError:
                        r = self._client.chat.completions.create(
                            model=self.mllm, messages=[{"role": "user", "content": content}],
                            max_tokens=self.max_tokens)
                    return {"text": (r.choices[0].message.content or "").strip()}
                except Exception as e:          # noqa: BLE001 - re-raised after the retries
                    last = e
                    if not any(t in str(e).lower() for t in _RETRYABLE):
                        raise
                    time.sleep(5 * (attempt + 1))
            raise last

        out = self._cached("think", _key(self.mllm, prompt, *blobs), produce)["text"]
        self.record.append({"tool": "think", "prompt": prompt[:1200], "reply": out[:2000],
                            "n_images": len(blobs)})
        return out

    def generate(self, prompt, images=(), sample=None):
        """One image. `sample` separates independent draws of the same prompt in the cache."""
        blobs = [_bytes(x) for x in images]
        with self._lock:
            self.calls["generate"] += 1
            n = self.calls["generate"]

        def produce():
            if blobs:
                files = []
                for i, b in enumerate(blobs):
                    bio = io.BytesIO(b)
                    bio.name = f"ref_{i + 1}.png"
                    files.append(bio)
                return self.backend.generate_multi(files, prompt)
            return self.backend.generate_text_only(prompt)

        key_parts = [self.backend.name, prompt, *blobs]
        if sample is not None:
            key_parts += ["sample", sample]
        img = self._cached("gen", _key(*key_parts), produce, binary=True)
        path = os.path.join(self.save_dir, f"round_{n}.png")
        with open(path, "wb") as f:
            f.write(img)
        rec = {"tool": "generate", "prompt": prompt, "n_images": len(blobs),
               "image": os.path.abspath(path)}
        if sample is not None:
            rec["sample"] = sample
        self.record.append(rec)
        return img
