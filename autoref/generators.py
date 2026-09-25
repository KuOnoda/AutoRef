"""The image generators: FLUX.2 [klein] 4B and 9B, and Qwen-Image-Edit-2511, run locally.

    from autoref.generators import get_generator
    gen = get_generator("flux-klein-4b")
    png = gen.generate_multi([open("a.png", "rb"), open("b.png", "rb")], "the dog from image 1 ...")

Each generator loads onto the first visible GPU on first use, pinned to the revision in
`autoref.models.REVISIONS`, and its `name` is part of the cache key of every image it draws. No seed
is fixed. For GPUs too small for Qwen-Image-Edit-2511 (about 58 GB in bf16):

    AUTOREF_CPU_OFFLOAD=1    keep each component on the CPU until it runs (slower; one 48 GB card)
    AUTOREF_SHARD_GPUS=n     split the transformer across n-1 GPUs, text encoder and VAE on the last
"""
import io
import os
import threading

from autoref.models import revision

SIZE = (1024, 1024)


def _pil(b):
    from PIL import Image
    raw = b.getvalue() if hasattr(b, "getvalue") else b
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _png(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class FluxKleinBackend:
    """FLUX.2 [klein] (Black Forest Labs) through diffusers. The distilled checkpoints sample in four
    steps and ignore the guidance scale; references are passed as they are (the pipeline crops each
    to a multiple of 16 and caps it at one megapixel). Every call holds one lock, so a runner's
    threads can share one pipeline."""

    def __init__(self, repo, name, steps=4):
        self.repo, self.name, self.steps = repo, name, steps
        self.width, self.height = SIZE
        self.lock = threading.Lock()
        self.pipe = None

    def _load(self):
        import torch
        from diffusers import Flux2KleinPipeline
        self.pipe = Flux2KleinPipeline.from_pretrained(self.repo, revision=revision(self.repo),
                                                       dtype=torch.bfloat16)
        if os.environ.get("AUTOREF_CPU_OFFLOAD") == "1":
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)

    def _run(self, images, prompt):
        with self.lock:
            if self.pipe is None:
                self._load()
            out = self.pipe(image=images or None, prompt=prompt, width=self.width, height=self.height,
                            guidance_scale=1.0, num_inference_steps=self.steps).images[0]
        return _png(out)

    def generate_multi(self, images, prompt, **_):
        """One image from the prompt and the references (named BytesIO objects)."""
        return self._run([_pil(b) for b in images], prompt)

    def generate_text_only(self, prompt):
        return self._run([], prompt)


class QwenImageEditBackend(FluxKleinBackend):
    """Qwen-Image-Edit-2511 (Alibaba) through diffusers' QwenImageEditPlusPipeline, which takes a
    list of references in one call. Not distilled: the model card's 40 steps and true CFG 4.0."""

    SPLIT_CLASS = "QwenImageTransformerBlock"

    def __init__(self, repo, name, steps=40, true_cfg=4.0):
        super().__init__(repo, name, steps)
        self.true_cfg = true_cfg

    def _load(self):
        import torch
        from diffusers import QwenImageEditPlusPipeline
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(self.repo, revision=revision(self.repo),
                                                              dtype=torch.bfloat16)
        self._place()
        self.pipe.set_progress_bar_config(disable=True)

    def _place(self):
        import torch
        n = int(os.environ.get("AUTOREF_SHARD_GPUS", "0"))
        if n < 3:
            if os.environ.get("AUTOREF_CPU_OFFLOAD") == "1":
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe.to("cuda")
            return
        from accelerate import dispatch_model, infer_auto_device_map
        from accelerate.utils import get_balanced_memory
        # the transformer is split evenly over the first n-1 cards; the text encoder and the VAE share
        # the last, because the pipeline creates its tensors on the text encoder's card and passes
        # them to the VAE as they are
        free = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        per = f"{max(8, int(free) - 8)}GiB"
        kw = dict(no_split_module_classes=[self.SPLIT_CLASS], dtype=torch.bfloat16)
        mem = get_balanced_memory(self.pipe.transformer, max_memory={i: per for i in range(n - 1)}, **kw)
        dmap = infer_auto_device_map(self.pipe.transformer, max_memory=mem, **kw)
        if any(d in ("cpu", "disk") for d in dmap.values()):
            raise RuntimeError(f"the transformer does not fit on {n - 1} cards at {per}")
        self.pipe.transformer = dispatch_model(self.pipe.transformer, device_map=dmap)
        self.pipe.text_encoder.to(f"cuda:{n - 1}")
        self.pipe.vae.to(f"cuda:{n - 1}")

    def _run(self, images, prompt):
        with self.lock:
            if self.pipe is None:
                self._load()
            out = self.pipe(image=images or None, prompt=prompt, width=self.width, height=self.height,
                            negative_prompt=" ", true_cfg_scale=self.true_cfg,
                            num_inference_steps=self.steps).images[0]
        return _png(out)


GENERATORS = {
    "flux-klein-4b": lambda: FluxKleinBackend("black-forest-labs/FLUX.2-klein-4B", "flux-klein-4b"),
    "flux-klein-9b": lambda: FluxKleinBackend("black-forest-labs/FLUX.2-klein-9B", "flux-klein-9b"),
    "qwen-image-edit-2511": lambda: QwenImageEditBackend("Qwen/Qwen-Image-Edit-2511", "qwen-image-edit-2511"),
}


def get_generator(name):
    if name not in GENERATORS:
        raise KeyError(f"unknown generator {name!r}; choose from {sorted(GENERATORS)}")
    return GENERATORS[name]()


def release(generator):
    """Drop a loaded pipeline and hand its GPU memory back (the judge runs next)."""
    if getattr(generator, "pipe", None) is None:
        return
    generator.pipe = None
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
