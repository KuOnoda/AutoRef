"""One generation from the prompt and the references, nothing else."""
from autoref.harness import Harness


class BaselinePlain(Harness):
    name = "baseline_plain"
    description = "one image from the prompt, conditioned on the references"

    def run(self, task, ctx):
        return ctx.generate(task["prompt"], images=task["refs"])
