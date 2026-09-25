"""The published GEMS loop (He et al., 2026), multi-reference variant, as a harness.

Stages: route the prompt through a skill, decompose it into yes/no questions, then up to four
rounds of generate -> verify each question against the image -> summarise the round as an
experience -> rewrite the prompt from the whole history. The verifier is shown the references
beside the generated image. The image with the most passed questions is returned, or the first
that passes all of them.

This is both the GEMS baseline of the paper and one of the two harnesses the search starts from.
GEMS (github.com/lcqysl/GEMS) is published without a license, so its prompts are not copied into
this repository: the five prompt templates and the skills are read at run time from the GEMS
checkout that scripts/setup_data.sh fetches at a pinned commit, and each template is checked
against the hash of the text the paper's runs used.
"""
import ast
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from autoref.harness import Harness, load_skills
from autoref.paths import GEMS

# sha256 of each template in lcqysl/GEMS@fec33ff64b65 agent/GEMS.py, as used in the paper
GEMS_PROMPTS = {
    "SUMMARIZE_EXPERIENCE_TEMPLATE": "6db67901a65ab8b8f411fa0afaea2dd2113fc96a877639446cb43b7963ee661d",
    "DECOMPOSE_PROMPT": "dbf8877ab2dda7feac0a1c2ca8ad69f2fa0f7f7a9682c2162ab1ff26441b91d8",
    "VERIFY_PROMPT_PREFIX": "f02f36c03acd857823a4d14af4bfad98cc18060eaa9d1e1b9145d8c24fbf7d04",
    "REFINE_PROMPT_TEMPLATE": "2bb90ed0e6deb449496bea56e125c4f1822dc0cddca23a2f563001e655b77193",
    "PLANNER_DECISION_PROMPT": "4f2acbdf3e86c1e14a25fbb8fa9707038313cff0bf51fb73aef71e0884acb2cd",
}
_PROMPTS = {}


def gems_prompts():
    """-> {template name: text}, read from the fetched GEMS checkout and verified."""
    if not _PROMPTS:
        path = os.path.join(GEMS, "agent", "GEMS.py")
        if not os.path.isfile(path):
            raise RuntimeError(f"{path} not found; run scripts/setup_data.sh to fetch GEMS")
        found = {}
        for node in ast.parse(open(path, encoding="utf-8").read()).body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id in GEMS_PROMPTS):
                found[node.targets[0].id] = ast.literal_eval(node.value)
        for name, digest in GEMS_PROMPTS.items():
            if name not in found or hashlib.sha256(found[name].encode()).hexdigest() != digest:
                raise RuntimeError(f"{name} in {path} is not the text the paper used; "
                                   f"check out lcqysl/GEMS at fec33ff64b65")
        _PROMPTS.update(found)
    return _PROMPTS


# GEMS reads the reasoning model's separate reasoning channel; a chat-completions model has none,
# so the thought is asked for in the answer instead
THOUGHT_FORMAT_SUFFIX = (
    "\n\n### Response Format:\n"
    "Respond in exactly two sections:\n"
    "THOUGHT: <your step-by-step reasoning>\n"
    "ANSWER: <the final answer only, following the task's output requirements>"
)


class BaselineGems(Harness):
    name = "baseline_gems"
    description = "the published GEMS loop: route, decompose, then generate/verify/refine for up to 4 rounds"
    max_iterations = 4

    # ---- helpers ------------------------------------------------------------
    def think_with_thought(self, ctx, prompt, images=()):
        raw = ctx.think(prompt + THOUGHT_FORMAT_SUFFIX, images=images)
        if "ANSWER:" in raw:
            thought, answer = raw.split("ANSWER:", 1)
            return answer.strip(), thought.replace("THOUGHT:", "", 1).strip()
        return raw.strip(), ""

    def plan(self, ctx, original_prompt):
        skills = load_skills()
        manifest = "".join(f"- SKILL_ID: {sid}\n  DESCRIPTION: {s['description']}\n"
                           for sid, s in skills.items())
        skill_id = ctx.think(gems_prompts()["PLANNER_DECISION_PROMPT"].format(
            manifest=manifest, user_prompt=original_prompt)).strip()
        if skill_id in skills:
            refine_task = (
                f"Based on the following skill instructions, enhance the user's prompt.\n"
                f"### Skill Instructions:\n{skills[skill_id]['instructions']}\n\n"
                f"### Original Prompt: {original_prompt}\n\n"
                f"Return ONLY the final enhanced prompt text."
            )
            return ctx.think(refine_task).strip()
        return original_prompt

    def decompose(self, ctx, prompt):
        response = ctx.think(f"{gems_prompts()['DECOMPOSE_PROMPT']}\n\nUser Prompt: {prompt}").strip()
        try:
            m = re.search(r"\[.*\]", response, re.S)
            questions = json.loads(m.group() if m else response)
            return [q for q in questions if isinstance(q, str)] if isinstance(questions, list) else []
        except Exception:                       # noqa: BLE001 - fall back to line splitting
            return [line.strip() for line in response.split("\n") if "?" in line]

    def verify(self, ctx, image, refs, questions):
        head = "".join(f"Reference image {i + 1}: <image>\n" for i in range(len(refs)))
        prefix = gems_prompts()["VERIFY_PROMPT_PREFIX"]

        def ask(q):
            try:
                a = ctx.think(f"{head}Generated image: <image>\n{prefix} {q}",
                              images=list(refs) + [image]).lower().strip()
                return {"question": q, "answer": a, "passed": a.startswith("yes")}
            except Exception as e:              # noqa: BLE001 - a failed check is a failed check
                return {"question": q, "answer": f"Error: {e}", "passed": False}

        with ThreadPoolExecutor(max_workers=min(len(questions), 8)) as ex:
            return list(ex.map(ask, questions))

    # ---- the loop -------------------------------------------------------------
    def run(self, task, ctx):
        prompts = gems_prompts()
        original_prompt = task["prompt"]
        refs = task["refs"]
        current_prompt = self.plan(ctx, original_prompt)
        current_thought = ("Initial submission enhanced by specialized skill instructions."
                           if current_prompt != original_prompt
                           else "Initial submission based on original prompt.")
        questions = self.decompose(ctx, original_prompt)
        if not questions:
            return ctx.generate(original_prompt, images=refs)

        history, best, best_passed = [], None, -1
        for i in range(1, self.max_iterations + 1):
            image = ctx.generate(current_prompt, images=refs)
            verifications = self.verify(ctx, image, refs, questions)
            failed = [v["question"] for v in verifications if not v["passed"]]
            passed = [v["question"] for v in verifications if v["passed"]]
            if len(passed) > best_passed:
                best_passed, best = len(passed), image
            if not failed:
                return image
            if i == self.max_iterations:
                break

            previous = "\n".join(f"Round {h['iteration']}: {h['experience']}" for h in history) \
                or "None (First round)"
            experience = ctx.think(prompts["SUMMARIZE_EXPERIENCE_TEMPLATE"].format(
                current_prompt=current_prompt,
                passed=", ".join(passed) or "None", failed=", ".join(failed) or "None",
                current_thought=current_thought, previous_experiences=previous),
                images=[image]).strip()
            history.append({"iteration": i, "prompt": current_prompt, "experience": experience,
                            "failed": failed, "passed": passed, "image": image})

            log, images = "", []
            for h in history:
                log += (f"Attempt {h['iteration']}:\n- Experience: {h['experience']}\n"
                        f"- Prompt: {h['prompt']}\n- Image Result: <image>\n"
                        f"- Failed Points: {', '.join(h['failed']) or 'None'}\n\n")
                images.append(h["image"])
            current_prompt, current_thought = self.think_with_thought(
                ctx, prompts["REFINE_PROMPT_TEMPLATE"].format(original_prompt=original_prompt,
                                                              history_log=log), images=images)
            current_prompt = current_prompt.strip()
        return best
