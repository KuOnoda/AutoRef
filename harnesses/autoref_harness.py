"""AutoRef-Harness: the harness AutoRef discovered for multi-reference image generation.

Three images per task, built in four steps (Section 5 of the paper):

1. Reference-grounded prompting (`rewrite` with SCENE_PROMPT). The reasoning model rewrites the
   instruction into a prompt that fixes one setting, light and medium, then names every subject
   "from image N" with a few identity words, its place and what it stands on; attributes taken from
   a reference (a garment, a pose) are attached to the subject that wears or holds them.
2. Structurally diverse drafts (`canvas_framing`). Draft A is drawn from that scene-first prompt.
   For draft B, the reference the instruction names as the background (or the style) is passed to
   the generator first, and the prompt declares it the picture to keep (or the style to paint in)
   and places the other subjects into it. Without such a reference, draft B is a second sample.
3. Failure-aware selection (`hard_failures`, `compare`, `duel`). A strict check counts missing and
   extra or duplicated subjects, and a wrong background twice; the draft with fewer failures wins.
   On a tie, the reasoning model lists the concrete differences a strict rater would score, in both
   presentation orders, and the challenger wins only if both orders name it.
4. Complaint-directed revision (`complaints`, `revise`). The reasoning model lists up to five
   concrete complaints about the winner, each naming its reference, and revises the winner's prompt
   to answer them; draft C is drawn from the revision and must beat the winner by the same rule.

    from autoref.generators import get_generator
    from autoref.harness import Context, load_harness

    harness = load_harness("autoref_harness")
    ctx = Context(generator=get_generator("flux-klein-4b"), save_dir="rounds")
    png = harness.run({"prompt": "...", "refs": ["img1.png", "img2.png"]}, ctx)

The code below is exactly the code the proposer wrote during the search (where it was named
`canvas_pair_pick`); only this description, the import and the class name were changed.
"""
import json
import re

from autoref.harness import Harness

SCENE_PROMPT = (
    "Reference images, numbered in order:\n{refs}Instruction: {prompt}\n\n"
    "Write a prompt for an image generation model that sees the same numbered images, in this fixed order:\n"
    "1. SETTING: one sentence naming the single place, time of day, light direction and medium (photograph, "
    "painting, illustration) of the whole picture. If the instruction names a reference for the background "
    "or the style, describe that reference's scene or style in concrete words; otherwise choose one "
    "plausible setting. No other reference's background is used.\n"
    "2. CAST: one sentence per subject, in the instruction's order: 'the <subject> from image N (<3-8 "
    "identity words: species, hair, colours, clothing>)', its place in the picture, and what it stands "
    "on, sits on or holds. If the instruction replaces or swaps a subject, the CAST names only the "
    "replacement, in the replaced one's place and role; the replaced subject is not in the picture.\n"
    "3. TRANSFERS: a garment, hairstyle, held object, texture or pose taken from a reference is described "
    "as an attribute of the cast member who wears or holds it, never as a person or scene of its own.\n"
    "4. Close with the number of characters in total and: one light with matching contact shadows, every "
    "subject on the ground, no text.\n"
    "Keep every 'image N' mention; never use the words collage, composite or montage. Under 120 words, plain "
    "prose. Return only the prompt."
)

ROLE_PROMPT = (
    "Reference images, numbered in order:\n{refs}Instruction: {prompt}\n\n"
    "Which reference does the instruction name as the BACKGROUND, SETTING or LOCATION of the whole picture, and "
    "which as the STYLE of the whole picture? A reference that supplies a subject, object, garment or pose is "
    "neither. Reply with ONLY JSON: {{\"background\": <image number or 0>, \"style\": <image number or 0>}}"
)

SCENE_CANVAS_PROMPT = (
    "Reference images, numbered in order (image 1 is the scene the picture keeps):\n{refs}Instruction: {prompt}\n\n"
    "Write a prompt for an image generation model that sees the same numbered images, in this order:\n"
    "1. Open with: 'Image 1 is the picture: keep its place, viewpoint, medium, brushwork or grain, light and "
    "colours exactly.' Then name image 1's light direction and medium in a few words.\n"
    "2. Then one sentence per subject the instruction places into it, in the instruction's order: 'the <subject> "
    "from image N (<3-8 identity words>)', where it stands, sits or rests in image 1's scene (name the surface), "
    "repainted in image 1's medium and lit by image 1's light, with a contact shadow. A subject the instruction "
    "replaces or removes is not in the picture.\n"
    "3. A garment, hairstyle, held object, texture or pose taken from a reference is an attribute of the cast "
    "member who wears or holds it, never a person or scene of its own.\n"
    "4. Close with the number of characters in total, and: nothing else added, no text.\n"
    "Keep every 'image N' mention. Under 120 words, plain prose. Return only the prompt."
)

STYLE_CANVAS_PROMPT = (
    "Reference images, numbered in order (image 1 sets the style of the picture):\n{refs}Instruction: {prompt}\n\n"
    "Write a prompt for an image generation model that sees the same numbered images, in this order:\n"
    "1. Open with: 'Image 1 sets the medium, palette, line and brushwork of the whole picture; no object, figure "
    "or place from it is copied.' Then name that medium and palette in a few words, and one setting for the "
    "picture (the instruction's if it gives one, else one plausible place, time of day and light direction).\n"
    "2. Then one sentence per subject the instruction places in the picture, in the instruction's order: 'the "
    "<subject> from image N (<3-8 identity words>)', where it stands, sits or rests (name the surface), drawn "
    "entirely in image 1's medium and lit by the setting's light, with a contact shadow. A subject the "
    "instruction replaces or removes is not in the picture.\n"
    "3. A garment, hairstyle, held object, texture or pose taken from a reference is an attribute of the cast "
    "member who wears or holds it, never a person or scene of its own.\n"
    "4. Close with the number of characters in total, and: nothing else added, no text.\n"
    "Keep every 'image N' mention. Under 120 words, plain prose. Return only the prompt."
)

CRITIQUE_PROMPT = (
    "Reference images, numbered in order:\n{refs}Instruction: {prompt}\nBest draft so far: <image>\n\n"
    "A strict rater will grade this draft against the references. Write the rater's complaints as a list of "
    "concrete, checkable statements, most severe first, at most 5, each naming the reference it concerns:\n"
    "1. a required subject or object that is absent, or replaced by a different one; a subject the instruction "
    "removes or replaces that is still present; an extra or duplicated person or animal; a setting that is not "
    "the named reference's.\n"
    "2. a subject whose face, hair, garment, colours, pattern or accessory differs from its reference -- say "
    "which and how.\n"
    "3. a subject lit, sharpened or rendered in a different medium from its surroundings, floating, without a "
    "contact shadow, or at the wrong scale.\n"
    "Only complaints the draft actually earns; an empty list is a valid answer. A subject's own reference "
    "background, scene, pose or composition is NOT required unless the instruction asks for it, so never "
    "complain that it is missing; a style reference supplies only its style, never its content.\n"
    "Reply with ONLY JSON: {{\"complaints\": [\"...\"]}}"
)

REVISE_PROMPT = (
    "Reference images, numbered in order:\n{refs}Instruction: {original}\n"
    "Current prompt for the image model: {current}\n\n"
    "A rater made these complaints about the draft drawn from it:\n{complaints}\n"
    "Revise the prompt for the same image model (it sees the same numbered images) so that the next draft "
    "answers each complaint and nothing else changes: a missing or replaced subject is named first with 'from "
    "image N', its identity words and its exact place; an extra or removed subject is stated as absent; a "
    "mismatched detail is spelled out as its reference shows it; a subject unlike its surroundings gets the "
    "scene's light, medium and a contact shadow named explicitly. Keep the SETTING sentence, the cast, every "
    "'image N' mention and the closing count and light line; add no place, object or figure the instruction "
    "does not ask for; never use the words collage, composite or montage. Under 130 words, plain prose. Return "
    "only the revised prompt."
)

CHECK_PROMPT = (
    "Reference images, numbered in order:\n{refs}Instruction: {prompt}\nGenerated image: <image>\n\n"
    "Check the generated image strictly for HARD failures only:\n"
    "- 'missing': a reference whose required content is not depicted at all, or is replaced by a clearly "
    "different subject/object (a different person or animal, or a different item). A changed pose, a "
    "changed style, a partially visible or small subject, or a stylised rendering is NOT missing. "
    "Style-only references are never missing.\n"
    "- 'extra': a person, animal or character that the instruction did not ask for, including a second "
    "copy of a required subject, and a subject the instruction says is replaced but is still present. "
    "Background passers-by that the instruction's setting implies are NOT extra.\n"
    "- 'wrong_background': the instruction names a reference image as the background or setting, and the "
    "generated image shows a different setting instead.\n"
    "Reply with ONLY a JSON object {{\"missing\": [list of 'image N'], \"extra\": [short descriptions], "
    "\"wrong_background\": true/false}}."
)

DIFF_PROMPT = (
    "Reference images, numbered in order:\n{refs}Instruction: {prompt}\n"
    "Candidate A: <image>\nCandidate B: <image>\n\n"
    "A strict rater will score each candidate on: instruction alignment (a required subject absent = 1; any "
    "composited look caps it at 6), reference consistency (fine details of every subject must match its "
    "reference), background-subject match (several pictures pasted together = 1; the slightest lighting, tone "
    "or style mismatch caps it at 4), physical realism (any impression of compositing caps it at 4; unclear "
    "ground contact caps it at 6) and visual quality.\n"
    "Do not score them separately. Instead, list the CONCRETE DIFFERENCES between A and B that would move "
    "those scores, checked against the references: which required subjects/objects each one shows or lacks, "
    "extra or duplicated people, which subject looks less like its reference and how, which one has subjects "
    "lit or rendered differently from its background, which one has floating or cut-out subjects, and which "
    "composition looks more like one natural picture. Where they are equal, say so.\n"
    "Then decide which candidate the rater would score higher in total, and how sure you are.\n"
    "Reply with ONLY JSON: {{\"differences\": [\"...\"], \"winner\": \"A\" or \"B\", "
    "\"margin\": <0-5, 0 = coin flip, 5 = certain>}}"
)


def _head(n):
    return "".join(f"Image {i + 1}: <image>\n" for i in range(n))


def _clean(text):
    t = (text or "").strip().strip('"').strip()
    for prefix in ("revised prompt:", "rewritten prompt:", "prompt:", "instruction:"):
        if t.lower().startswith(prefix):
            t = t[len(prefix):].strip()
    return t


def _json(text):
    try:
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group() if m else text)
    except Exception:                       # noqa: BLE001
        return {}


def rewrite(ctx, task, template):
    """-> the rewritten prompt from `template`, or the original one if the LLM gives nothing usable."""
    refs = task["refs"]
    try:
        out = _clean(ctx.think(template.format(refs=_head(len(refs)), prompt=task["prompt"]), images=refs))
    except Exception:                       # noqa: BLE001
        out = ""
    if len(out.split()) < 8 or len(out.split()) > 260:
        return task["prompt"]
    return out


def rewrite_text(ctx, instruction, refs, template):
    """-> the rewritten prompt from `template` for a (possibly renumbered) instruction and reference order."""
    try:
        out = _clean(ctx.think(template.format(refs=_head(len(refs)), prompt=instruction), images=refs))
    except Exception:                       # noqa: BLE001
        out = ""
    if len(out.split()) < 8 or len(out.split()) > 260:
        return instruction
    return out


def roles(ctx, task):
    """-> (background reference index, style reference index), 1-based, 0 when the instruction names none."""
    refs = task["refs"]
    try:
        d = _json(ctx.think(ROLE_PROMPT.format(refs=_head(len(refs)), prompt=task["prompt"]), images=refs))
    except Exception:                       # noqa: BLE001
        return 0, 0

    def idx(v):
        try:
            i = int(v)
        except (TypeError, ValueError):
            return 0
        return i if 1 <= i <= len(refs) else 0
    return idx(d.get("background")), idx(d.get("style"))


def renumber(instruction, order):
    """The instruction with every 'image N' renamed to N's position in `order` (1-based old indices)."""
    new_of = {old: i + 1 for i, old in enumerate(order)}
    return re.sub(r"(?i)\bimage (\d+)\b",
                  lambda m: f"image {new_of.get(int(m.group(1)), m.group(1))}", instruction)


def canvas_framing(ctx, task):
    """-> (prompt, reordered refs, renumbered instruction) with the anchoring reference first, or None."""
    bg, style = roles(ctx, task)
    anchor = bg or style
    if not anchor:
        return None
    order = [anchor] + [i for i in range(1, len(task["refs"]) + 1) if i != anchor]
    refs = [task["refs"][i - 1] for i in order]
    instruction = renumber(task["prompt"], order)
    template = SCENE_CANVAS_PROMPT if bg else STYLE_CANVAS_PROMPT
    prompt = rewrite_text(ctx, instruction, refs, template)
    if prompt == instruction:
        return None
    return prompt, refs, instruction


def hard_failures(ctx, task, image):
    """-> weight of the strict check: missing and extra count 1 each, a wrong setting 2."""
    refs = task["refs"]
    try:
        d = _json(ctx.think(CHECK_PROMPT.format(refs=_head(len(refs)), prompt=task["prompt"]),
                            images=list(refs) + [image]))
    except Exception:                       # noqa: BLE001
        return 0
    missing = [m for m in (d.get("missing") or []) if isinstance(m, str) and m.strip()]
    extra = [e for e in (d.get("extra") or []) if isinstance(e, str) and e.strip()]
    return len(missing) + len(extra) + (2 if d.get("wrong_background") is True else 0)


def compare(ctx, task, first, second):
    """-> ('A'|'B'|None, margin 0-5) from one difference-first pairwise call."""
    refs = task["refs"]
    try:
        d = _json(ctx.think(DIFF_PROMPT.format(refs=_head(len(refs)), prompt=task["prompt"]),
                            images=list(refs) + [first, second]))
    except Exception:                       # noqa: BLE001
        return None, 0.0
    w = d.get("winner")
    m = d.get("margin")
    return (w if w in ("A", "B") else None), (float(m) if isinstance(m, (int, float)) else 1.0)


def duel(ctx, task, incumbent, challenger):
    """One rung. Candidates are (image, weight). Fewer hard failures wins outright; equal counts go to the
    pairwise rater in both orders, and the challenger replaces the incumbent only when both orders name it."""
    if challenger[1] != incumbent[1]:
        return challenger if challenger[1] < incumbent[1] else incumbent
    w1, _ = compare(ctx, task, incumbent[0], challenger[0])       # 'B' -> challenger
    w2, _ = compare(ctx, task, challenger[0], incumbent[0])       # 'A' -> challenger
    # the rater leans to the second position, and when both orders give the same letter the stated margins
    # decide below chance (26 judged pairs), so only two orders that both name the challenger replace the incumbent
    return challenger if (w1 == "B" and w2 == "A") else incumbent


def complaints(ctx, task, image):
    """-> up to five concrete complaints about the draft, most severe first; [] when it earns none."""
    refs = task["refs"]
    try:
        d = _json(ctx.think(CRITIQUE_PROMPT.format(refs=_head(len(refs)), prompt=task["prompt"]),
                            images=list(refs) + [image]))
    except Exception:                       # noqa: BLE001
        return []
    out = [c.strip() for c in (d.get("complaints") or []) if isinstance(c, str) and c.strip()]
    return out[:5]


def revise(ctx, instruction, refs, current, notes):
    """-> the prompt revised to answer the complaints, or '' if the LLM gives nothing usable. `instruction` and
    `refs` are the numbering and order the current prompt was written for."""
    listing = "\n".join(f"- {c}" for c in notes)
    try:
        out = _clean(ctx.think(REVISE_PROMPT.format(refs=_head(len(refs)), original=instruction,
                                                    current=current, complaints=listing), images=refs))
    except Exception:                       # noqa: BLE001
        return ""
    if len(out.split()) < 8 or len(out.split()) > 280 or out == current:
        return ""
    return out


class AutoRefHarness(Harness):
    name = "autoref_harness"
    description = ("first pair = scene-first draft + canvas draft with the background/style reference passed first "
                   "as the picture to keep; failure-first pick; third draft revised to answer complaints about "
                   "the winner in its own numbering; final failure-first pick")

    def run(self, task, ctx):
        refs, instruction = list(task["refs"]), task["prompt"]
        scene = rewrite(ctx, task, SCENE_PROMPT)
        canvas = canvas_framing(ctx, task)
        if canvas is None:
            canvas = (scene + "\nOne seamless picture.", refs, instruction)        # no anchor: a second sample
        # candidates: (image, weight, prompt, refs, instruction)
        cands = []
        for prompt, order, instr in ((scene, refs, instruction), canvas):
            img = ctx.generate(prompt, images=order)
            cands.append((img, hard_failures(ctx, task, img), prompt, order, instr))
        best = duel(ctx, task, cands[0], cands[1])
        notes = complaints(ctx, task, best[0])
        revised = revise(ctx, best[4], best[3], best[2], notes) if notes else ""
        if not revised:
            revised = best[2] + "\nEverything in one continuous scene."           # a plain resample
        third = ctx.generate(revised, images=best[3])
        best = duel(ctx, task, best, (third, hard_failures(ctx, task, third), revised, best[3], best[4]))
        return best[0]
