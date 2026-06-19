# Emote cutout lessons (pale / white-haired anime characters)

Hard-won notes from trying to produce clean transparent emotes from AI-generated
images of a pale, **white-haired** anime character. Read this before adding more
cutout post-processing — several "obvious" ideas were tried and failed.

## TL;DR / current policy

- **Background choice dominates everything.** For a pale/white-haired subject the
  matte is only as good as the contrast the backdrop gives it.
- **White / very-light backgrounds are the worst case**, not the best: white hair
  on white has no edge for a learned matte to find, and no boundary for a flood
  fill to stop at.
- We currently **generate on a flat light-neutral backdrop and use the image
  as-is** (no auto-cutout). Cutout is optional/manual and uses the **plain model
  matte only** — no aggressive backdrop suppression. Pixel-perfect transparency
  for white hair is not a solved problem here; "good enough when contrast allows"
  is the accepted bar.

## What was tried and why it failed

1. **Fixed-color / chroma key against the backdrop color.** Fails because an
   AI-generated "flat gray" backdrop is *not* one value — it is shaded darker in
   the gaps between hair layers (e.g. ~#505050 vs a #808080 border). A key
   centered on the border color misses the darker trapped gray, and partial
   distance-based removal leaves a semi-transparent **dirty mottle**.

2. **Neutral + smooth region-growing from the border** (color-value agnostic;
   grow through low-chroma, low-texture pixels within an adaptive luminance
   window). This is the most principled approach and removed most trapped gray,
   but the luminance window needed to catch shaded gaps also reaches **pale skin
   and light clothing**, so it can leak in from the boundary and **erase parts of
   the subject**. An "enclosed pocket" pass made it worse — it removed smooth
   neutral *features* (sunglasses lenses, etc.). Aggressive cleanup that fixes one
   image damages another.

3. **Flood-fill the light background, stopping at the character outline.** Clean
   in theory (high contrast, hard boundary) but depends on the generator drawing a
   **perfectly continuous dark outline** around all hair. It cannot. A single gap
   lets the fill pour through the (light) white hair and **eat the whole head** —
   worse than leaving gray. Validated on a real near-white-bg image: catastrophic.

4. **A stronger model (`toonout` BiRefNet anime fine-tune).** Only marginally
   better than fast `isnet-anime` on the gray case. The defect is edge definition
   and enclosed-gap semantics, not raw model quality, so a model swap is not the
   fix. (`toonout` is still the better `--quality` option; just not a silver
   bullet.)

## Methodological lessons

- **Composite-over-background tests are NOT representative.** Compositing an
  already-clean cutout onto a new backdrop gives the matte an artificial *hard*
  edge, so even white-on-white looks "fine." The real difficulty is the AI
  *drawing* pale hair softly into the backdrop. Background-color choices can only
  be validated on **real generations**, not composites.
- **Don't stack patches.** Each heuristic added to rescue the previous one widened
  the failure surface. Prefer fixing the *input* (generation backdrop / contrast)
  over post-processing a hard input.
- **Validate before shipping.** Several confident claims here were wrong until
  tested on real images (magenta/green composite inspection at full res).

## Still-available knobs (dormant unless explicitly enabled)

The experimental cleanup lives in `background.py` (`suppress_background_color`,
region-growing) and `masks.contract_alpha` (`--edge-contract`), plus
`--flatten-bg`. They are **off by default in the emote pipeline** (`--no-bg-suppress`).
Revisit only with a contrast-friendly backdrop and real-generation validation.

## If revisiting for clean transparency

The unsolved direction worth real-generation A/B testing: a backdrop that
genuinely contrasts with white hair — a **medium-tone or lightly-tinted** color
(chroma gives the matte a handle that neutral light gray does not). Pale *neutral*
backgrounds are the worst of both worlds (no luminance and no chroma contrast).
