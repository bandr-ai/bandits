# Bandr README: visual and editorial direction

Reviewed 2026-11-13. This pass updates the README and its artwork; it does not
change the CLI or conversion behavior.

The earlier reference set (uv, fzf, GitHub CLI, Anthropic skills) helped with
installation and information structure. It was insufficient research for the
visual identity the project previously had. The historic Bandits README used
an orange-accented terminal animation, light/dark variants, a pipeline figure,
a worked trace, and research plots. Its specific labeling/training story no
longer describes Mint, but its concrete visual storytelling is worth retaining.

## References studied

- [Reef](https://github.com/Human-Agent-Society/reef/blob/main/README.md): a centered identity, clear ecosystem position, use-case tables, and a staged explanation of the system. Borrow the orientation; avoid its long recipe/news/community surface for this smaller product.
- [Agent Orchestrator](https://github.com/OrchestratorInc/agent-orchestrator/blob/main/README.md): direct outcome-led opening, a large product image, and feature descriptions supported by screenshots. Borrow the pacing and visible proof; skip the extensive platform and integration inventory.
- [VHS](https://github.com/charmbracelet/vhs/blob/main/README.md): a recognizable identity and an immediate animation with linked source. Use a rebuildable visual and label whether it is a recording or an illustration.
- [Gum](https://github.com/charmbracelet/gum/blob/main/README.md): distinctive brand artwork, an early demo, and a concrete tutorial. Borrow its specificity; Mint should show actual work rather than decorative terminal chrome.
- [Glow](https://github.com/charmbracelet/glow/blob/master/README.md): the product's appearance comes before detailed setup. A visual tool benefits from letting readers see the result early.
- [Rich](https://github.com/Textualize/rich/blob/master/README.md): immediately demonstrates its output, then explains how to use it. For Mint, show task construction and real gate output.
- [bat](https://github.com/sharkdp/bat/blob/master/README.md): feature-sized explanations paired with output images. Avoid making the reader infer all the behavior from one abstract architecture diagram.
- [delta](https://github.com/dandavison/delta/blob/main/README.md): clear identity, an actionable setup, and concrete examples. Keep the install-to-command naming distinction brief.
- [zoxide](https://github.com/ajeetdsouza/zoxide/blob/main/README.md): a concise mental model, demo, and compact command examples; collapsible setup details prevent alternatives dominating the page.
- [Ollama](https://github.com/ollama/ollama/blob/main/README.md): simple identity and direct commands. Useful restraint; its long integration catalog does not belong in Mint's first-run path.
- [DSPy](https://github.com/stanfordnlp/dspy/blob/main/README.md): strong conceptual framing and a short path to documentation. Its compactness is useful, but it is not our main visual reference.
- [Harbor](https://github.com/harbor-framework/harbor/blob/main/README.md): installation and a runnable example establish the runtime context. Explain our relationship to it instead of reproducing its documentation.
- [Zed](https://github.com/zed-industries/zed/blob/main/README.md): a useful counterexample to “popular repo means visual README.” Its current README is restrained and documentation-oriented; not the direction requested here.
- [Flowise](https://github.com/FlowiseAI/Flowise/blob/main/README.md): an immediately visible workflow demo. Its current archival notice means it is a historical presentation reference, not a product recommendation.

## Applied direction

Centered title, a direct conversion promise, two useful badges, and four navigation
links. The opening uses large type and a live reveal of a Harbor package. Black
and white variants share a restrained blue accent and identical content.

Keep the primary install path and skill prompt short. Put installer alternatives,
full status definitions, and developer details into disclosures. Use an honest
use-case table, a four-step workflow, an inspectable package tree, and a captured
CLI excerpt from the bundled example. Do not invent benchmarks, integrations,
community channels, or publication badges.

The animation follows the real package-writing contract. Its source and fixture
are linked for inspection. The separate CLI excerpt comes from an actual local run. The example is synthetic and intentionally small. Local gate
passes stay generated; no Harbor execution is implied. The renderer reruns the
example, asserts those results, and emits static alternatives alongside the loop.
A production-job recording can replace the synthetic example once a reproducible,
shareable source and real runtime validation are available.

## Current opening

Re-read PRs #154–#158 and the parser/generator/materializer contracts before this
revision. Center the product on whole-job conversion into Harbor format, without
reducing it to failure analysis or numerical answer checking. The visible paths
are the actual materializer outputs. The bundled revenue fixture supplies the
source request and starting file; its arithmetic is not the marketing story.

Use large type and an open composition, black/white surfaces, and a restrained
blue accent. Keep only bandr as the brand. Reveal the Harbor package with smooth
25 fps eased motion; hold the finished result and crossfade the loop. Export
actual GIFs, lossless WebPs, and static/reduced-motion alternatives. Verify the
example and the displayed package paths while rendering. No model-solve,
conversion-speed, or Harbor-validation claim is made by the animation.

## Downstream execution loop

Keep the approved conversion composition, then continue with a model/agent
attempt inside Harbor, a verifier reward, and the resulting run traces returning
to the next conversion. Each stage gets its own readable scene instead of
squeezing four panels into the README width. The verifier values are sourced
from the runnable fixture's local gate evidence. The model segment is schematic.
The loop links artifacts and executions; it does not imply automatic model
training, automatic launches by Mint, or an observed model-improvement metric.
