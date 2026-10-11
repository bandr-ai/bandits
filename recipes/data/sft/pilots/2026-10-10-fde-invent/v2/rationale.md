## What the seeds actually are

Each inspector is a single-question analyzer with a frozen enum contract (conclusion / root_cause / confidence / detail), a small toolset, and one deterministic answer at the end. The tool contracts are narrow and non-overlapping: compare_screenshots is pure vision over whatever images shipped (it can say 'different page', 'no', 'OS/system dialog', but never HAR, DOM, tree, ancestry or a cause); a11y_search returns role/name nodes and a count per side; locator_probe only counts (unique / no_match / not_unique / unverifiable / unavailable); query_har only knows the network log and is often legitimately unavailable; walk_history only carries run history, record-time facts and driver contexts. Several seeds legitimately end in null / page_ready / not_occluded with honest low-to-mid confidence and even explicitly state that nothing was observed. Downgrading to null when the detecting artefact was never returned is the normal, correct behaviour of this agent family, not a rare fallback.

## What the pilot got wrong (from the diagnosis)

1. Positive causes named without their detecting evidence (NSE_SHADOW_BOUNDARY from a locator miss plus an a11y hit).
2. Final details asserting artefacts no tool read (runtime HTML, DOM ancestry, record-side content).
3. Record-side results cited as proof of live facts.
4. Strong outcome skew toward positives (13/18) versus the seeds where null/carry-no-cause dominates — this teaches over-diagnosis.
5. Near-copies of seeds in repeated jobs (same element, locator stem, page kind).
6. Low meta-diversity (mostly sign-in divergences, similar tool orders).
7. Attribution slips in the detail sentence (live capture vs recorded crop).
Formatting was clean (20/20 format_ok, valid args), so nothing about the mechanical shape needed changing.

## Why the template now says what it says

- **Sections 2–5 are the fix for causes 1–3 and 7.** Section 2 forces one evidence budget, declared in evidence_available and honoured by every tool result, which is where the pilot drifted. Section 4 is the required-evidence matrix: for each cause family it names the artefact that must actually appear in a tool result (shadow map / frame tree / window inventory / HAR / readyState / live-screenshot occlusion read / disabled signal / crash or ANR evidence / screenshot affirming absence, etc.), states that a locator miss plus an a11y hit is not shadow evidence, requires calling the tool that would expose the cause when its input exists, and states flatly that side 'record' is baseline-only and cannot support a live fact. Section 5 adds the grounding linter for the detail sentence: every clause maps to an [obs:N] or the failure context, and artefacts must be attributed to the source that actually returned them (live screenshot vs recorded crop vs runtime tree).
- **Section 8 is the fix for cause 4.** It restates the seed batch's real balance, prefers the cautious outcome under ambiguity, caps positives at one per trajectory and only when the matrix row is satisfied, and forbids copying the seed's verdict to imitate its shape. Section 7 covers causes 5 and 6: anti-copy on names, locators, verbs, observation sentences and failure family; forced variation in investigation order (which family opens the case) and in page kind (explicitly no default to sign-in for divergent verdicts).
- **Sections 1 and 6 keep what worked.** The message-list shape, the literal 'Failure context:' opener, the optional deterministic hint, empty assistant content on tool turns, the call-(uuid)-(n) id pattern, [obs:N] numbering, the occasional 'almost out of steps' nudge, and the no-narration rule are all preserved from v1 because the pilot recorded 20/20 format_ok. Section 1 adds a platform artefact checklist so web vs mobile vocabulary does not drift.

The overall stance is that the generator should invent a case whose evidence genuinely leads to a verdict — and that the honest verdict is frequently 'nothing supported', which the template now treats as a first-class outcome rather than a failure to diagnose.