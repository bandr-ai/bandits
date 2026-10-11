## What I learned from the pilot

The pilot's failures were almost entirely contract and consistency failures rather than story drift: the world was preserved, but the trajectories broke format (missing `[obs:N] `), broke vocabulary (a variation that demanded `keyboard_occluded` from the element-presence check), broke completeness (`different_page` without `page_kind`), or broke evidence consistency (a comparison read collapsed to a live-only description, and context advertising slots a tool then reported uncaptured). The seed-preservation rules were doing their job; the contract rules were not. So this version keeps the preservation language nearly intact and rewrites the contract language as hard, checkable rules.

## Harness description

Kept the structure (task / input / output / per-tool observability / evidence rules / unknowns) because it worked, and added:
- the `[obs:N] ` prefix as an explicit part of the tool-result contract, tagged [inferred: seeds + real tool results] with the producer flagged [unknown];
- `page_kind` as a required key of the `element_presence` output when the conclusion is `different_page`, tagged [established: S1, shaping.py];
- for every tool, the distinct payload for each outcome (success, empty-but-captured, unavailable, unverifiable, error), because the pilot's most common semantic error was collapsing two of them;
- a note that S1's JSON contract enumerates a shorter cause list than the cause-knowledge block under it — flagged [unknown] rather than resolved into a rule, since the source does not say which binds.

## Variations

- **Dropped** `keyboard_covers_target_tree_sees_it`: it named `keyboard_occluded` / `NSE_KEYBOARD_OCCLUSION`, which the element-presence check's vocabulary does not contain, so it could never produce a valid run. **Replaced** by `target_covered_presence_occluded` (S1/S10), which teaches the same decision boundary — present-in-tree plus covered-region is a hand-off — using the presence check's own term `occluded` with null cause.
- **Dropped** `overlay_covers_region_no_source`. It was unexercised and its teaching (covered with no source is occluded) is already carried by `target_covered_presence_occluded` plus the two source-missing variations; that slot is better spent on the "two sources disagree" case the request names, so **added** `a11y_tree_captured_empty_visual_affirms`, which is a genuine decision moment from S1's own text (an empty programmatic search plus an affirmative crop read is still found_in_place).
- **Fixed** `a11y_tree_not_captured`: the variation now also removes the tree slot from `evidence_available`, and its `correct_when` is check-agnostic (no conclusion may convert a missing tree into absence or cite node counts) so it holds for S3/S5 as well as S1/S10.
- **Fixed** `live_screenshot_missing_tree_only` the same way, and pinned the exact unavailable message the tool returns.
- **Fixed** `generic_control_on_signin_page`: `page_kind: "sign_in"` is now a required JSON key, not a prose detail.
- **Fixed** `spinner_affirms_loading_no_har`: added the rule that any regenerated comparison read must name both sides or say which side is missing.
- **Kept** the pilot's clean performers unchanged in substance: `keyboard_shown_target_above`, `app_dialog_not_system_owned`, `same_page_promo_overlay_not_variant`, `probe_unverifiable_record_alias`, `locale_shift_same_page`, `fresh_session_interstitial_history` (the last two unexercised, so their vocabulary was re-checked against their own prompts: `locale_variance`/`NSE_LOCALE_VARIANCE` and `NSE_SESSION_INTERSTITIAL` are both in-contract).
- Every variation now states what must not be claimed, which is what stops a variation's prose from overriding the system prompt.

## Template

Changes in response to the diagnosis, consolidating rather than accumulating:
- the `[obs:N] ` rule is one numbered ground rule (rule 7) and also appears in `<obs_examples>` as real prefixed payloads — one rule in two places, not two rules;
- the four-key output paragraph is gone; it is replaced by "the keys the system prompt names", with the extra-key case (`page_kind`) stated once in both the harness description and rule 8;
- vocabulary is now explicitly closed, with `<agent_system_prompt>` as the sole authority and an explicit fallback instruction for a variation that names an out-of-vocabulary term — this is the template-level guard the pilot asked for and it subsumes the older, weaker "closest valid term" note;
- a cause-naming guard (rule 9) replaces the informal "be cautious" framing: name a cause only when the run's own generated evidence supports it, otherwise null;
- a comparison-read rule (rule 10) replaces the implicit expectation that reads describe both sides;
- the required-args rule (rule 1) now says every declared field, with `""` wildcards;
- the two rules that previously overlapped (harness-contract discipline and evidence reading) were merged and re-cut as rules 4 and 5 so neither repeats the other;
- removed the instruction to describe the output format (code appends it), and removed any wording that treated a variation's `correct_when` as a vocabulary source.

## Residual risk

The pilot reported near-duplicates among kept runs. The template's "only this changes" plus the variation's `dependencies` are what force the evidence to actually differ, so I made each `dependencies` block name the concrete payloads and sentences that must change — but a generator that only paraphrases the detail sentence would still produce a near-duplicate, which the contract rules alone cannot prevent.