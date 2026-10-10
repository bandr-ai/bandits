## What I changed in response to the diagnosis

**1. The availability lie (top rejection, `locale_shift_same_page`).** Both rejected and weak traces invented a runtime a11y tree the run did not advertise. Two fixes: (a) the harness now states a *slot-scoped availability invariant* — `available: true` is possible only for a slot `evidence_available` names, and the absent slot returns the tool's own unavailable payload with its reason (query_har is the documented exception); (b) template Ground rule 3 makes updating `evidence_available` explicit permission inside the ONE change rather than drift, and Ground rule 4 makes re-evaluating every call that targeted the changed screen mandatory rather than a general principle. Ground rule 12's old wording ("everything else stays as in the seed") is now qualified so a slot mutation is not read as forbidden.

**2. Stale observations (weak kept, same seed).** Old a11y node counts survived a live-page change. Ground rule 4 now names that exact failure mode (an a11y landmark search aimed at the old page) and tells the generator to update the result or drop the call.

**3. Call-count pressure.** Removed the line "Keep the seed's call count, roughly — one more or one fewer". Replaced with "call count may change freely; evidence consistency beats call count" inside rule 4 — one place instead of two conflicting instructions.

**4. "Every declared field must be present".** Removed. The harness fills omitted arguments from `arg_defaults` before validation, and the generated traces legitimately omit wildcard fields. Rule 1 now states the true contract (pass what the agent knows; `""` is the wildcard; an omitted argument is defaulted).

**5. Cause-sweep checklist rejection (`probe_unverifiable_record_alias` on S6).** The variation's `correct_when` demanded visual/tree follow-up that an S6 seed cannot perform, and S6 outputs carry no conclusion key for the judge's checklist. The variation is now restricted to S1/S10, `requires_tools` is only `locator_probe`, and its `correct_when` is written in terms of what a presence check can do with a runtime probe. No S6-targeted variation remains.

**6. `locale_shift_same_page`.** `a11y_search` is no longer listed as required and its dependency sentence now says: corroborate with a landmark only when `runtime_a11y` is advertised; otherwise keep the unavailable payload. Same conditional was added to `same_page_promo_overlay_not_variant`.

**7. `target_covered_presence_occluded`.** Preserve wording fixed: "the interacting step the seed records" instead of "the step that entered text". Also rewrote a leaky phrase ("never keyboard_occluded") into a rule-shaped sentence.

**8. `fresh_session_interstitial_history`.** Verified `NSE_SESSION_INTERSTITIAL` *is* in S7's closed catalog ("Overlay fingerprints; execution history (new-session pattern)"), so the variation is kept; its dependency now names the catalog row and requires the `walk_history` call to precede the cause.

**9. Harness consolidation.** The loose availability statements scattered across the input contract, the tool sections and the rules are merged into one invariant paragraph plus rule 1. The exact result shapes (success / empty / unavailable / error / no-live-anchor message) for all five tools are stated per tool as before; the `[obs:N] ` prefix rule moved out of the input contract prose into the tool-result paragraph ("N increments once per tool result, including error and unavailable payloads") so the generator sees it once. `evidence_available` and the prefix remain tagged [inferred] because their producers are not in the visible source.

## What I kept deliberately

- The five tools' contract detail and the 14 evidence-to-answer rules (shaping.py's demotions, the fork's step-id adjudication, D44's missing-side rule, D36/D41's absence demotions) — these are what the generator must obey, not invent.
- The ten variations that kept 100% of their jobs (`a11y_tree_captured_empty_visual_affirms`, `spinner_affirms_loading_no_har`, `keyboard_shown_target_above`, `target_covered_presence_occluded`, `app_dialog_not_system_owned`, `generic_control_on_signin_page`, `same_page_promo_overlay_not_variant`) are unchanged except for the wording fixes above; each remained contract-shaped in the pilot.

## Known, unaddressed-in-template

The pilot's near-duplicate rate came partly from over-sampling one variation. That is a scheduling choice, not a variation defect, so it is not fixed here: cap per-variation job counts (e.g. ≤ 2) and spread jobs across seed families before regenerating.
