# Review notes

## 1. Most common rejection: missing `[obs:N]` harness prefix (3/6 rejected traces; 6 rule issues)
- `v1/01a0de87-7f4f-7f82-8029-228201239f59__0`: locator_probe results 1 and 2 and compare_screenshots result 3 all lack `[obs:N]`.
- `v1/01a0e008-9fbb-7af0-95f4-ab6b76735db5__1`: locator_probe results 1 and 2 lack prefix.
- `v1/01a0de24-8f30-7852-aeb4-70db1aadab17__0`: a11y_search result 1 lacks prefix; later compare_screenshots results are `[obs:2]` / `[obs:3]`, so the sequence is also off by one.
- Pilot counts: 2x tool result 1 locator_probe, 2x tool result 2 locator_probe, 1x a11y_search, 1x compare_screenshots.
- Template fix: hard rule that every `TOOL_RESULT` content begins with `[obs:N] ` and N increments once per tool result, including error and unavailable payloads. Never emit raw JSON as the observation. This belongs in both <obs_examples> and the ground rules.

## 2. Invalid variation vocabulary: `keyboard_covers_target_tree_sees_it`
- Trace `v1/01a0de24-8f30-7852-aeb4-70db1aadab17__1` returns `keyboard_occluded` and `NSE_KEYBOARD_OCCLUSION`; the pilot rejects both because they are not in the element_presence conclusion/cause lists. The harness vocabulary has `occluded` as a hand-off with no root cause, not `keyboard_occluded`.
- The variation itself instructs the invalid answer, so the run count is 1 job, 0 kept.
- Verdict: drop, or rewrite to `occluded` with `root_cause: null`. Do not let a variation's correct_when override <system_prompt>.
- Template fix: before writing the final JSON, verify conclusion/root_cause against the system prompt's own vocabulary. If the variation names an out-of-vocabulary term, fall back to the closest valid term or treat the variation as invalid.

## 3. Missing required `page_kind` on `different_page`
- Trace `v1/01a0de87-7f4f-7f82-8029-228201239f59__1` (generic_control_on_signin_page): the crop read plainly reports a sign-in page, the final answer is `different_page` / null, but it omits `page_kind: sign_in`; judge fails task_success/policy. The variation's dependencies require page_kind.
- Fix template output contract: base keys plus any additional required field named by the system prompt. For `different_page`, `page_kind` must be in the JSON, not only in the prose detail.
- Variation verdict: keep but fix to state the required JSON field explicitly.

## 4. Single-side comparison read in a rejected spinner run
- `v1/01a0de3d-1176-7ca3-8940-035c3e4ed92c__2`: the full_page compare_screenshots read only says the live screenshot shows an active spinner/skeleton and omits the recorded reference. The two kept spinner runs (`__0`, `__1`) both describe recorded versus live. This run was rejected with `obs_consistent=false` even though the judge checks passed.
- Fix template: for comparison kinds, the vision read's observations must describe each available side (recorded and live) or explicitly state that a side is missing. Do not collapse a comparison into a live-only description.
- Variation verdict: keep, but add a dependency that the full_page read is regenerated consistently and names both sides.

## 5. Uncaptured sources leave stale `evidence_available`
- In a11y_tree_not_captured kept runs `v1/01a0de28-d969-7ab0-9d94-02ed01d772ec__0` and `v1/01a0de25-9668-71b2-9866-e44fcd3ead6d__0`, the context still advertises `runtime_a11y` / `record_a11y` while a11y_search returns no captured tree. The judge flags the tension. Not fatal because the agent never relied on the missing tree, but it weakens consistency.
- Fix variation: when a variation makes a source uncaptured, either drop that slot from `evidence_available` or state explicitly that the unavailable payload is the canonical signal and the context is intentionally preserved. The same risk applies to `live_screenshot_missing_tree_only` and `overlay_covers_region_no_source`.
- Verdict: fix a11y_tree_not_captured.

## 6. Variation correct_when too narrow for multi-check variations
- `a11y_tree_not_captured` applies to S1, S10, S3 and S5, but its correct_when talks only about element_presence `found_in_place`. Kept runs `v1/01a0de28-d969-7ab0-9d94-02ed01d772ec__0` and `v1/01a0de25-9668-71b2-9866-e44fcd3ead6d__0` are page-identity checks and correctly return `diverged`.
- Fix variation: make correct_when check-agnostic: no conclusion may convert the missing tree into absence, and no conclusion may cite node counts; the remaining screenshot/probe evidence decides.

## 7. Probe variation is mostly sound, but guard root-cause naming
- `probe_unverifiable_record_alias` rejected traces `v1/01a0de87-7f4f-7f82-8029-228201239f59__0` and `v1/01a0e008-9fbb-7af0-95f4-ab6b76735db5__1` were lost only to missing prefixes. Their content respects `unverifiable` as no information and answers from the runtime side.
- In `v1/01a0de87-7f4f-7f82-8029-228201239f59__0`, `found_in_place` is given `NSE_SAME_PAGE_LOCATOR` from a runtime `no_match` plus a positive visual read. That is plausible locator drift, but the variation is teaching `unverifiable != no_match`, not cause naming. Add a template guard: do not name a root cause unless the check's detecting evidence is present and the variation/correct_when asks for one; otherwise use the conclusion's default/null.
- Verdict: keep.

## 8. Verdict per variation
- keep: `spinner_affirms_loading_no_har` (3 jobs, 2 kept; fix the single-side full_page read), `keyboard_shown_target_above` (4/4 kept), `same_page_promo_overlay_not_variant` (2/2 kept), `app_dialog_not_system_owned` (2/2 kept), `probe_unverifiable_record_alias` (3 jobs, 1 kept; prefix loss only, add cause guard).
- fix: `a11y_tree_not_captured` (3 jobs, 2 kept; evidence_available and correct_when), `generic_control_on_signin_page` (2 jobs, 1 kept; require page_kind in JSON).
- drop or rewrite: `keyboard_covers_target_tree_sees_it` (1 job, 0 kept; invalid conclusion and cause).
- unexercised: `live_screenshot_missing_tree_only`, `overlay_covers_region_no_source`, `locale_shift_same_page`, `fresh_session_interstitial_history` have no jobs in the pilot's by_variation table. Keep for the next pilot, but verify their vocabulary (`locale_variance` / `NSE_LOCALE_VARIANCE`, `NSE_SESSION_INTERSTITIAL`) and update `evidence_available` when a source becomes uncaptured.

## 9. Template changes, including removals
- Add the hard `[obs:N]` prefix rule.
- Replace the four-key-only output contract with: base keys plus any required key named in the system prompt, e.g. `page_kind` for `different_page`.
- Add a pre-final vocabulary check: conclusions and root causes may come only from <system_prompt>; variation examples cannot create new terms.
- Add a comparison-read rule: for comparison kinds, describe both available sides or state which side is missing.
- Add a required-args rule: include every `args_schema` field and use `""` wildcards when not constraining (the kept a11y_tree run `v1/01a0de28-d969-7ab0-9d94-02ed01d772ec__0` did this correctly for a11y_search).
- Clarify `only this changes` so it forbids semantic drift but still requires every dependent observation, later call and required output key to be updated.
- Remove any wording that treats the variation's correct_when as a vocabulary source; leave <system_prompt> as the sole authority.
- Keep the seed-preservation rules as written. The failures here are format/contract failures plus one invalid variation, not story drift.

## 10. Minor pilot note
- The pilot reports 6 near-duplicates among kept runs. That is not a reason to ban seed content, but for training value prefer one high-fidelity run per seed/variation or ensure repeated runs actually change the evidence the agent must handle rather than only paraphrasing the final detail.