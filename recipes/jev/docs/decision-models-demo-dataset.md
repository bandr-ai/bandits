# Decision Models: demo dataset and success bar (#67)

2026-09-24. Part of [decision-models-plan.md](decision-models-plan.md) (step A0). The success bar in §3 is fixed as of this commit, before any training run (#68). No test split has been scored.

## Status

| Step | State |
| --- | --- |
| Label source + license audit for every candidate | Done (§1) |
| Import each candidate with fixed train / dev / calibration / test | Done: converter `recipes/jev/scripts/decision_candidates.py`; every candidate imports with 0 quarantined rows (§2) |
| Success bar written before #68 runs | Done (§3) |
| Frozen dev scores for each candidate | **Not run.** Needs a GPU and the base-model choice (Qwen3.5-4B vs 9B, plan Q2/Q3). Commands in §4. |
| Pick the winner | Rule fixed in §5; provisional pick is CaseHOLD, confirmed only after §4's dev scores |

## 1. Audit

Licenses are from each dataset card's metadata. Label source comes from the card, the paper, or both.

| Candidate | Options | License | How labels were made | Audit |
| --- | --- | --- | --- | --- |
| LexGLUE `case_hold` (`coastalcph/lex_glue`) | 5, different per item | CC-BY-4.0 | **Found, not annotated:** the correct holding is the one the citing court actually wrote for the cited case (Harvard case-law corpus, Zheng et al. 2021). The four wrong holdings are other real holdings. No LLM involved. Card: `annotations_creators: found`. | **Pass** |
| `zeroshot/twitter-financial-news-topic` | 20, fixed | MIT | Card says "annotated corpus" but never says who annotated it or how. Card: `annotations_creators: other`. | **Conditional**: license is fine, but label provenance is undocumented |
| `jackhhao/jailbreak-classification` | 2 | Apache-2.0 | The label records **which source a prompt came from**, not a judgment about the prompt: jailbreaks come from `verazuo/jailbreak_llms`; "benign" comes from OpenOrca and GPTeacher, which are partly LLM-generated. A model can learn the style of each source instead of what a jailbreak is. | **Fail** (labels are source of origin, partly LLM-made; 1.3k rows) |
| `deepset/prompt-injections` | 2 | Apache-2.0 | Undocumented: the card is a placeholder. | **Fail** (unknown provenance; 662 rows, test half is 50) |
| `tals/vitaminc` | 3 | CC-BY-SA-3.0 | Claim/evidence pairs from real Wikipedia revisions, labeled by crowd workers, **plus** synthetically built pairs. The `revision_type` field separates them; we keep `real` only. | **Pass as the external set** (real revisions only) |

Every license allows publishing results. CC-BY-SA only matters if we redistribute the data itself, which we don't. Rejected before the audit: Banking77 (77 options), ContractNLI (non-commercial).

Caveats that apply whichever dataset wins:

- **Contamination.** CaseHOLD (public since 2021) and VitaminC may be in the base model's pretraining data. That affects untrained and trained alike, so trained − untrained stays honest. It does weaken any claim against Jev or other models. VitaminC rows carry a BIG-bench canary string.
- CaseHOLD's option ids are the positions `0`–`4`. The prompt builder re-letters options, and training shuffles their order, so a position shortcut can't be learned. Macro F1 over positions is still reported, but accuracy is the headline number.

## 2. Splits

`scripts/decision_candidates.py` (in the recipe) downloads each dataset's parquet from Hugging Face. It records the dataset commit and each shard's sha256 in `<name>.provenance.json`, then writes `jev import` JSONL. The publisher's own test split becomes our locked test wherever one exists. Anything missing is carved from a hash of the row's content (never its position), so rerunning can't move a row between splits. Rows carry no explicit id, so the importer identifies them by content: an exact duplicate that lands in two different splits is quarantined on both sides, so no row can sit across train/test.

| Candidate (commit) | Publisher splits | Ours after import (train / dev / calibration / test) |
| --- | --- | --- |
| case_hold (`c23fdff`) | train 45,000 · validation 3,900 · test 3,600 | ≈40,536 / 3,899 / 4,463 / 3,600 (a few exact duplicates removed) |
| twitter (`acbc8af`) | train 16,990 · validation 4,117 | 15,344 / 2,090 / 1,646 / 2,027 (validation halved into dev and test) |
| jailbreak (`2f2ceeb`) | train 1,044 · test 262 | ≈927 / 133 / 102 / 129 (duplicates removed) |
| injections (`4f61ecb`) | train 546 · test 116 | 506 / 66 / 40 / 50 |
| vitaminc (`be6febb`), external | test 55,197 | – / 2,898 / – / 3,101 (944 / 1,007 `case_id` groups; `--limit 6000` whole groups, real revisions only) |

No row's state appears in more than one split in any imported dataset (checked after import).

**VitaminC is split in two on purpose.** The go/no-go (§3) has to look at an external set, and that makes the set a decision input. So the go/no-go uses VitaminC **dev**, and the final report (#69) uses VitaminC **test**, which nothing has looked at. JevBench public, imported with `import_jevbench` (always `test`), goes next to it in the report.

## 3. Success bar (fixed before #68)

Compare the best-on-dev checkpoint from `jev train` against the untrained base on the chosen dataset's **dev** split. Use the same base model and revision, single option order, the same prompt version, and paired rows (scored by both). Intervals are paired percentile bootstraps with 2,000 draws and seed 0. They resample by `group_id` where the dataset has groups (VitaminC) and by row otherwise, as `jev report` computes them (#69).

**Go** only if all three hold:

1. **Real gain:** the 95% interval of trained − untrained **accuracy** on dev has its lower bound above 0.
2. **Big enough to show:** the point estimate is at least **+5 accuracy points**. A statistically real +1 point on 3,900 rows doesn't make a launch demo.
3. **No meaningful loss elsewhere:** on the external VitaminC **dev** half (2,898 rows in 944 `case_id` groups, resampled by group), the 95% interval of trained − untrained accuracy has its lower bound at or above **−3 points**.

**No-go** otherwise. Per the plan, change the setup first (the pre-registered fallback is attention-only LoRA and/or a lower learning rate), then the dataset (§5's fallback order). Every retry is judged by this same bar on the same dev splits. Calibration (temperature) is not part of go/no-go. It is fitted afterwards on the calibration split.

The locked test splits (the chosen dataset's `test`, VitaminC `test`) are scored once, for the final report, and never before go/no-go.

## 4. Frozen dev scoring (not run yet: needs a GPU)

```bash
uv run --with pyarrow python scripts/decision_candidates.py case_hold twitter jailbreak injections --out work/candidates
uv run --with pyarrow python scripts/decision_candidates.py vitaminc --limit 6000 --out work/candidates
for n in case_hold twitter jailbreak injections vitaminc; do
  uv run jev import work/candidates/$n.jsonl --project work/decide
done
# For each dataset id printed above (dev only; never --allow-test):
DATASET_ID=decision-dataset-...        # from the import output
REVISION=...                           # the pinned model commit SHA
uv run --extra train jev score "$DATASET_ID" --split dev \
  --model Qwen/Qwen3.5-4B --revision "$REVISION" --project work/decide
```

Each run is saved as a `decision_scorer_run` artifact. Fill in:

| Candidate | Base | Untrained dev accuracy | Majority-class dev accuracy | Rejected | Scorer run |
| --- | --- | --- | --- | --- | --- |
| case_hold | | | 0.206 | | |
| twitter | | | 0.215 | | |
| jailbreak | | | 0.571 | | |
| injections | | | 0.500 | | |
| vitaminc (external) | | | 0.504 | | |

Prompts are short: CaseHOLD's median is about 1,600 characters of state plus options (p99 about 2,300), well under the 8k-token limit, so no rejections are expected. Jailbreak has a long tail (p99 about 6,000 characters).

## 5. Choosing the winner

Only candidates that passed the audit are eligible: **CaseHOLD**, then **Twitter topics** as the fallback. Jailbreak and prompt-injections fail the audit and are dropped from the demo. They're also too small: their dev and test sets have 50–133 rows, which gives confidence intervals around ±10 points.

Rule, fixed now:

- Pick **CaseHOLD** if its untrained dev accuracy is **below 90%**. It is the only candidate whose labels are fully traceable, and it shows the Jev shape best: a state, a question, and five candidate answers that change with every item, not one fixed label set.
- Otherwise, or if CaseHOLD fails §3's bar after the setup fallback, move to **Twitter topics**. The launch copy must then say that the dataset's annotation process is undocumented.
- If both fail, stop. No demo claim is made from a dataset that failed the bar.

Record the decision here, with the §4 table filled in, before #68's final run.
