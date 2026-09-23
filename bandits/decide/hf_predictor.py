"""A real ``LogitPredictor`` backed by a pinned Hugging Face model.

The only module in ``bandits.decide`` allowed to import torch/transformers,
and only inside function bodies -- importing this module at all requires the
``decide`` extra, but *loading* it (importing ``bandits.decide.scorer`` or
``bandits.decide.dataset``) never pulls torch in, since nothing else in the
package imports this module at module scope.
"""

from __future__ import annotations

from bandits.decide.scorer import LogitPrediction, TokenizationError

_ANSWER_CUE = "Answer:"


class HFPredictor:
    """One forward pass over a frozen (or LoRA-adapted) causal LM, never
    ``generate()``. Pinned to an exact model id + revision so a scorer run
    or a training run can be reproduced byte-for-byte from its recorded
    contract."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.dtype = dtype

        torch_dtype = getattr(torch, dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=torch_dtype
        ).to(device)
        self._model.eval()
        self._torch = torch

    def token_count(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

    def _letter_token_id(self, letter: str) -> int:
        """The token id a letter takes on immediately after "Answer:" --
        not just any encoding of the bare letter, since many tokenizers
        encode " A" and "A" differently and only the former is what the
        model will actually produce after the prompt's own cue. Raises
        TokenizationError unless that encoding is exactly one token."""
        probe = f"{_ANSWER_CUE} {letter}"
        cue_ids = self._tokenizer.encode(_ANSWER_CUE, add_special_tokens=False)
        probe_ids = self._tokenizer.encode(probe, add_special_tokens=False)
        answer_ids = probe_ids[len(cue_ids) :]
        if len(answer_ids) != 1:
            raise TokenizationError(
                f"option letter {letter!r} is not exactly one token after "
                f"{_ANSWER_CUE!r} for {self.model_id}@{self.revision} "
                f"(got {len(answer_ids)} tokens)"
            )
        return answer_ids[0]

    def predict(self, prompt: str, letters: list[str]) -> LogitPrediction:
        torch = self._torch
        letter_token_ids = {letter: self._letter_token_id(letter) for letter in letters}
        distinct = set(letter_token_ids.values())
        if len(distinct) != len(letter_token_ids):
            raise TokenizationError(
                f"requested letters do not map to distinct tokens: {letter_token_ids}"
            )

        inputs = self._tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        # attention_mask marks real (non-pad) tokens per row; the last 1 in
        # each row is that row's true final position, regardless of whether
        # the tokenizer pads left or right.
        attention_mask = inputs["attention_mask"][0]
        last_real_position = int(attention_mask.nonzero()[-1].item())
        logits = outputs.logits[0, last_real_position, :].float()

        letter_logits = {letter: float(logits[token_id].item()) for letter, token_id in letter_token_ids.items()}
        full_vocab_logsumexp = float(torch.logsumexp(logits, dim=-1).item())
        return LogitPrediction(letter_logits=letter_logits, full_vocab_logsumexp=full_vocab_logsumexp)
