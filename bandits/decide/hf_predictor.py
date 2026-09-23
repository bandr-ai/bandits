"""A real ``LogitPredictor`` backed by a pinned Hugging Face model.

One of two modules in ``bandits.decide`` allowed to import torch/transformers
(the other is ``hf_trainer.py``), and only inside function bodies --
importing this module at all requires the ``decide`` extra, but *loading* it
(importing ``bandits.decide.scorer`` or ``bandits.decide.dataset``) never
pulls torch in, since nothing else in the package imports this module at
module scope.
"""

from __future__ import annotations

from bandits.decide.scorer import LogitPrediction, TokenizationError


def letter_token_id(tokenizer, prompt: str, letter: str, *, model_label: str) -> int:
    """The token id a letter takes on in the *actual rendered prompt* (which
    already ends in the template's "Answer:" cue), not a letter tokenized in
    isolation. Tokenizing "Answer:" and "Answer: A" separately and assuming
    the former's ids are a prefix of the latter's is not guaranteed for a
    BPE tokenizer -- adding text can retokenize the boundary between them.
    Instead, tokenize the full prompt with and without the letter appended
    and verify the shared prefix explicitly: only if the first N ids are
    identical and exactly one new id follows is that new id trusted as "the
    letter's token in this exact context". Raises TokenizationError
    otherwise. Shared by the frozen predictor and the trainer so both read
    letters through the exact same contract -- see ``HFPredictor.predict``
    and ``HFTrainer``."""
    base_ids = tokenizer.encode(prompt, add_special_tokens=False)
    with_letter_ids = tokenizer.encode(f"{prompt} {letter}", add_special_tokens=False)
    if with_letter_ids[: len(base_ids)] != base_ids:
        raise TokenizationError(
            f"appending option letter {letter!r} retokenized the prompt itself "
            f"(not just added a token) for {model_label}; "
            "the letter cannot be read as a single token in this context"
        )
    answer_ids = with_letter_ids[len(base_ids) :]
    if len(answer_ids) != 1:
        raise TokenizationError(
            f"option letter {letter!r} is not exactly one token in this prompt's "
            f"answer context for {model_label} (got {len(answer_ids)} tokens)"
        )
    return answer_ids[0]


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
        adapter_path: str | None = None,
    ) -> None:
        """``adapter_path``, when given, loads a LoRA adapter (saved by
        ``HFTrainer.save_adapter``) on top of the pinned base model. This is
        how a trained checkpoint is scored through the exact same code path
        as the frozen baseline -- ``HFTrainer`` uses this same class
        internally for its own forward passes during training, so training
        and post-training scoring can never silently diverge (see
        ``hf_trainer.py``)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.dtype = dtype
        self.adapter_path = adapter_path

        torch_dtype = getattr(torch, dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch_dtype)
        if adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        self._model = model.to(device)
        self._model.eval()
        self._torch = torch

    def token_count(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

    def predict(self, prompt: str, letters: list[str]) -> LogitPrediction:
        torch = self._torch
        model_label = f"{self.model_id}@{self.revision}"
        letter_token_ids = {
            letter: letter_token_id(self._tokenizer, prompt, letter, model_label=model_label)
            for letter in letters
        }
        distinct = set(letter_token_ids.values())
        if len(distinct) != len(letter_token_ids):
            raise TokenizationError(
                f"requested letters do not map to distinct tokens: {letter_token_ids}"
            )

        # No padding: we tokenize and score exactly one prompt per call, so
        # there is nothing to pad against, and `padding=True` on a batch of
        # one raises on any tokenizer with no pad token configured (GPT-2,
        # Llama, Mistral, ...). The final position is simply the last token.
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        logits = outputs.logits[0, -1, :].float()

        letter_logits = {letter: float(logits[token_id].item()) for letter, token_id in letter_token_ids.items()}
        full_vocab_logsumexp = float(torch.logsumexp(logits, dim=-1).item())
        return LogitPrediction(letter_logits=letter_logits, full_vocab_logsumexp=full_vocab_logsumexp)
