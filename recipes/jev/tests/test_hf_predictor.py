"""Tokenizer-contract checks against a real tiny public tokenizer
(hf-internal-testing/tiny-random-gpt2). Skipped without the `decide` extra."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from bandits_jev.hf_predictor import encode_prompt
from bandits_jev.scorer import TokenizationError

_MODEL_ID = "hf-internal-testing/tiny-random-gpt2"


def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(_MODEL_ID, revision="main")


def test_encode_prompt_keeps_the_answer_cue_as_the_last_position() -> None:
    tokenizer = _tokenizer()
    prompt = "Question: which fruit\nAnswer:"
    inputs = encode_prompt(tokenizer, prompt, "cpu", model_label=_MODEL_ID)
    assert inputs["input_ids"][0].tolist() == tokenizer.encode(prompt, add_special_tokens=False)


def test_encode_prompt_rejects_a_tokenizer_that_appends_eos() -> None:
    """A tokenizer configured to append EOS would move the readout to the
    position after EOS; reject it rather than read the wrong logits."""
    from tokenizers.processors import TemplateProcessing

    tokenizer = _tokenizer()
    eos = tokenizer.eos_token
    tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
        single=f"$A {eos}", special_tokens=[(eos, tokenizer.eos_token_id)]
    )
    with pytest.raises(TokenizationError, match="trailing special token"):
        encode_prompt(tokenizer, "Question: which fruit\nAnswer:", "cpu", model_label=_MODEL_ID)
