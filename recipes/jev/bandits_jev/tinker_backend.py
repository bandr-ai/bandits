"""Train and score through the Tinker API instead of a local GPU.

``TinkerTrainer`` implements ``trainer.Trainable`` and ``TinkerPredictor``
implements ``scorer.LogitPredictor``, so training, the pipeline and the scorer
run unchanged on either backend.

The loss is ``HFTrainer``'s: cross-entropy over the option letters, renormalized
over the letters alone. Tinker's built-in loss is full-vocabulary, so each option
becomes its own datum whose only target token is its letter; the server returns
that letter's log-probability and the loss is a ``log_softmax`` over the options'
log-probabilities. Scoring reads the same log-probabilities.

Tinker has no LoRA dropout and serves a named base model with no revision to pin,
so a nonzero dropout is refused rather than recorded and not applied. A checkpoint
is a local directory whose ``tinker_checkpoint.json`` names the weights on Tinker.

``tinker`` and ``torch`` (the custom loss runs client-side) are imported inside
function bodies only.
"""

from __future__ import annotations

import platform
from importlib.metadata import version
from pathlib import Path

from bandits.store import Contract
from bandits_jev.dataset import DecisionExample
from bandits_jev.hf_predictor import letter_token_ids
from bandits_jev.prompt import build_prompt
from bandits_jev.scorer import LogitPrediction, TokenizationError

DEFAULT_TINKER_REVISION = "tinker-hosted"
CHECKPOINT_FILE = "tinker_checkpoint.json"

# torch.optim.AdamW as HFTrainer builds it; Tinker's defaults differ.
_ADAM = {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.01}


class TinkerCheckpoint(Contract):
    base_model: str
    step: int
    state_path: str
    """``tinker://`` path of the weights and optimizer state, to resume."""
    sampler_path: str
    """``tinker://`` path of the weights saved for sampling, to score."""


def read_checkpoint(path: str | Path) -> TinkerCheckpoint:
    record = Path(path) / CHECKPOINT_FILE
    if not record.is_file():
        raise FileNotFoundError(f"no {CHECKPOINT_FILE} in {path}; not a Tinker checkpoint")
    return TinkerCheckpoint.model_validate_json(record.read_bytes())


def lr_multiplier(step: int, *, total_steps: int, warmup_steps: int) -> float:
    """Linear warmup then linear decay to zero; the schedule ``HFTrainer`` gets
    from ``get_linear_schedule_with_warmup``. ``step`` is optimizer steps taken."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


def restricted_soft_cross_entropy(letter_logprobs, target_probs):
    """Cross-entropy of ``target_probs`` against the softmax over the letters'
    log-probabilities. A constant shift leaves that softmax unchanged, so this
    equals the same loss over raw logits."""
    import torch

    return -(target_probs * torch.log_softmax(letter_logprobs, dim=-1)).sum()


def encode_prompt_ids(tokenizer, prompt: str, *, model_label: str) -> list[int]:
    """Token ids whose last token is the answer cue; a tokenizer that appends a
    special token is rejected, as in ``hf_predictor.encode_prompt``."""
    ids = tokenizer.encode(prompt)
    base_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not base_ids or ids[-1] != base_ids[-1]:
        raise TokenizationError(
            f"tokenizer for {model_label} appends a trailing special token; "
            "the final position is not the answer cue"
        )
    return list(ids)


class TinkerPredictor:
    """``LogitPredictor`` over a Tinker sampling client: the base model, or the
    weights a ``TinkerTrainer`` checkpoint saved for sampling."""

    dtype = "tinker"
    device = "tinker"
    max_context_tokens = None

    def __init__(self, sampling_client, tokenizer, *, model_id: str, revision: str) -> None:
        self.model_id = model_id
        self.revision = revision
        self._sampling_client = sampling_client
        self._tokenizer = tokenizer

    @classmethod
    def from_service(
        cls, model_id: str, *, revision: str = DEFAULT_TINKER_REVISION, adapter_path: str | None = None
    ) -> TinkerPredictor:
        import tinker

        service = tinker.ServiceClient()
        if adapter_path is None:
            sampling_client = service.create_sampling_client(base_model=model_id)
        else:
            checkpoint = read_checkpoint(adapter_path)
            if checkpoint.base_model != model_id:
                raise ValueError(f"{adapter_path} was trained on {checkpoint.base_model}, not {model_id}")
            sampling_client = service.create_sampling_client(model_path=checkpoint.sampler_path)
        return cls(sampling_client, sampling_client.get_tokenizer(), model_id=model_id, revision=revision)

    def token_count(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

    def predict(self, prompt: str, letters: list[str]) -> LogitPrediction:
        """Tinker's log-probabilities are already normalized over the vocabulary,
        so they stand in for logits with a full-vocabulary log-sum-exp of zero."""
        from tinker import types

        label = f"{self.model_id}@{self.revision}"
        letter_ids = letter_token_ids(self._tokenizer, prompt, letters, model_label=label)
        ids = encode_prompt_ids(self._tokenizer, prompt, model_label=label)
        futures = {
            letter: self._sampling_client.compute_logprobs(types.ModelInput.from_ints([*ids, token_id]))
            for letter, token_id in letter_ids.items()
        }
        letter_logits = {}
        for letter, future in futures.items():
            logprob = future.result()[-1]
            if logprob is None:
                raise TokenizationError(f"Tinker returned no log-probability for letter {letter!r} in {label}")
            letter_logits[letter] = float(logprob)
        return LogitPrediction(letter_logits=letter_logits, full_vocab_logsumexp=0.0)


class TinkerTrainer:
    """LoRA SFT on Tinker over a named base model."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str = DEFAULT_TINKER_REVISION,
        seed: int,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        learning_rate: float = 5e-5,
        training_client=None,
    ) -> None:
        """``training_client`` is for tests; otherwise one is created on ``model_id``
        (needs ``TINKER_API_KEY``)."""
        if lora_dropout != 0.0:
            raise ValueError(
                f"Tinker has no LoRA dropout, but the config asks for {lora_dropout}; "
                "pass --lora-dropout 0 so the recorded config is what ran"
            )
        self.model_id = model_id
        self.revision = revision
        self.lora_config = {"rank": lora_rank, "alpha": lora_alpha, "dropout": 0.0, "target_modules": "all-linear"}
        self.learning_rate = learning_rate
        self.max_context_tokens = None
        if training_client is None:
            import tinker

            training_client = tinker.ServiceClient().create_lora_training_client(
                base_model=model_id, rank=lora_rank, seed=seed
            )
        self._client = training_client
        self._tokenizer = training_client.get_tokenizer()
        self._model_label = f"{model_id}@{revision}"
        self._step = 0
        self._schedule: tuple[int, int] | None = None

    @classmethod
    def from_config(cls, config) -> TinkerTrainer:
        return cls(
            config.base_model_id,
            revision=config.base_revision,
            seed=config.seed,
            lora_rank=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            learning_rate=config.learning_rate,
        )

    def configure_schedule(self, *, total_steps: int, warmup_steps: int) -> None:
        self._schedule = (total_steps, warmup_steps)

    def token_count(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

    def check_letters(self, prompt: str, letters) -> None:
        letter_token_ids(self._tokenizer, prompt, letters, model_label=self._model_label)

    def _learning_rate_now(self) -> float:
        if self._schedule is None:
            return self.learning_rate
        total_steps, warmup_steps = self._schedule
        return self.learning_rate * lr_multiplier(self._step, total_steps=total_steps, warmup_steps=warmup_steps)

    def train_step(
        self, examples: list[DecisionExample], *, option_orders: list[dict[str, str]] | None = None
    ) -> float:
        """One optimizer step over the batch; returns the mean per-example loss."""
        import torch
        from tinker import types

        orders = option_orders or [dict(e.options) for e in examples]
        data = []
        targets: list[list[float]] = []
        for example, options in zip(examples, orders, strict=True):
            prompt, letters = build_prompt(example.state, example.question, options)
            letter_ids = letter_token_ids(self._tokenizer, prompt, letters.values(), model_label=self._model_label)
            ids = encode_prompt_ids(self._tokenizer, prompt, model_label=self._model_label)
            targets.append([example.target.probabilities.get(option_id, 0.0) for option_id in options])
            for option_id in options:
                # only the answer position carries loss, with the option's letter as target
                data.append(
                    types.Datum(
                        model_input=types.ModelInput.from_ints(ids),
                        loss_fn_inputs={
                            "target_tokens": ids[1:] + [letter_ids[letters[option_id]]],
                            "weights": [0.0] * (len(ids) - 1) + [1.0],
                        },
                    )
                )
        mean_loss: list[float] = []

        def loss_fn(_data, logprobs_list):
            losses = []
            offset = 0
            for target in targets:
                letter_logprobs = torch.stack([lp[-1] for lp in logprobs_list[offset : offset + len(target)]]).float()
                losses.append(restricted_soft_cross_entropy(letter_logprobs, torch.tensor(target, dtype=torch.float32)))
                offset += len(target)
            loss = torch.stack(losses).mean()
            mean_loss.append(float(loss.item()))
            return loss, {"loss": mean_loss[-1]}

        self._client.forward_backward_custom(data, loss_fn).result()
        self._client.optim_step(types.AdamParams(learning_rate=self._learning_rate_now(), **_ADAM)).result()
        self._step += 1
        return mean_loss[-1]

    def predictor(self, adapter_path: str | None = None) -> TinkerPredictor:
        if adapter_path is None:
            sampling_client = self._client.save_weights_and_get_sampling_client()
        else:
            sampling_client = self._client.create_sampling_client(read_checkpoint(adapter_path).sampler_path)
        return TinkerPredictor(sampling_client, self._tokenizer, model_id=self.model_id, revision=self.revision)

    def save_checkpoint(self, path: str) -> None:
        name = f"jev-step-{self._step}"
        state = self._client.save_state(name).result()
        sampler = self._client.save_weights_for_sampler(name).result()
        Path(path).mkdir(parents=True, exist_ok=True)
        record = TinkerCheckpoint(
            base_model=self.model_id, step=self._step, state_path=state.path, sampler_path=sampler.path
        )
        (Path(path) / CHECKPOINT_FILE).write_text(record.model_dump_json())

    def load_checkpoint(self, path: str) -> None:
        checkpoint = read_checkpoint(path)
        if checkpoint.base_model != self.model_id:
            raise ValueError(f"{path} was trained on {checkpoint.base_model}, not {self.model_id}")
        self._client.load_state_with_optimizer(checkpoint.state_path).result()
        self._step = checkpoint.step

    def environment(self) -> dict[str, str]:
        return {
            "python": platform.python_version(),
            "backend": "tinker",
            "tinker": version("tinker"),
            "device": "tinker",
            "lora_alpha": "not configurable on Tinker; the config value is not applied",
        }


def make_trainer(config) -> TinkerTrainer:
    return TinkerTrainer.from_config(config)


def make_predictor_factory(model_id: str, *, revision: str = DEFAULT_TINKER_REVISION):
    def make(adapter_path: str | None) -> TinkerPredictor:
        return TinkerPredictor.from_service(model_id, revision=revision, adapter_path=adapter_path)

    return make
