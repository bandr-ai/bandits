"""LoRA SFT over the same forward pass and letter readout as the frozen
scorer. Cross-entropy over the option-letter logits at the answer position,
never a full-vocabulary language-modeling loss.

One of two modules in ``bandits_jev`` allowed to import torch/transformers
(the other is ``hf_predictor.py``), and only inside function bodies. The
loss reads letters through the same ``encode_prompt``/``letter_token_id`` as
``HFPredictor``, and dev scoring goes through ``HFPredictor`` itself
(``predictor``), so a checkpoint's training-time and scoring-time
probabilities cannot drift apart.
"""

from __future__ import annotations

import platform
from importlib.metadata import version

from bandits_jev.dataset import DecisionExample
from bandits_jev.hf_predictor import HFPredictor, encode_prompt, letter_token_ids
from bandits_jev.prompt import build_prompt

_TRAINING_STATE_FILE = "training_state.pt"


class HFTrainer:
    """LoRA SFT over a pinned base model. Only the LoRA weights update; the
    base model stays frozen throughout (``requires_grad=False`` on every
    base parameter, enforced by ``peft.get_peft_model``)."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str,
        seed: int,
        device: str = "cuda",
        dtype: str = "bfloat16",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        learning_rate: float = 5e-5,
    ) -> None:
        """``seed`` seeds every RNG (python, numpy, torch, CUDA) before the
        LoRA weights are initialized, so the adapter's initial weights and
        dropout masks are reproducible -- not only the data order."""
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.dtype = dtype
        self.lora_config = {
            "rank": lora_rank,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "target_modules": "all-linear",
        }
        self.learning_rate = learning_rate

        set_seed(seed)
        torch_dtype = getattr(torch, dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        base_model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch_dtype)
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        self._model = get_peft_model(base_model, peft_config).to(device)
        self._optimizer = torch.optim.AdamW(
            (p for p in self._model.parameters() if p.requires_grad), lr=learning_rate
        )
        self._scheduler = None
        self._torch = torch

    @classmethod
    def from_config(cls, config) -> HFTrainer:
        """Build from a ``TrainingRunConfig`` so the recorded config and the
        trainer actually used can never disagree."""
        return cls(
            config.base_model_id,
            revision=config.base_revision,
            seed=config.seed,
            device=config.device,
            dtype=config.dtype,
            lora_rank=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            learning_rate=config.learning_rate,
        )

    def configure_schedule(self, *, total_steps: int, warmup_steps: int) -> None:
        """Linear warmup to the peak LR over ``warmup_steps``, then linear
        decay to zero at ``total_steps``. Must be called before the first
        ``train_step`` and before ``load_checkpoint`` (which restores the
        scheduler's position)."""
        from transformers import get_linear_schedule_with_warmup

        self._scheduler = get_linear_schedule_with_warmup(
            self._optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

    def token_count(self, prompt: str) -> int:
        """Same count ``HFPredictor.token_count`` uses for its overlength check."""
        return len(self._tokenizer.encode(prompt))

    def check_letters(self, prompt: str, letters) -> None:
        """Raise ``TokenizationError`` if the scorer would reject this prompt's
        letters, so training never learns from a row dev scoring can't read."""
        letter_token_ids(self._tokenizer, prompt, letters, model_label=f"{self.model_id}@{self.revision}")

    def compute_loss(self, example: DecisionExample, *, option_order: dict[str, str] | None = None):
        """Cross-entropy over this example's option-letter logits at the
        answer position, target = the correct option's letter. ``option_order``
        overrides ``example.options``' own order (how the training-time
        option shuffle is applied -- see ``trainer.shuffled_options``); the
        target letter is remapped to match whatever order was used, never
        assumed to stay at a fixed position.

        Soft targets use the same cross-entropy formula against the target
        distribution instead of a one-hot vector; not exercised by the
        launch path, but not a separate code path either. Computed in
        float32, like the scorer's softmax, whatever the model dtype.
        """
        torch = self._torch
        options = option_order or dict(example.options)
        prompt, letters = build_prompt(example.state, example.question, options)
        model_label = f"{self.model_id}@{self.revision}"

        letter_ids = letter_token_ids(self._tokenizer, prompt, letters.values(), model_label=model_label)

        inputs = encode_prompt(self._tokenizer, prompt, self.device, model_label=model_label)
        outputs = self._model(**inputs)
        logits = outputs.logits[0, -1, :].float()

        candidate_ids = [letter_ids[letters[option_id]] for option_id in options]
        candidate_logits = logits[candidate_ids]

        target_probs = torch.tensor(
            [example.target.probabilities.get(option_id, 0.0) for option_id in options],
            dtype=torch.float32,
            device=candidate_logits.device,
        )
        log_probs = torch.log_softmax(candidate_logits, dim=-1)
        loss = -(target_probs * log_probs).sum()
        return loss

    def train_step(self, examples: list[DecisionExample], *, option_orders: list[dict[str, str]] | None = None) -> float:
        """One optimizer step over a batch of examples (gradient-accumulated
        by summing each example's loss before a single ``backward()`` --
        this is what "effective batch size" means here, matching how a
        larger accumulation window would behave without a real batch
        dimension). Returns the mean per-example loss as a plain float."""
        self._model.train()
        self._optimizer.zero_grad()
        total_loss = 0.0
        orders = option_orders or [dict(e.options) for e in examples]
        for example, options in zip(examples, orders, strict=True):
            loss = self.compute_loss(example, option_order=options)
            (loss / len(examples)).backward()
            total_loss += float(loss.item())
        self._optimizer.step()
        if self._scheduler is not None:
            self._scheduler.step()
        return total_loss / len(examples)

    def predictor(self, adapter_path: str | None = None) -> HFPredictor:
        """An ``HFPredictor`` over this trainer's live model, for dev scoring
        without loading a second copy of the base weights."""
        return HFPredictor.from_loaded(
            self._model,
            self._tokenizer,
            model_id=self.model_id,
            revision=self.revision,
            device=self.device,
            dtype=self.dtype,
            adapter_path=adapter_path,
        )

    def save_adapter(self, path: str) -> None:
        self._model.save_pretrained(path)

    def save_checkpoint(self, path: str) -> None:
        """Adapter weights plus everything a resume needs to continue as if
        uninterrupted: optimizer and scheduler state and the RNG states that
        drive dropout."""
        torch = self._torch
        self.save_adapter(path)
        state = {
            "optimizer": self._optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict() if self._scheduler is not None else None,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(state, f"{path}/{_TRAINING_STATE_FILE}")

    def load_checkpoint(self, path: str) -> None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        torch = self._torch
        set_peft_model_state_dict(self._model, load_file(f"{path}/adapter_model.safetensors"))
        state = torch.load(f"{path}/{_TRAINING_STATE_FILE}", weights_only=True)
        self._optimizer.load_state_dict(state["optimizer"])
        if state["scheduler"] is not None:
            if self._scheduler is None:
                raise ValueError("checkpoint has scheduler state; call configure_schedule before load_checkpoint")
            self._scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def environment(self) -> dict[str, str]:
        torch = self._torch
        env = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "peft": version("peft"),
            "device": self.device,
        }
        if torch.cuda.is_available():
            env["cuda"] = str(torch.version.cuda)
            env["gpu"] = torch.cuda.get_device_name(0)
        return env
