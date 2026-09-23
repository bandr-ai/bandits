"""LoRA SFT over the same forward pass and letter readout as the frozen
scorer. Cross-entropy over the option-letter logits at the answer position,
never a full-vocabulary language-modeling loss.

One of two modules in ``bandits.decide`` allowed to import torch/transformers
(the other is ``hf_predictor.py``), and only inside function bodies.
Post-training scoring reuses ``HFPredictor`` with the saved adapter attached
(see ``HFPredictor.__init__``'s ``adapter_path``), not a separate readout
implementation -- so a checkpoint's training-time and scoring-time
probabilities can never drift apart by construction.
"""

from __future__ import annotations

from bandits.decide.dataset import DecisionExample
from bandits.decide.hf_predictor import letter_token_id
from bandits.decide.prompt import build_prompt


class HFTrainer:
    """LoRA SFT over a pinned base model. Only the LoRA weights update; the
    base model stays frozen throughout (``requires_grad=False`` on every
    base parameter, enforced by ``peft.get_peft_model``)."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        learning_rate: float = 5e-5,
    ) -> None:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

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
        self._torch = torch

    def compute_loss(self, example: DecisionExample, *, option_order: dict[str, str] | None = None):
        """Cross-entropy over this example's option-letter logits at the
        answer position, target = the correct option's letter. ``option_order``
        overrides ``example.options``' own order (how the training-time
        option shuffle is applied -- see ``trainer.shuffled_options``); the
        target letter is remapped to match whatever order was used, never
        assumed to stay at a fixed position.

        Soft targets use the same cross-entropy formula against the target
        distribution instead of a one-hot vector; not exercised by the
        launch path (see the module docstring), but not a separate code
        path either.
        """
        torch = self._torch
        options = option_order or dict(example.options)
        prompt, letters = build_prompt(example.state, example.question, options)
        model_label = f"{self.model_id}@{self.revision}"

        letter_token_ids = {
            letter: letter_token_id(self._tokenizer, prompt, letter, model_label=model_label)
            for letter in letters.values()
        }

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self._model(**inputs)
        logits = outputs.logits[0, -1, :]

        candidate_ids = [letter_token_ids[letters[option_id]] for option_id in options]
        candidate_logits = logits[candidate_ids]

        target_probs = torch.tensor(
            [example.target.probabilities.get(option_id, 0.0) for option_id in options],
            dtype=candidate_logits.dtype,
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
        return total_loss / len(examples)

    def save_adapter(self, path: str) -> None:
        self._model.save_pretrained(path)

    def load_adapter(self, path: str) -> None:
        """Load a previously saved adapter's weights into this trainer's
        live model, e.g. when resuming from a checkpoint mid-run."""
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        state_dict = load_file(f"{path}/adapter_model.safetensors")
        set_peft_model_state_dict(self._model, state_dict)

    def optimizer_state_dict(self) -> dict:
        return self._optimizer.state_dict()

    def load_optimizer_state_dict(self, state: dict) -> None:
        self._optimizer.load_state_dict(state)
