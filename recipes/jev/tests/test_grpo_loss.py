"""The policy-gradient step moves sampled tokens the right way (tiny local Qwen3, CPU).

Needs torch and transformers, which the recipe's default environment does not
install for every run, so it is skipped without them.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from bandits_jev.grpo import policy_gradient_step  # noqa: E402


def tiny_model():
    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=64,
    )
    return transformers.Qwen3ForCausalLM(config)


def sampled_logprob(model, ids, mask):
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0, :-1].float()
    logprobs = torch.log_softmax(logits, -1).gather(-1, torch.tensor(ids[1:]).unsqueeze(-1)).squeeze(-1)
    return float((logprobs * torch.tensor(mask[1:], dtype=torch.float)).sum())


IDS, MASK = [1, 5, 7, 9, 11, 13], [0, 0, 1, 1, 0, 1]


@pytest.mark.parametrize("advantage", [1.0, -1.0])
def test_one_step_raises_or_lowers_the_sampled_tokens(advantage):
    model = tiny_model()
    before = sampled_logprob(model, IDS, MASK)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    out = policy_gradient_step(model, [(IDS, MASK, advantage)], normalizer=3.0, device="cpu")
    optimizer.step()
    after = sampled_logprob(model, IDS, MASK)
    assert out["trained_tokens"] == 3
    assert (after - before) * advantage > 0


def test_loss_matches_a_full_logit_computation_on_masked_tokens_only():
    model = tiny_model()
    out = policy_gradient_step(model, [(IDS, MASK, 1.0)], normalizer=1.0, device="cpu")
    assert out["mean_logprob"] == pytest.approx(sampled_logprob(model, IDS, MASK) / 3, rel=1e-4)


def test_zero_advantage_and_unmasked_items_add_no_gradient():
    model = tiny_model()
    out = policy_gradient_step(model, [(IDS, MASK, 0.0), (IDS, [0] * len(IDS), 1.0)], normalizer=1.0, device="cpu")
    assert out["trained_tokens"] == 0
    assert all(p.grad is None for p in model.parameters())
