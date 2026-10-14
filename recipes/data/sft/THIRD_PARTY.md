# Third-party attribution

This recipe incorporates code and adapted prompts from [Microsoft Simia-Agent-Training](https://github.com/microsoft/Simia-Agent-Training/tree/ea77d9d13362cc56cd1e0f33dc4cea4508aea28f), based on [Simia (Li et al.)](https://arxiv.org/abs/2511.01824).

- Upstream revision: `ea77d9d13362cc56cd1e0f33dc4cea4508aea28f`.
- Copyright (c) 2025 Microsoft. MIT license: [LICENSE-UPSTREAM](LICENSE-UPSTREAM).
- Vendored parsing, repair, filtering and reference-conversation helpers: `bandits_sft/upstream.py` (the source header records original paths and adaptations).
- Adapted generation and environment prompts: `bandits_sft/prompts.py`; domain-specific blocks are replaced with generic ones and optional feature blocks are added.

The upstream license is also shipped with the installed wheel.
