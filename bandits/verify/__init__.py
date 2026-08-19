"""Verifier synthesis, grading and anti-cheat. The reward function lives here."""

from bandits.verify.anticheat import (
    AntiCheatReport,
    Finding,
    RolloutAction,
    RolloutRecord,
    check_rollout,
    enforce,
)
from bandits.verify.run import RewardMode, UnreviewedVerifierError, evaluate
from bandits.verify.synthesize import UnlabeledTraceError, synthesize_verifier

__all__ = [
    "AntiCheatReport",
    "Finding",
    "RewardMode",
    "RolloutAction",
    "RolloutRecord",
    "UnlabeledTraceError",
    "UnreviewedVerifierError",
    "check_rollout",
    "enforce",
    "evaluate",
    "synthesize_verifier",
]
