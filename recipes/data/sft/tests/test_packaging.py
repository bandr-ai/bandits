"""Runtime resources ship inside the package, so a wheel install works outside the source checkout."""
import importlib.resources as ir

from bandits_sft import pipeline, rubric_loop
from bandits_sft.config import load_config


def test_resources_load_from_the_package():
    pkg = ir.files("bandits_sft")
    assert pkg.joinpath("defaults", "pipeline_fireworks.json").is_file()
    for name in ("generic_parts_v2.md", "induce_rubric_prompt_v2.md", "selftest_prompts.md"):
        assert pkg.joinpath("rubrics", name).is_file(), name
    assert str(pipeline.DEFAULT_CONFIG).startswith(str(pkg))
    assert load_config(pipeline.DEFAULT_CONFIG).features.rubric_judge
    assert "{consistency_map}" in rubric_loop.generic_v2()
    assert rubric_loop.prompt_section("INDUCE", "induce_rubric_prompt_v2.md")


def test_scenario_recipe_exports():
    from bandits_sft import scenario_pipeline as sp
    assert sp.SCENARIO_STEPS == ("ingest", "judge_setup", "contracts", "coverage", "scenarios")
    assert issubclass(sp.ScenarioPipeline, pipeline.Pipeline) and callable(sp.main)
