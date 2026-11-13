"""Package the shared Mint source without the legacy Bandits import package."""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if self.target_name in ("wheel", "sdist"):
            local = Path(self.root) / "mint"
            source = local if local.is_dir() else Path(self.root).parents[1] / "mint"
            build_data["force_include"][str(source)] = "mint"
