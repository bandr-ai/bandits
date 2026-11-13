"""Package the shared Mint source without the legacy Bandits import package."""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if self.target_name in ("wheel", "sdist"):
            local = Path(self.root) / "mint"
            source = local if local.is_dir() else Path(self.root).parents[1] / "mint"
            # Explicit package resources only: force-including a directory also ships ignored
            # credentials, databases and bytecode from a developer checkout.
            for path in sorted(source.rglob("*")):
                rel = path.relative_to(source)
                if (
                    path.is_file()
                    and not path.is_symlink()
                    and not any(part.startswith(".") or part == "__pycache__" for part in rel.parts)
                    and path.suffix in {".py", ".md", ".json"}
                ):
                    build_data["force_include"][str(path)] = "mint/" + rel.as_posix()
