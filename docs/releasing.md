# Releasing

Bandits is published to PyPI as [`bandits-cli`](https://pypi.org/project/bandits-cli/).
The command and the import are both `bandits`.

## One-time setup

1. On PyPI, add a trusted publisher for `bandits-cli` (as a *pending* publisher
   until the first upload creates the project):
   owner `bandr-ai`, repository `bandits`, workflow `release.yml`,
   environment `pypi`.
2. In this repo's settings, create the `pypi` environment. Requiring a
   reviewer there means nothing is published without a click.

No PyPI token is stored anywhere.

## Each release

1. In a PR:
   - set `__version__` in `bandits/__init__.py`;
   - in `CHANGELOG.md`, move the `Unreleased` notes under
     `## [X.Y.Z] - YYYY-MM-DD` and add its compare link at the bottom.
2. After it merges, tag `main` and push the tag:

   ```bash
   git checkout main && git pull
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

3. `.github/workflows/release.yml` checks that the tag, the version and the
   changelog agree, runs the tests, builds, installs the wheel in a clean
   environment and runs it, publishes to PyPI, and creates the GitHub release
   with that changelog section as its notes.

If a step fails before publishing, fix it, delete the tag
(`git push --delete origin vX.Y.Z && git tag -d vX.Y.Z`) and tag again. Once a
version is on PyPI it cannot be replaced; release the next patch version.

## What a version number means

Before 1.0, a minor version (0.2 → 0.3) may change the command line or the
artifact format and says so in the changelog. A patch version (0.2.0 → 0.2.1)
does not.
