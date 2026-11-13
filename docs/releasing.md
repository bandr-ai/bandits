# Releasing

The distribution is named `bandr`. The first planned PyPI release is **0.3.0**;
it has not been published yet.
The command is `mint`. Python imports remain
`bandits` and `mint`.

The first Bandr release requires a new trusted publisher for `bandr`; the old
`bandits-cli` publisher does not authorize publishing this distribution.

## One-time setup

1. On PyPI, add a trusted publisher for `bandr` (as a *pending* publisher
   until the first upload creates the project):
   owner `bandr-ai`, repository `bandits`, workflow `release.yml`,
   environment `pypi`.
2. In this repo's settings, create the `pypi` environment. Requiring a
   reviewer there means nothing is published without a click.

No PyPI token is stored anywhere.

Before tagging 0.3.0, confirm the pending publisher exists on PyPI and that the
GitHub `pypi` environment has a required reviewer. Laxman owns the PyPI step.
Do not push the release tag until both are ready. A pending publisher configures
a future publication; it does not guarantee a package name is reserved.

## Each release

1. In a PR:
   - set matching versions in `bandits/__init__.py`, `mint/__init__.py`, and
     `recipes/mint/pyproject.toml`;
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

## After publication

In a fresh virtual environment outside the checkout, install `bandr==0.3.0`, run
`mint --version`, `mint schema`, and `mint skill install --project`, then run the
[synthetic example](../examples/mint). Run Harbor validation with a working Docker
engine before claiming container execution evidence. Check the PyPI metadata and
GitHub release assets. Remove the README's release-preparation notice only after
the install from PyPI succeeds.
