# Releasing

One tag publishes two distributions from the same Mint source, always at the
same version:

| PyPI project | Built from | Contains |
|---|---|---|
| [`bandr`](https://pypi.org/project/bandr/) | repository root | `mint` (command and import) and the `bandits` ingestion library |
| [`bandr-mint`](https://pypi.org/project/bandr-mint/) | `recipes/mint` | `mint` only, no runtime dependencies, for hosted integrations |

Both provide the same `mint` import and command; install one per environment.

## One-time setup

1. On PyPI, each project needs a trusted publisher (a *pending* publisher until
   its first upload creates the project): owner `bandr-ai`, repository
   `bandits`, workflow `release.yml`, environment `pypi`.
   - `bandr`: done (first published as 0.3.0).
   - `bandr-mint`: add it before the first release that includes it (0.3.1).
2. The GitHub `pypi` environment requires a reviewer and only deploys `v*`
   tags, so nothing is published without a click.

No PyPI token is stored anywhere.

## Each release

1. In a PR:
   - set the same version in `bandits/__init__.py`, `mint/__init__.py`, and
     `recipes/mint/pyproject.toml` (the release fails if they differ);
   - in `CHANGELOG.md`, move the `Unreleased` notes under
     `## [X.Y.Z] - YYYY-MM-DD` and add its compare link at the bottom.
2. After it merges, tag `main` and push the tag:

   ```bash
   git checkout main && git pull
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

3. `.github/workflows/release.yml` checks that the tag, the versions and the
   changelog agree, runs the tests, builds both distributions, installs each
   wheel in a clean environment and runs it, then waits for approval.
4. In GitHub Actions, approve the `pypi` deployment. The workflow publishes
   `bandr`, then `bandr-mint`, and creates the GitHub release with both sets of
   files and that changelog section as its notes.

If a step fails before publishing, fix it, delete the tag
(`git push --delete origin vX.Y.Z && git tag -d vX.Y.Z`) and tag again. If one
of the two uploads fails (for example a missing trusted publisher), fix the
cause and re-run the failed jobs: already-uploaded files are skipped. Once a
version is on PyPI it cannot be replaced; release the next patch version.

## What a version number means

Before 1.0, a minor version (0.2 → 0.3) may change the command line or the
artifact format and says so in the changelog. A patch version (0.2.0 → 0.2.1)
does not.

## After publication

In fresh virtual environments outside the checkout, install `bandr==X.Y.Z` and
`bandr-mint==X.Y.Z` (separately), run `mint --version`, `mint schema`, and
`mint skill install --project`, then run the [synthetic example](../examples/mint).
Run Harbor validation with a working Docker engine before claiming container
execution evidence. Check the PyPI metadata and GitHub release assets.
