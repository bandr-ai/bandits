# Releasing

The distribution is [`bandr`](https://pypi.org/project/bandr/) on PyPI. The
command is `mint`; Python imports are `bandits` and `mint`.

Releases are automatic. Merging a PR that changes the version publishes it;
ordinary merges publish nothing.

## How to release

In a PR:

1. Set the same new version in `bandits/__init__.py`, `mint/__init__.py` and
   `recipes/mint/pyproject.toml` (the release fails if they differ).
2. In `CHANGELOG.md`, move the `Unreleased` notes under
   `## [X.Y.Z] - YYYY-MM-DD` and add its compare link at the bottom.

Merge it. That's all.

## What happens on merge

On every push to `main`, `.github/workflows/release.yml` reads the version in
`bandits/__init__.py`:

- If tag `vX.Y.Z` already exists, it stops. Nothing is published.
- Otherwise it checks the changelog and versions, runs lint and the tests,
  builds, installs the wheel in a clean environment and runs it, publishes to
  PyPI, then creates tag `vX.Y.Z` on that commit and the GitHub release with the
  changelog section as its notes.

There is no approval step: a green build publishes. A version on PyPI cannot be
replaced or reused, so fix mistakes by releasing the next patch version.

## When something fails

- **Before publishing** (changelog, tests, build): nothing reached PyPI and no
  tag was created. Fix it in a new PR; merging it retries the same version.
- **Publishing or the GitHub release failed** (e.g. PyPI was down): re-run the
  failed jobs in Actions, or run the `release` workflow on `main` by hand.
  Files PyPI already has are skipped.

## Setup (already done)

- PyPI trusted publisher for `bandr`: owner `bandr-ai`, repository `bandits`,
  workflow `release.yml`, environment `pypi`. No PyPI token is stored anywhere.
- GitHub environment `pypi`: deploys only from `main`, no required reviewers.

## What a version number means

Before 1.0, a minor version (0.2 → 0.3) may change the command line or the
artifact format and says so in the changelog. A patch version (0.2.0 → 0.2.1)
does not.
