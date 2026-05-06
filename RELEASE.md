# Cutting a release

`nmf-rs` is published to PyPI via the **PyPI Trusted Publishing** flow — no
API tokens stored in the repo. Wheels for Linux (x86_64 + aarch64), macOS
(universal2), and Windows (x86_64) plus a source dist are built by
`.github/workflows/release.yml` and uploaded to PyPI on every `v*` tag push.

## One-time setup (must do before the first release)

1. **Reserve the PyPI project name.** Sign in at https://pypi.org as a
   maintainer for the omicverse / Starlitnightly account and reserve
   `nmf-rs` (currently HTTP 404 on
   https://pypi.org/pypi/nmf-rs/json so it's available).
2. **Configure trusted publishing.** Go to
   https://pypi.org/manage/account/publishing/ and add a *pending publisher*
   with these values:
   - PyPI Project Name: `nmf-rs`
   - Owner: `omicverse`
   - Repository name: `rust-NMF`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
3. **Add the `pypi` environment in GitHub.** Repo Settings → Environments
   → "New environment" → name it `pypi`. Optionally restrict deployments
   to the `main` branch / signed tags.

After step 2, the first `release.yml` run on a tag will create the project
on PyPI and uploads work without any secrets configured.

## Cutting a release

```bash
# 1. update CHANGELOG.md and pyproject.toml version field together
$EDITOR CHANGELOG.md pyproject.toml          # bump 0.1.0 → 0.2.0

# 2. commit + tag + push
git commit -am "release 0.2.0"
git tag v0.2.0
git push origin main --tags
```

The push of the tag triggers `.github/workflows/release.yml`. It builds:

- `nmf_rs-X.Y.Z-cp39-abi3-manylinux_2_28_x86_64.whl`
- `nmf_rs-X.Y.Z-cp39-abi3-manylinux_2_28_aarch64.whl`
- `nmf_rs-X.Y.Z-cp39-abi3-macosx_10_12_universal2.whl`
- `nmf_rs-X.Y.Z-cp39-abi3-win_amd64.whl`
- `nmf_rs-X.Y.Z.tar.gz`

Then the `publish` job uploads all five to PyPI.

The wheels use the `abi3-py39` flag (one wheel works for any Python ≥ 3.9 on
that platform), so we don't need a per-Python matrix.

## Sanity check before tagging

```bash
# Ensure local build still works and tests pass
maturin develop --release --manifest-path rust/Cargo.toml
pytest tests/ -v
cargo test --release --manifest-path rust/Cargo.toml --lib

# Build a candidate wheel locally and inspect it
maturin build --release --manifest-path rust/Cargo.toml -o dist
ls dist/
```

## Re-publishing a botched release

PyPI does not allow re-uploading a deleted version. If `v0.2.0` is broken,
yank it on PyPI and release `v0.2.1`. Don't try to recreate the same tag.
