# Release Runbook

## 1) Bump version
- `__init__.py`
- `pyproject.toml`
- `CHANGELOG.md`

## 2) Validate
- `python -m py_compile zimage_hq_nodes.py`

## 3) Tag and push
- `git add .`
- `git commit -m "release: vX.Y.Z"`
- `git tag vX.Y.Z`
- `git push origin main --tags`

## 4) GitHub Actions
- Tag push triggers `.github/workflows/release.yml` and uploads zip asset.
