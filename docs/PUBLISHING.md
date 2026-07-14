# Publishing pioneer-detection to PyPI

One-time setup:

1. Create an account at https://pypi.org and enable 2FA.
2. Create an API token: Account settings → API tokens → "Add API token" (scope: entire account for the first upload; switch to project-scoped after).

Publish a release:

```bash
cd ~/pioneer-detection-method
python -m pip install --upgrade build twine
rm -rf dist/
python -m build          # creates dist/*.tar.gz and dist/*.whl
python -m twine upload dist/*
# username: __token__
# password: <your PyPI API token>
```

Verify with `pip install pioneer-detection` in a fresh environment.

For each new version: bump `version` in `pyproject.toml` and `__version__` in
`pioneer_detection/__init__.py`, update `CITATION.cff`, tag the release
(`git tag v1.x.y && git push --tags`), then rebuild and upload.

After the first upload, add the PyPI badge to README.md:

```markdown
[![PyPI](https://img.shields.io/pypi/v/pioneer-detection)](https://pypi.org/project/pioneer-detection/)
```
