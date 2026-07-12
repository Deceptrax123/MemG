# Contributing

Thanks for contributing to GeoLogic Memory.

1. Create a virtual environment and install development dependencies with `python -m pip install -e '.[dev]'`.
2. Add tests for behavior changes.
3. Run `python -m pytest`, `python -m compileall -q geologic_memory mcp`, `python -m build`, and `python -m twine check dist/*`.
4. Open a focused pull request that explains the change and any checkpoint or API compatibility impact.

Please do not commit API keys, checkpoints, generated caches, private dialogue, or saved `.pkl` sessions. By contributing, you agree that your contribution is licensed under the MIT License.
