Commit changes often.

After making changes:
1. Run black, isort, ruff.
2. Run pyright and vibelint in tandem to determine best way to fix any issues. You can only see 20 vibelint issues at a time by default (check pyproject.toml for linting config)