# vibelint

This project uses kaia-guardrails hooks for development workflow automation.

## Shell Configuration

Claude Code has limitations with shell configuration loading. To work around this:

1. **For commands needing shell environment**:
   ```bash
   source .claude/shell-helper.sh && your_command
   ```

2. **For one-off commands with shell config**:
   ```bash
   zsh -c 'source ~/.zshrc && your_command'
   ```

3. **Always use absolute paths** when possible to avoid PATH issues

4. **Quote paths with spaces** to prevent command parsing errors

5. **Set environment variables** in `.claude/settings.json` rather than shell config

See `.claude/shell-config.md` for detailed troubleshooting.

