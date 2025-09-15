# Print Statement Suppression in vibelint

vibelint provides sophisticated detection of print statements that should be replaced with proper logging. However, it also recognizes that some print statements are legitimate for CLI tools and stdout communication.

## Suppression Methods

### 1. Inline Comment Suppression

You can suppress print statement warnings using inline comments:

```python
# Explicit stdout communication marker
print("Server running on port 8080")  # vibelint: stdout

# General vibelint suppression
print("Debug output")  # vibelint: ignore

# Specific print suppression (compatible with other linters)
print("Status update")  # noqa: print

# General noqa suppression
print("Progress: 50%")  # noqa

# Type ignore (also suppresses print warnings)
print("Complex output")  # type: ignore
```

### 2. Auto-Detection of CLI Patterns

vibelint automatically recognizes legitimate CLI output patterns:

```python
# URLs are auto-detected as CLI output
print("Server at http://localhost:8000")

# Port numbers indicate server/network communication
print("Listening on port 5556")

# Status indicators are recognized
print("Starting calibration...")
print("SUCCESS: Operation completed")

# Setup/configuration messages
print("Configuring audio device...")
```

### 3. File-Level Exclusions

Configure files to exclude from print statement validation:

```toml
[tool.vibelint.print_validation]
exclude_globs = [
    "cli.py",
    "main.py",
    "__main__.py",
    "*_cli.py",
    "*_cmd.py",
    "scripts/**/*.py",
    "tools/**/*.py"
]
```

## Best Practices

### When to Use Print Statements

✅ **Legitimate uses:**
- CLI tool output and user interaction
- Server startup messages
- Configuration/setup instructions
- Progress indicators for long-running operations
- URLs and connection information

### When to Use Logging

❌ **Replace prints with logging for:**
- Debug information
- Error messages (use logger.error)
- Internal state tracking
- Production code diagnostics
- Library code (never print in libraries)

## Examples

### CLI Tool Example
```python
def main():
    """Main CLI entry point."""
    print("Welcome to MyTool v1.0")  # vibelint: stdout
    print(f"Connecting to {server_url}...")  # Auto-detected (URL)

    if debug_mode:
        logger.debug("Debug mode enabled")  # Use logging for debug

    print("Ready for commands. Type 'help' for usage.")  # vibelint: stdout
```

### Server Application Example
```python
def start_server(port: int):
    """Start the web server."""
    # These are auto-detected as legitimate
    print(f"Starting server on port {port}")
    print(f"Server running at http://localhost:{port}")
    print("Press Ctrl+C to stop")

    # Use logging for internal diagnostics
    logger.info(f"Server initialized with config: {config}")
    logger.debug(f"Request handlers registered: {handlers}")
```

### Calibration/Setup Example
```python
def run_calibration():
    """Run device calibration."""
    print("=== Device Calibration ===")  # vibelint: stdout
    print("Step 1: Connect your device")  # vibelint: stdout

    # Progress indicators are auto-detected
    print("Calibrating... 0%")
    print("Calibrating... 50%")
    print("Calibrating... 100%")

    print("SUCCESS: Calibration complete!")  # Auto-detected (status)
```

## Integration with CI/CD

To enforce print statement rules in CI while allowing legitimate uses:

```yaml
# .github/workflows/lint.yml
- name: Run vibelint
  run: |
    # Check for print statements
    vibelint check --categories core

    # Or check specific rule
    vibelint check --rule PRINT-STATEMENT
```

## Migration Strategy

When migrating from prints to logging:

1. **Identify all print statements:**
   ```bash
   vibelint check --rule PRINT-STATEMENT
   ```

2. **Add suppression comments for legitimate prints:**
   ```python
   print("Starting server...")  # vibelint: stdout
   ```

3. **Replace debug prints with logging:**
   ```python
   # Before
   print(f"Processing {item}")

   # After
   logger.debug(f"Processing {item}")
   ```

4. **Verify changes:**
   ```bash
   vibelint check --categories core
   ```

## Troubleshooting

If legitimate prints are being flagged:

1. Check if the content matches CLI patterns (URLs, ports, status)
2. Add explicit suppression: `# vibelint: stdout`
3. Consider file-level exclusion for CLI scripts
4. Review auto-detection patterns in your use case

For questions or to report issues with print detection, please open an issue on the vibelint repository.