# Extending Vibelint with Third-Party Validators

Vibelint supports third-party validators through Python entry points. This allows you to create custom validators in separate packages and register them with vibelint.

## Creating a Third-Party Validator

### 1. Create Your Validator Class

Your validator must inherit from `vibelint.plugin_system.BaseValidator`:

```python
from pathlib import Path
from typing import Iterator
from vibelint.plugin_system import BaseValidator, Finding, Severity

class MyCustomValidator(BaseValidator):
    """Custom validator for my specific coding standards."""

    rule_id = "MY-CUSTOM-RULE"
    name = "My Custom Rule"
    description = "Validates my specific coding standards"
    default_severity = Severity.WARN

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Validate a file and yield findings."""
        # Your validation logic here
        lines = content.splitlines()
        for line_num, line in enumerate(lines, 1):
            if "my_bad_pattern" in line:
                yield self.create_finding(
                    message="Found bad pattern in code",
                    file_path=file_path,
                    line=line_num,
                    suggestion="Replace with better pattern"
                )
```

### 2. Single-File vs Project-Wide Validators

Choose the appropriate base class based on your validator's needs:

#### Single-File Validators
- Analyze individual files in isolation
- Inherit from `BaseValidator`
- Implement `validate(file_path, content, config)`
- Should not access other files or require project context

#### Project-Wide Validators
- Analyze entire projects with knowledge of all files
- Inherit from `BaseValidator` but implement `validate_project()`
- Should also implement `requires_project_context() -> bool` returning `True`

```python
from typing import Dict
from vibelint.plugin_system import BaseValidator, Finding

class MyProjectWideValidator(BaseValidator):
    """Validator that needs to see the whole project."""

    def validate_project(self, project_files: Dict[Path, str], config=None) -> Iterator[Finding]:
        """Validate entire project with knowledge of all files."""
        # Your project-wide validation logic here
        pass

    def requires_project_context(self) -> bool:
        """This validator requires full project context."""
        return True

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Single-file validate should not be used for project-wide validators."""
        raise NotImplementedError(
            f"{self.__class__.__name__} is a project-wide validator. "
            "Use validate_project() instead of validate()."
        )
```

### 3. Package Structure

Organize your package like this:

```
my_vibelint_validators/
├── pyproject.toml
├── src/
│   └── my_vibelint_validators/
│       ├── __init__.py
│       └── validators.py
```

### 4. Register via Entry Points

In your `pyproject.toml`, register your validators:

```toml
[project.entry-points."vibelint.validators"]
"MY-CUSTOM-RULE" = "my_vibelint_validators.validators:MyCustomValidator"
"MY-PROJECT-RULE" = "my_vibelint_validators.validators:MyProjectWideValidator"
```

### 5. Installation and Usage

Users install your package alongside vibelint:

```bash
pip install vibelint my-vibelint-validators
```

Your validators are automatically discovered and available:

```bash
vibelint check --rule MY-CUSTOM-RULE src/
```

## Best Practices

### Rule ID Conventions
- Use UPPERCASE with hyphens: `MY-CUSTOM-RULE`
- Be descriptive but concise
- Avoid conflicts with built-in rules

### Performance Considerations
- Single-file validators should be fast since they run on every file
- Project-wide validators should cache expensive operations
- Use lazy loading for heavy dependencies

### Configuration Support
Validators receive a `config` parameter that contains user configuration:

```python
def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
    # Access user configuration
    my_config = config.get("my_validator", {}) if config else {}
    threshold = my_config.get("threshold", 10)
    # ... validation logic
```

Users can configure your validator in their `pyproject.toml`:

```toml
[tool.vibelint.my_validator]
threshold = 5
custom_patterns = ["pattern1", "pattern2"]
```

### Error Handling
- Handle syntax errors gracefully (invalid Python files)
- Log warnings for configuration issues
- Never raise exceptions that crash vibelint

### Testing
Test your validators thoroughly:

```python
def test_my_validator():
    validator = MyCustomValidator()
    content = "my_bad_pattern in code"
    findings = list(validator.validate(Path("test.py"), content))
    assert len(findings) == 1
    assert findings[0].rule_id == "MY-CUSTOM-RULE"
```

## Built-in Validator Examples

See the built-in validators for examples:
- Single-file: `vibelint.validators.single_file.emoji`
- Project-wide: `vibelint.validators.project_wide.dead_code`
- Architecture: `vibelint.validators.architecture.semantic_similarity`

## Auto-Fix Support (Optional)

Validators can support automatic fixes:

```python
class MyFixableValidator(BaseValidator):
    def can_fix(self, finding: Finding) -> bool:
        """Check if this finding can be automatically fixed."""
        return finding.rule_id == self.rule_id

    def apply_fix(self, content: str, finding: Finding) -> str:
        """Apply automatic fix to content."""
        lines = content.splitlines(True)
        if finding.line <= len(lines):
            lines[finding.line - 1] = lines[finding.line - 1].replace("old", "new")
        return "".join(lines)
```

## Distribution

Publish your validators to PyPI so others can easily install them:

```bash
pip install build twine
python -m build
twine upload dist/*
```

This makes vibelint extensible while maintaining a clean plugin architecture!