Commit changes often.

After making changes:
1. Run black, isort, ruff.
2. Run pyright and vibelint in tandem to determine best way to fix any issues. You can only see 20 vibelint issues at a time by default (check pyproject.toml for linting config)
3. Run tests and assess test quality using the procedures below.

## Testing Procedures

### Standard Testing Pipeline
1. **Run existing tests**: `tox -e py311` (or appropriate Python version)
2. **Check coverage**: Tests should maintain high pass rate and reasonable coverage
3. **Run agentic test assessment**: `vibelint coverage-vibe src/ tests/ --assess-tests`

### Agentic Test Quality Assessment
When implementing new features or fixing bugs:

1. **Self-Validating Assessment**:
   ```bash
   vibelint coverage-vibe src/vibelint/ tests/ --assess-tests --max-suggestions 10
   ```

2. **Review AI Assessment Results**:
   - Input validation scores (target: >0.8)
   - Output validation scores (target: >0.7)
   - Behavioral issues identified
   - Missing requirements discovered

3. **Address High-Priority Issues**:
   - Fix tests with confidence scores <0.6
   - Add tests for uncovered requirements
   - Implement AI-suggested improvements

4. **Validate Changes**:
   - Re-run assessment after improvements
   - Ensure scores improve and issues decrease
   - Verify requirements coverage increases

### Test Development Guidelines
- Write tests that validate **behavior**, not just coverage
- Include edge cases and boundary conditions
- Use descriptive test names: `test_function_should_handle_empty_input`
- Validate both expected outputs AND side effects
- Consider requirements compliance from the start

### Continuous Quality Improvement
- Use AI suggestions to identify blind spots in testing
- Cross-validate test logic using dual LLM assessment
- Regularly assess test suite quality, not just code coverage
- Address systematically uncovered requirements across the codebase

The goal is **self-improving test quality** where AI continuously assesses and enhances our testing approach.