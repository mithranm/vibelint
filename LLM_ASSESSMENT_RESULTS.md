# LLM Assessment Results

**Date:** 2025-09-17 10:57:02
**Assessment:** Configuration vs Actual Performance

## Fast LLM Assessment

### Current Configuration:
- **Model:** openai/gpt-oss-20b
- **API:** http://100.94.250.88:8001
- **Configured Max Tokens:** 2,048
- **Configured Context:** 0

### Discovered Performance:
- **Engine Type:** openai_compatible
- **Actual Max Context:** 1,000 tokens
- **Effective Context:** 900 tokens
- **Success Rate:** 75.0%
- **Avg Latency:** 569ms

### Assessment:
💡 **INFO:** No context limit configured, discovered 900 tokens
⚠️  **WARNING:** Low success rate (75.0%) - LLM may be overloaded
✅ **OK:** Acceptable latency (569ms)

## Orchestrator LLM Assessment

### Current Configuration:
- **Model:** C:\dev\openai_gpt-oss-120b-MXFP4.gguf
- **API:** http://100.116.54.128:11434
- **Configured Max Tokens:** 8,192
- **Configured Context:** 28,800

### Discovered Performance:
- **Engine Type:** openai_compatible
- **Actual Max Context:** 32,000 tokens
- **Effective Context:** 28,800 tokens
- **Success Rate:** 87.5%
- **Avg Latency:** 8160ms

### Assessment:
✅ **OK:** Configured context within actual capacity
⚠️  **WARNING:** Low success rate (87.5%) - LLM may be overloaded
✅ **OK:** Acceptable latency (8160ms)

## 🔧 Recommended Actions

Based on the assessment, consider updating your `pyproject.toml`:

```toml
[tool.vibelint.llm]
# Recommended limits for fast LLM based on testing
fast_max_context_tokens = 900
fast_max_prompt_tokens = 800
```

### Next Steps

1. Review any warnings above
2. Update configuration if recommended
3. Re-run diagnostics after changes to verify
4. Monitor performance in production use
