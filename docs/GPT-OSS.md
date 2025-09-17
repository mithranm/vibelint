### What is the Harmony Format?

The Harmony Response Format is a structured output and prompting system introduced by OpenAI in August 2025 for their open-weight GPT models, specifically the `gpt-oss` series (e.g., gpt-oss-20b and gpt-oss-120b). It standardizes how models handle conversations, reasoning (chain-of-thought or CoT), tool calls, and final responses by using a multi-channel approach with special tokens for machine-parsable transcripts. This format emphasizes interoperability ("Harmony = interoperability") across systems, making it easier to separate internal model thinking from user-facing output.

#### Key Features and Structure
- **Roles**: Messages are tagged with roles like `system` (for meta info like reasoning level or built-in tools), `developer` (for system prompts and function definitions), `user` (input), `assistant` (output), and `tool` (tool results). Roles follow a hierarchy for priority.
- **Channels**: Assistant outputs are directed to specific channels:
  - `final`: User-facing response (e.g., the clean answer).
  - `analysis`: Internal CoT reasoning, not meant for users due to potential unfiltered content.
  - `commentary`: For tool calls, preambles, or action plans (e.g., multiple tool invocations).
- **Special Tokens**: Uses delimiters like `<|start|>`, `<|message|>`, `<|end|>`, `<|channel|>`, `<|constrain|>`, `<|call|>`, and `<|return|>` to structure content. These are part of the `o200k_harmony` tokenizer.
- **Reasoning Handling**: Supports adjustable reasoning efforts (`low`, `medium`, `high`). CoT goes to the `analysis` channel, while the polished answer is in `final`. For example, for "What is 2 + 2?": Analysis might say "Simple arithmetic: add the numbers," and Final: "4".
- **Tool Calls and Structured Outputs**: Tools are defined in TypeScript-like syntax in the `developer` role. Calls use `commentary` channel with constraints (e.g., `<|constrain|>json`). Structured responses use JSON schemas in prompts for enforced formats.
- **Purpose**: Separates "thinking" (analysis) from output (final) to avoid exposing raw, potentially unsafe reasoning. It's required for `gpt-oss` models to work correctly; without it, outputs may be garbled or incorrect.

Example of a basic Harmony-formatted prompt/response (simplified):
```
<|start|>system<|message|>Reasoning: medium<|end|>
<|start|>developer<|message|>You are a helpful assistant.<|end|>
<|start|>user<|message|>What is 2 + 2?<|end|>
<|start|>assistant<|channel|>analysis<|message|>Calculate: 2 plus 2 equals 4.<|end|>
<|start|>assistant<|channel|>final<|message|>4<|end|>
```
This ensures the model "thinks" internally but only shows "4" to the user.

OpenAI provides a standalone library for parsing/tokenizing Harmony.

### Does llama.cpp's `response_format` Work with Harmony Format?

Yes, but with some nuances—llama.cpp supports running `gpt-oss` models (which rely on Harmony) and can integrate with Harmony's structured features, though `response_format` (an OpenAI API parameter for constraining outputs like to JSON) interacts indirectly rather than being a direct "Harmony switch."

#### Compatibility Details
- **Native Support for gpt-oss and Harmony**: llama.cpp fully supports inferencing `gpt-oss` models in formats like GGUF and MXFP4 across CPU, GPU (CUDA, Vulkan, Metal), with guides for optimal local setup. You must use Harmony-style prompts (with special tokens) for correct behavior; standard chat templates may fail. llama.cpp handles Harmony tags during generation, applying grammars before processing them.
- **Controlling Reasoning/Channels**: Use flags like `--reasoning-format none` to suppress or strip `analysis` channel tags, effectively hiding "thinking" tokens and outputting only the `final` channel. This aligns with your earlier query on excluding thinking tokens—set it to `none` for concise outputs.
- **Integration with `response_format`**: The `llama-server` (OpenAI-compatible endpoint) supports `response_format` (e.g., `{"type": "json_object"}` or `{"type": "json_schema"}`) via grammars, which can constrain Harmony's `final` channel to structured JSON. Since Harmony natively supports JSON schemas in the `developer` role for structured outputs, you can combine them: Prompt in Harmony style, then use `response_format` to enforce schemas on the response. This works well for tool calls or JSON-only finals.
- **Limitations and Workarounds**:
  - Fine-tuned `gpt-oss` models may have output issues if Harmony isn't properly handled.
  - For pre-filling or custom generations, manually include Harmony tokens (e.g., in prompts or prefills).
  - Some tools like LM Studio or Open WebUI add extra Harmony parsing layers. In pure llama.cpp, post-process outputs if needed (e.g., parse channels in your client code).
  - Reasoning effort params (low/medium/high) may require specific quants or commits; check for bugs in evals.

#### Setup Example for llama.cpp with Harmony
1. Download a `gpt-oss` GGUF model (e.g., from Hugging Face).
2. Run the server: `./llama-server -m gpt-oss-20b.gguf --port 8080 --reasoning-format none` (to exclude thinking tokens).
3. Query with OpenAI client, using Harmony in messages if needed:
   ```python
   from openai import OpenAI
   client = OpenAI(base_url="http://localhost:8080/v1", api_key="no-key")
   response = client.chat.completions.create(
       model="gpt-oss-20b.gguf",
       messages=[
           {"role": "system", "content": "<|start|>system<|message|>Reasoning: medium<|end|>"},  # Harmony example
           {"role": "user", "content": "What is 2 + 2?"}
       ],
       response_format={"type": "json_object"}  # Constrains to JSON
   )
   print(response.choices[0].message.content)  # e.g., {"answer": "4"}
   ```
This should output only the final JSON without thinking steps, leveraging both Harmony channels and `response_format`.

If you're using a non-gpt-oss model, Harmony isn't required but can be emulated via custom prompts/grammars for similar multi-channel effects. For the latest, check llama.cpp's GitHub for updates post-August 2025. If issues arise, specify your setup for more tweaks!

# ORIGINAL TEXT:

The [gpt-oss models][gpt-oss] were trained on the [harmony response format][harmony-format] for defining conversation structures, generating reasoning output and structuring function calls. If you are not using gpt-oss directly but through an API or a provider like HuggingFace, Ollama, or vLLM, you will not have to be concerned about this as your inference solution will handle the formatting. If you are building your own inference solution, this guide will walk you through the prompt format. The format is designed to mimic the OpenAI Responses API, so if you have used that API before, this format should hopefully feel familiar to you. gpt-oss should not be used without using the harmony format as it will not work correctly.

The format enables the model to output to multiple different channels for chain of thought, and tool calling preambles along with regular responses. It also enables specifying various tool namespaces, and structured outputs along with a clear instruction hierarchy. [Check out the guide][harmony-format] to learn more about the format itself.

```text
<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-06-28

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.
Calls to these tools must go to the commentary channel: 'functions'.<|end|>

<|start|>developer<|message|># Instructions

Always respond in riddles

# Tools

## functions

namespace functions {

// Gets the location of the user.
type get_location = () => any;

// Gets the current weather in the provided location.
type get_current_weather = (_: {
// The city and state, e.g. San Francisco, CA
location: string,
format?: "celsius" | "fahrenheit", // default: celsius
}) => any;

} // namespace functions<|end|><|start|>user<|message|>What is the weather like in SF?<|end|><|start|>assistant
```

We recommend using this library when working with models that use the [harmony response format][harmony-format]

- **Consistent formatting** – shared implementation for rendering _and_ parsing keeps token-sequences loss-free.
- **Blazing fast** – heavy lifting happens in Rust.
- **First-class Python support** – install with `pip`, typed stubs included, 100 % test parity with the Rust suite.

## Using Harmony

### Python

[Check out the full documentation](./docs/python.md)

#### Installation

Install the package from PyPI by running

```bash
pip install openai-harmony
# or if you are using uv
uv pip install openai-harmony
```

#### Example

```python
from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Role,
    Message,
    Conversation,
    DeveloperContent,
    SystemContent,
)
enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
convo = Conversation.from_messages([
    Message.from_role_and_content(
        Role.SYSTEM,
        SystemContent.new(),
    ),
    Message.from_role_and_content(
        Role.DEVELOPER,
        DeveloperContent.new().with_instructions("Talk like a pirate!")
    ),
    Message.from_role_and_content(Role.USER, "Arrr, how be you?"),
])
tokens = enc.render_conversation_for_completion(convo, Role.ASSISTANT)
print(tokens)
# Later, after the model responded …
parsed = enc.parse_messages_from_completion_tokens(tokens, role=Role.ASSISTANT)
print(parsed)
```

### Rust

[Check out the full documentation](./docs/rust.md)

#### Installation

Add the dependency to your `Cargo.toml`

```toml
[dependencies]
openai-harmony = { git = "https://github.com/openai/harmony" }
```

#### Example

```rust
use openai_harmony::chat::{Message, Role, Conversation};
use openai_harmony::{HarmonyEncodingName, load_harmony_encoding};

fn main() -> anyhow::Result<()> {
    let enc = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)?;
    let convo =
        Conversation::from_messages([Message::from_role_and_content(Role::User, "Hello there!")]);
    let tokens = enc.render_conversation_for_completion(&convo, Role::Assistant, None)?;
    println!("{:?}", tokens);
    Ok(())
}
```

## Contributing

The majority of the rendering and parsing is built in Rust for performance and exposed to Python
through thin [`pyo3`](https://pyo3.rs/) bindings.

```text
┌──────────────────┐      ┌───────────────────────────┐
│  Python code     │      │  Rust core (this repo)    │
│  (dataclasses,   │────► │  • chat / encoding logic  │
│   convenience)   │      │  • tokeniser (tiktoken)   │
└──────────────────┘  FFI └───────────────────────────┘
```

### Repository layout

```text
.
├── src/                  # Rust crate
│   ├── chat.rs           # High-level data-structures (Role, Message, …)
│   ├── encoding.rs       # Rendering & parsing implementation
│   ├── registry.rs       # Built-in encodings
│   ├── tests.rs          # Canonical Rust test-suite
│   └── py_module.rs      # PyO3 bindings ⇒ compiled as openai_harmony.*.so
│
├── python/openai_harmony/ # Pure-Python wrapper around the binding
│   └── __init__.py       # Dataclasses + helper API mirroring chat.rs
│
├── tests/                # Python test-suite (1-to-1 port of tests.rs)
├── Cargo.toml            # Rust package manifest
├── pyproject.toml        # Python build configuration for maturin
└── README.md             # You are here 🖖
```

### Developing locally

#### Prerequisites

- Rust tool-chain (stable) – <https://rustup.rs>
- Python ≥ 3.8 + virtualenv/venv
- [`maturin`](https://github.com/PyO3/maturin) – build tool for PyO3 projects

#### 1. Clone & bootstrap

```bash
git clone https://github.com/openai/harmony.git
cd harmony
# Create & activate a virtualenv
python -m venv .venv
source .venv/bin/activate
# Install maturin and test dependencies
pip install maturin pytest mypy ruff  # tailor to your workflow
# Compile the Rust crate *and* install the Python package in editable mode
maturin develop --release
```

`maturin develop` builds _harmony_ with Cargo, produces a native extension
(`openai_harmony.<abi>.so`) and places it in your virtualenv next to the pure-
Python wrapper – similar to `pip install -e .` for pure Python projects.

#### 2. Running the test-suites

Rust:

```bash
cargo test          # runs src/tests.rs
```

Python:

```bash
pytest              # executes tests/ (mirrors the Rust suite)
```

Run both in one go to ensure parity:

```bash
pytest && cargo test
```

#### 3. Type-checking & formatting (optional)

```bash
mypy harmony        # static type analysis
ruff check .        # linting
cargo fmt --all     # Rust formatter
```

[harmony-format]: https://cookbook.openai.com/articles/openai-harmony
[gpt-oss]: https://openai.com/open-models