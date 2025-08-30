import os
import time
import json
import textwrap
from typing import Iterator, List, Dict

import streamlit as st

# --- Optional dependency: ollama ---
# The app prefers using a local Ollama server for Llama 2 / Code Llama.
# Install: https://ollama.com  -> then: `ollama pull codellama:7b-instruct` (recommended)
# Python client: `pip install ollama`
try:
    from ollama import Client  # type: ignore
    OLLAMA_AVAILABLE = True
except Exception:
    Client = None
    OLLAMA_AVAILABLE = False

APP_TITLE = "🔧 AI Code Generator — Llama 2/Code Llama"
DEFAULT_MODEL = os.getenv("LLAMA_MODEL", "codellama:7b-instruct")  # or "llama2:7b-chat"
DEFAULT_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ------------------------------
# UI SETUP
# ------------------------------
st.set_page_config(page_title="AI Code Generator", page_icon="🧠", layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.text_input("Model (Ollama)", value=DEFAULT_MODEL, help="e.g., codellama:7b-instruct, llama2:7b-chat, codellama:13b-instruct if your machine can handle it")
    host = st.text_input("Ollama host", value=DEFAULT_HOST, help="Ollama server URL; default is http://localhost:11434")

    st.subheader("Generation Controls")
    temperature = st.slider("temperature", 0.0, 1.5, 0.2, 0.05)
    top_p = st.slider("top_p", 0.0, 1.0, 0.9, 0.05)
    num_ctx = st.slider("context window (num_ctx)", 512, 8192, 4096, 256)
    seed = st.number_input("seed (deterministic if set)", value=0, min_value=0, step=1)
    stop_seq = st.text_input("stop sequence (optional)", value="")

    st.markdown("""
    **Tip:** *Code Llama Instruct* models are usually better at following coding prompts. Start with `codellama:7b-instruct`.
    """)

st.markdown(
    """
    Provide a problem description and choose a target language. The model will return production-ready code with comments and tests when possible.
    """
)

# ------------------------------
# Prompt Form
# ------------------------------
with st.form("prompt_form"):
    col1, col2 = st.columns(2)
    with col1:
        language = st.selectbox(
            "Target language",
            [
                "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Rust",
                "PHP", "Swift", "Kotlin", "Ruby", "SQL"
            ],
            index=0,
        )
        framework = st.text_input("Framework / Library (optional)", placeholder="e.g., FastAPI, React, Spring Boot, Express, Flask")
        function_sig = st.text_input("Function/Class signature (optional)", placeholder="e.g., def two_sum(nums: List[int], target: int) -> List[int]:")
    with col2:
        constraints = st.text_area(
            "Constraints / Requirements (optional)",
            height=120,
            placeholder="e.g., time complexity O(n log n), no external deps, include unit tests, follow PEP8, handle edge cases",
        )
        examples = st.text_area(
            "Examples / I/O (optional)",
            height=120,
            placeholder="e.g., Input: [2,7,11,15], 9 -> Output: [0,1]",
        )

    problem = st.text_area(
        "🧩 Problem / Task Description",
        height=200,
        placeholder="Describe what you want to build: a CLI JSON validator, a REST API, a sorting function, a React component, etc.",
    )

    colA, colB, colC = st.columns([1,1,1])
    with colA:
        generate_btn = st.form_submit_button("🚀 Generate Code", use_container_width=True)
    with colB:
        explain_btn = st.form_submit_button("💬 Ask for Explanation", use_container_width=True)
    with colC:
        refine_btn = st.form_submit_button("✨ Refine / Improve", use_container_width=True)

# ------------------------------
# Prompt Templates
# ------------------------------
SYSTEM_PROMPT = """
You are a senior software engineer AI pair-programmer.
- Always output code *first* inside a single fenced code block with the correct language tag.
- After the code, optionally include a brief explanation with bullet points.
- Prefer simplicity, readability, and performance. Add inline comments.
- If tests make sense, include them under a `# Tests` section in the same language.
- Obey user constraints strictly. If a constraint conflicts with correctness, explain and choose correctness.
""".strip()

USER_TEMPLATE = """
Write {language} code for the following task.
{framework_line}
{signature_line}
{constraints_line}
{examples_line}

Task:
{problem}

Return ONLY one main solution. Include minimal setup/usage notes if needed.
""".strip()


def build_user_prompt(language: str, framework: str, function_sig: str, constraints: str, examples: str, problem: str) -> str:
    framework_line = f"Use the {framework} framework/library." if framework else ""
    signature_line = f"Target signature: {function_sig}" if function_sig else ""
    constraints_line = f"Constraints/requirements: {constraints}" if constraints else ""
    examples_line = f"Examples: {examples}" if examples else ""
    return USER_TEMPLATE.format(
        language=language,
        framework_line=framework_line,
        signature_line=signature_line,
        constraints_line=constraints_line,
        examples_line=examples_line,
        problem=problem,
    )


# ------------------------------
# Ollama chat utilities
# ------------------------------
class NullStream:
    def __init__(self):
        self.buffer = []
    def write(self, s: str):
        self.buffer.append(s)
    def getvalue(self) -> str:
        return "".join(self.buffer)


def stream_ollama_chat(client: "Client", model: str, messages: List[Dict], options: Dict) -> Iterator[str]:
    """Yield tokens from an Ollama streaming chat response."""
    stream = client.chat(
        model=model,
        messages=messages,
        stream=True,
        options=options,
    )
    for chunk in stream:
        delta = chunk.get("message", {}).get("content", "")
        if delta:
            yield delta


# ------------------------------
# Main actions
# ------------------------------
if generate_btn or explain_btn or refine_btn:
    if not problem and not explain_btn and not refine_btn:
        st.warning("Please enter a task description.")
        st.stop()

    # Prepare messages (maintain chat history in session state)
    if "history" not in st.session_state:
        st.session_state.history = []  # list of {role, content}

    # Create/validate client
    if not OLLAMA_AVAILABLE:
        st.error("Python package 'ollama' not installed. Run: pip install ollama")
        st.stop()

    try:
        client = Client(host=host)
    except Exception as e:
        st.error(f"Could not connect to Ollama at {host}. Is the server running?\nError: {e}")
        st.stop()

    # Build the current user message
    if generate_btn:
        user_prompt = build_user_prompt(language, framework, function_sig, constraints, examples, problem)
    elif explain_btn:
        user_prompt = "Explain the previously generated code step-by-step and suggest improvements."
    else:  # refine_btn
        user_prompt = "Refine and improve the previously generated code for performance, readability, and edge cases. Include tests if missing."

    # Combine with history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.history + [{"role": "user", "content": user_prompt}]

    # Options for generation
    options = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "num_ctx": int(num_ctx),
    }
    if seed:
        options["seed"] = int(seed)
    if stop_seq.strip():
        options["stop"] = [stop_seq]

    # Stream output
    st.subheader("🧪 Output")
    output_area = st.empty()

    # Use write_stream to render tokens as they arrive
    try:
        full_text = st.write_stream(stream_ollama_chat(client, model_name, messages, options))
    except Exception as e:
        st.error(f"Generation failed: {e}")
        st.stop()

    # Persist in history and offer download
    st.session_state.history.extend([
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": full_text},
    ])

    if full_text:
        st.download_button(
            label="⬇️ Download Result",
            data=full_text,
            file_name="generated_code.txt",
            mime="text/plain",
            use_container_width=True,
        )

# ------------------------------
# Helper: quick examples
# ------------------------------
st.divider()
st.markdown("### 🧪 Quick Prompts")
ex_col1, ex_col2, ex_col3 = st.columns(3)
with ex_col1:
    if st.button("CLI JSON Validator (Python)", use_container_width=True):
        st.session_state.update({
            "prompt_form-problem": "Build a Python CLI tool that validates a JSON file against a JSON Schema. Accept paths via CLI args. Print a helpful error with line/column if invalid.",
            "prompt_form-language": "Python",
            "prompt_form-framework": "",
            "prompt_form-function_sig": "",
            "prompt_form-constraints": "No external deps beyond jsonschema; include tests; handle big files.",
            "prompt_form-examples": "",
        })
with ex_col2:
    if st.button("REST CRUD (Node/Express)", use_container_width=True):
        st.session_state.update({
            "prompt_form-problem": "Create a REST API for a 'tasks' resource with Express. Include routes, validation, and in-memory store. Add Jest tests.",
            "prompt_form-language": "JavaScript",
            "prompt_form-framework": "Express",
            "prompt_form-function_sig": "",
            "prompt_form-constraints": "No DB; structuring routes/controllers; include Jest tests.",
            "prompt_form-examples": "",
        })
with ex_col3:
    if st.button("Binary Search (Go)", use_container_width=True):
        st.session_state.update({
            "prompt_form-problem": "Implement binary search over a sorted slice of ints in Go. Return index or -1.",
            "prompt_form-language": "Go",
            "prompt_form-framework": "",
            "prompt_form-function_sig": "func BinarySearch(nums []int, target int) int",
            "prompt_form-constraints": "O(log n); include unit tests.",
            "prompt_form-examples": "",
        })

st.caption(
    "Run an Ollama model locally (e.g., `codellama:7b-instruct` or `llama2:7b-chat`) for best results. Larger models require more RAM/VRAM."
)
