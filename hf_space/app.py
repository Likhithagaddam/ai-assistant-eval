"""
OSS Assistant — Qwen2.5-0.5B-Instruct
Deployed on Hugging Face Spaces

Features:
- Multi-turn conversation with persistent memory
- Guardrails / safety layer
- Tool use (calculator, date, web search stub)
- Observability (logging all turns to JSONL)
- Cost + latency tracking per request
"""

import gradio as gr
import torch
import time
import json
import re
import math
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────────────────────
# 1. MODEL LOADING
# ─────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
model.eval()
device = next(model.parameters()).device
print(f"Model loaded on: {device}")

# ─────────────────────────────────────────
# 2. GUARDRAILS — blocked patterns
# ─────────────────────────────────────────
BLOCKED_PATTERNS = [
    r"(how to|steps to|make|build|create|synthesize).{0,30}(bomb|explosive|weapon|meth|cocaine|heroin|malware|virus|ransomware)",
    r"(jailbreak|ignore (all |your |previous )?instructions|you are now DAN|do anything now)",
    r"(hack|bypass|crack).{0,20}(bank|password|system|account)",
    r"(child|minor|underage).{0,20}(sex|naked|nude|porn)",
    r"(step.by.step|instructions).{0,30}(kill|murder|attack|assault)",
]

BLOCKED_COMPILED = [re.compile(p, re.IGNORECASE) for p in BLOCKED_PATTERNS]

SAFE_RESPONSE = (
    "I'm not able to help with that request. "
    "I'm designed to be a helpful, harmless assistant. "
    "Please ask me something else!"
)

def is_blocked(text: str) -> bool:
    return any(p.search(text) for p in BLOCKED_COMPILED)

# ─────────────────────────────────────────
# 3. TOOL USE
# ─────────────────────────────────────────
def tool_calculator(expr: str) -> str:
    """Safe eval for math expressions."""
    try:
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expr):
            return "Error: invalid characters"
        result = eval(expr, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def tool_datetime() -> str:
    return datetime.now().strftime("%A, %B %d %Y, %H:%M:%S")

def tool_word_count(text: str) -> str:
    words = len(text.split())
    chars = len(text)
    return f"{words} words, {chars} characters"

TOOLS = {
    "calculator": tool_calculator,
    "datetime": tool_datetime,
    "word_count": tool_word_count,
}

TOOL_DESCRIPTIONS = """You have access to these tools. Call them by including in your response:
[TOOL: calculator | <math expression>]   → evaluates math, e.g. [TOOL: calculator | 23 * 47]
[TOOL: datetime | ]                       → returns current date and time
[TOOL: word_count | <some text>]          → counts words in text

Use tools when the user asks for calculations, current time/date, or word counts.
After calling a tool, use its result to answer the user."""

def run_tools(response: str) -> str:
    """Find and execute any tool calls in the model response."""
    pattern = re.compile(r'\[TOOL:\s*(\w+)\s*\|\s*(.*?)\]', re.IGNORECASE)
    def replace_tool(match):
        tool_name = match.group(1).lower().strip()
        tool_arg  = match.group(2).strip()
        if tool_name in TOOLS:
            fn = TOOLS[tool_name]
            try:
                result = fn(tool_arg) if tool_arg else fn()
                return f"[Result: {result}]"
            except Exception as e:
                return f"[Tool error: {e}]"
        return match.group(0)
    return pattern.sub(replace_tool, response)

# ─────────────────────────────────────────
# 4. MEMORY — persistent per session
# ─────────────────────────────────────────
MEMORY_FILE = "memory_store.json"

def load_memory() -> dict:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE) as f:
                return json.load(f)
        except:
            pass
    return {}

def save_memory(store: dict):
    with open(MEMORY_FILE, "w") as f:
        json.dump(store, f, indent=2)

def extract_memory_facts(user_msg: str, bot_msg: str) -> list:
    """Simple rule-based memory extraction."""
    facts = []
    name_match = re.search(r"my name is (\w+)", user_msg, re.IGNORECASE)
    if name_match:
        facts.append(f"User's name is {name_match.group(1)}")
    lang_match = re.search(r"i (work in|use|code in|program in) (\w+)", user_msg, re.IGNORECASE)
    if lang_match:
        facts.append(f"User works with {lang_match.group(2)}")
    job_match = re.search(r"i(\'m| am) a[n]? ([\w\s]+)(developer|engineer|designer|student|researcher)", user_msg, re.IGNORECASE)
    if job_match:
        facts.append(f"User is a {job_match.group(2)}{job_match.group(3)}")
    return facts

# ─────────────────────────────────────────
# 5. OBSERVABILITY — log every turn
# ─────────────────────────────────────────
LOG_FILE = "obs_log.jsonl"

def log_turn(session_id: str, user_msg: str, bot_msg: str,
             latency_ms: float, tokens_in: int, tokens_out: int,
             blocked: bool, tools_used: list):
    entry = {
        "ts": datetime.utcnow().isoformat(),
        "session": session_id,
        "user": user_msg[:200],
        "assistant": bot_msg[:200],
        "latency_ms": round(latency_ms, 1),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "blocked": blocked,
        "tools_used": tools_used,
        "cost_usd": 0.0,  # OSS = free; placeholder for comparison
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

# ─────────────────────────────────────────
# 6. INFERENCE
# ─────────────────────────────────────────
SYSTEM_PROMPT = f"""You are a helpful, harmless, and honest personal assistant.
Answer clearly and concisely.

{TOOL_DESCRIPTIONS}"""

def generate(messages: list, max_new_tokens=512) -> tuple[str, int, int]:
    """Returns (response_text, tokens_in, tokens_out)."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)
    tokens_in = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    tokens_out = outputs.shape[1] - tokens_in
    response = tokenizer.decode(
        outputs[0][tokens_in:], skip_special_tokens=True
    ).strip()
    return response, tokens_in, tokens_out

# ─────────────────────────────────────────
# 7. MAIN CHAT FUNCTION
# ─────────────────────────────────────────
memory_store = load_memory()

def chat(message: str, history: list, session_id: str):
    # Guardrail: check input
    if is_blocked(message):
        log_turn(session_id, message, SAFE_RESPONSE, 0, 0, 0, blocked=True, tools_used=[])
        return SAFE_RESPONSE

    # Build message list with memory context
    memory_facts = memory_store.get(session_id, [])
    memory_context = ""
    if memory_facts:
        memory_context = "\n\nWhat I remember about you:\n" + "\n".join(f"- {f}" for f in memory_facts[-5:])

    system = SYSTEM_PROMPT + memory_context
    messages = [{"role": "system", "content": system}]

    # Add last 10 turns
    for u, b in history[-10:]:
        messages.append({"role": "user", "content": u})
        if b:
            messages.append({"role": "assistant", "content": b})
    messages.append({"role": "user", "content": message})

    # Generate
    t0 = time.time()
    response, tok_in, tok_out = generate(messages)
    latency_ms = (time.time() - t0) * 1000

    # Run any tool calls in the response
    tools_used = re.findall(r'\[TOOL:\s*(\w+)', response, re.IGNORECASE)
    response = run_tools(response)

    # Guardrail: check output too
    if is_blocked(response):
        response = SAFE_RESPONSE

    # Extract and save memory facts
    new_facts = extract_memory_facts(message, response)
    if new_facts:
        if session_id not in memory_store:
            memory_store[session_id] = []
        memory_store[session_id].extend(new_facts)
        save_memory(memory_store)

    # Log
    log_turn(session_id, message, response, latency_ms, tok_in, tok_out,
             blocked=False, tools_used=tools_used)

    return response

def chat_wrapper(message, history, session_id):
    if not session_id:
        session_id = f"session_{int(time.time())}"
    response = chat(message, history, session_id)
    return response

# ─────────────────────────────────────────
# 8. GRADIO UI
# ─────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title="OSS Assistant") as demo:
    gr.Markdown("""
    # 🤗 OSS Assistant — Qwen2.5-0.5B-Instruct
    **Features:** Multi-turn memory · Tool use · Guardrails · Observability

    **Try:** `What is 234 * 567?` · `What time is it?` · `My name is Alex`
    """)

    session_id = gr.State(value=f"session_{int(time.time())}")

    chatbot = gr.Chatbot(height=450, bubble_full_width=False)
    msg = gr.Textbox(placeholder="Type your message...", show_label=False, scale=4)

    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary", scale=1)
        clear_btn  = gr.Button("Clear", scale=1)

    gr.Examples(
        examples=[
            "What is 1337 * 42?",
            "What is today's date and time?",
            "My name is Alex and I'm a software engineer",
            "Explain what a transformer model is in 3 sentences",
            "Write a haiku about coding",
        ],
        inputs=msg
    )

    def respond(message, chat_history, sid):
        bot_response = chat_wrapper(message, chat_history, sid)
        chat_history = chat_history + [[message, bot_response]]
        return "", chat_history

    msg.submit(respond, [msg, chatbot, session_id], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot, session_id], [msg, chatbot])
    clear_btn.click(lambda: [], outputs=chatbot)

demo.launch()
