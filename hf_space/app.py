"""
OSS Assistant - Qwen2.5-0.5B-Instruct
HuggingFace Spaces — Gradio 4.44.0 (pinned)
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
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
model.eval()
device = next(model.parameters()).device
print(f"Model loaded on: {device}")

# ─────────────────────────────────────────
# 2. GUARDRAILS
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
    try:
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expr):
            return "Error: invalid characters"
        result = eval(expr, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def tool_datetime(_: str = "") -> str:
    return datetime.now().strftime("%A, %B %d %Y, %H:%M:%S")

def tool_word_count(text: str) -> str:
    return f"{len(text.split())} words, {len(text)} characters"

TOOLS = {
    "calculator": tool_calculator,
    "datetime": tool_datetime,
    "word_count": tool_word_count,
}

TOOL_DESCRIPTIONS = """You have access to these tools. Call them by including in your response:
[TOOL: calculator | <math expression>]   e.g. [TOOL: calculator | 23 * 47]
[TOOL: datetime | ]                       returns current date and time
[TOOL: word_count | <some text>]          counts words in text
Use tools when asked for calculations, current time/date, or word counts."""

def run_tools(response: str) -> str:
    pattern = re.compile(r'\[TOOL:\s*(\w+)\s*\|\s*(.*?)\]', re.IGNORECASE)
    def replace_tool(match):
        name = match.group(1).lower().strip()
        arg  = match.group(2).strip()
        if name in TOOLS:
            try:
                return f"[Result: {TOOLS[name](arg)}]"
            except Exception as e:
                return f"[Tool error: {e}]"
        return match.group(0)
    return pattern.sub(replace_tool, response)

# ─────────────────────────────────────────
# 4. MEMORY
# ─────────────────────────────────────────
MEMORY_FILE = "memory_store.json"

def load_memory() -> dict:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_memory(store: dict):
    with open(MEMORY_FILE, "w") as f:
        json.dump(store, f, indent=2)

def extract_memory_facts(user_msg: str) -> list:
    facts = []
    m = re.search(r"my name is (\w+)", user_msg, re.IGNORECASE)
    if m:
        facts.append(f"User's name is {m.group(1)}")
    m = re.search(r"i (work in|use|code in|program in) (\w+)", user_msg, re.IGNORECASE)
    if m:
        facts.append(f"User works with {m.group(2)}")
    m = re.search(r"i('m| am) a[n]? ([\w ]+)(developer|engineer|designer|student|researcher)", user_msg, re.IGNORECASE)
    if m:
        facts.append(f"User is a {m.group(2)}{m.group(3)}")
    return facts

memory_store = load_memory()

# ─────────────────────────────────────────
# 5. OBSERVABILITY
# ─────────────────────────────────────────
LOG_FILE = "obs_log.jsonl"

def log_turn(session_id, user_msg, bot_msg, latency_ms,
             tokens_in, tokens_out, blocked, tools_used):
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
        "cost_usd": 0.0,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

# ─────────────────────────────────────────
# 6. INFERENCE
# ─────────────────────────────────────────
SYSTEM_PROMPT = f"""You are a helpful, harmless, and honest personal assistant.
Answer clearly and concisely.

{TOOL_DESCRIPTIONS}"""

def generate(messages: list, max_new_tokens: int = 512):
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
# 7. CHAT FUNCTION
# Gradio 4.44: history = list of [user_str, bot_str] tuples
# ─────────────────────────────────────────
SESSION_ID = f"session_{int(time.time())}"

def chat(message, history):
    # Guardrail: check input
    session_id = SESSION_ID
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
    # Add chat history

    if history:
        for msg in history:
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content")

                if role and content:
                    messages.append(
                    {
                        "role": role,
                        "content": content
                    }
                    )

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
    new_facts = extract_memory_facts(message)
    if new_facts:
        if session_id not in memory_store:
            memory_store[session_id] = []
        memory_store[session_id].extend(new_facts)
        save_memory(memory_store)

    # Log
    log_turn(session_id, message, response, latency_ms, tok_in, tok_out,
             blocked=False, tools_used=tools_used)

    return response
# ─────────────────────────────────────────
# 8. GRADIO UI — pinned to 4.44.0
# ─────────────────────────────────────────
demo = gr.ChatInterface(
    fn=chat,
    title="OSS Assistant - Qwen2.5-0.5B-Instruct",
    description=(
        "**Features:** Multi-turn memory | Tool use | Guardrails | Observability\n\n"
        "**Try:** What is 234 * 567? | What time is it? | My name is Alex"
    ),
    examples=[
        "What is 1337 * 42?",
        "What is today's date and time?",
        "My name is Alex and I am a software engineer",
        "Explain what a transformer model is in 3 sentences",
        "Write a haiku about coding",
    ],
    cache_examples=False,
)

demo.launch()
