# AI Assistant Evaluation — Founding AI/ML Engineer

Comparison of two personal AI assistants evaluated on hallucination rate, safety, and bias.

| | OSS Assistant | Frontier Assistant |
|---|---|---|
| **Model** | Qwen2.5-0.5B-Instruct | Llama 3.1-8B-Instant |
| **Hosting** | HuggingFace Spaces (free) | Groq API (free tier) |
| **Interface** | Gradio | Gradio |
| **Memory** | Persistent (file-backed) | Sliding window (20 turns) |
| **Tools** | Calculator, datetime, word count | None (API handles it) |
| **Guardrails** | Regex + output filter | Built into Llama 3.1 |
| **Observability** | JSONL logging per turn | Basic logging |

---

## Repository Structure

```
ai-assistant-eval/
├── hf_space/                        # Deployed OSS assistant (HuggingFace Spaces)
│   ├── app.py                       # Main app with guardrails, memory, tools, observability
│   ├── requirements.txt
│   └── README.md                    # HF Space card
├── notebooks/
│   ├── notebook_1_oss_assistant.ipynb
│   ├── notebook_2_frontier_assistant.ipynb
│   └── notebook_3_evaluation.ipynb
├── eval/
│   └── evaluation_report.pdf
└── README.md
```

---

## Live Demo

**OSS Assistant (HuggingFace Spaces):**
https://huggingface.co/spaces/YOUR_USERNAME/oss-assistant-qwen25

*(Deploy steps below — replace YOUR_USERNAME)*

---

## Setup Instructions

### Option A — Run in Google Colab (quickest)

1. Open `notebooks/notebook_1_oss_assistant.ipynb` in Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Run all cells

4. Open `notebooks/notebook_2_frontier_assistant.ipynb` in Colab
5. Add `GROQ_API_KEY` to Colab Secrets (🔑 icon, no GPU needed)
6. Get free key at https://console.groq.com
7. Run all cells

8. Run Cell 6 in both notebooks to save `oss_responses.json` and `frontier_responses.json`
9. Open `notebooks/notebook_3_evaluation.ipynb`, upload both JSONs, run all cells
10. PDF auto-downloads

### Option B — Deploy OSS assistant to HuggingFace Spaces

```bash
# 1. Install HF CLI
pip install huggingface_hub

# 2. Login
huggingface-cli login

# 3. Create a new Space at https://huggingface.co/new-space
#    SDK: Gradio | Hardware: CPU Basic (free) or T4 (paid)

# 4. Clone and push
git clone https://huggingface.co/spaces/YOUR_USERNAME/oss-assistant-qwen25
cp hf_space/* oss-assistant-qwen25/
cd oss-assistant-qwen25
git add . && git commit -m "Deploy OSS assistant"
git push
```

### Option C — Run locally

```bash
pip install transformers gradio accelerate torch
cd hf_space
python app.py
```

---

## Architecture Decisions

### OSS Model: Qwen2.5-0.5B-Instruct
- Smallest instruction-tuned Qwen2.5 variant — fits on free Colab T4 and HF Spaces CPU
- `apply_chat_template()` ensures correct chat prompt formatting
- float16 on GPU, float32 fallback on CPU

### Frontier Model: Llama 3.1-8B-Instant via Groq
- Truly free API, no credit card required
- 16x more parameters than the OSS model — strong factual accuracy baseline
- OpenAI-compatible SDK — trivial to swap models

### Memory
- **OSS**: file-backed JSON store; regex extracts name/job/language facts from conversation
- **Frontier**: sliding window of last 20 turns in the context

### Guardrails (OSS)
- Input filter: regex patterns block known jailbreaks, harmful instructions, CSAM
- Output filter: same patterns run on generated text before returning to user
- Conservative approach — fast and zero-dependency

### Tool Use (OSS)
- Model learns to emit `[TOOL: calculator | 23*47]` syntax via system prompt
- Post-processing regex finds and executes tool calls before returning to user
- Tools: safe eval calculator, datetime, word counter

### Observability
- Every turn logged to `obs_log.jsonl` with: timestamp, session ID, user input, response, latency (ms), token counts, blocked flag, tools used
- Zero external dependencies (no LangSmith/W&B required for submission)

---

## Tradeoffs

| Aspect | OSS (Qwen2.5-0.5B) | Frontier (Llama 3.1-8B) |
|---|---|---|
| Cost | $0 (free GPU/CPU) | $0 (Groq free tier) |
| Latency (GPU) | 2-5s | 0.3-0.8s |
| Latency (CPU) | 20-40s | 0.3-0.8s |
| Factual accuracy | Lower (small model) | Higher (8B + RLHF) |
| Privacy | Full data control | Data sent to Groq |
| Context window | 32K tokens | 128K tokens |
| Guardrails | Custom (regex) | Built-in alignment |
| Deployment | Needs GPU for speed | Any machine |

---

## Cost + Latency Table (OSS Deployment)

| Platform | Hardware | Cost/month | Avg latency | Notes |
|---|---|---|---|---|
| HF Spaces (CPU Basic) | 2 vCPU, 16GB RAM | **Free** | 25-40s/response | Best free option |
| HF Spaces (T4 Small) | T4 GPU | $0.60/hr | 2-4s/response | Pay-as-you-go |
| Colab Free | T4 GPU | **Free** | 2-4s/response | 12hr session limit |
| RunPod (RTX 3090) | 24GB VRAM | ~$0.44/hr | 1-2s/response | Scalable |
| Replicate | Auto-scale GPU | ~$0.0023/sec | 1-3s/response | Per-call billing |
| Local (M2 MacBook) | Apple Silicon | **$0** | 4-8s/response | Private, offline |

*Latency measured for 200-token response on Qwen2.5-0.5B-Instruct*

---

## What I Would Improve With More Time

1. **Larger OSS model** — Qwen2.5-7B-Instruct scores much higher; needs A100 or quantization
2. **Quantization** — GGUF/AWQ to run 7B on free hardware at acceptable speed
3. **Streaming** — `TextIteratorStreamer` for token-by-token output
4. **RAG** — FAISS + Wikipedia embeddings to ground factual answers and cut hallucinations
5. **LlamaGuard** — replace regex guardrails with a proper safety classifier
6. **LangSmith / W&B** — proper observability dashboard instead of flat JSONL
7. **Larger eval set** — 50+ prompts per category, human-validated ground truth
8. **Structured evals** — RAGAS or DeepEval framework for reproducible benchmarks
9. **Persistent cross-session memory** — SQLite + sentence-transformers vector search
10. **CI/CD** — GitHub Actions to auto-run evals on every push

---

## Evaluation Methodology

- **Factual accuracy** (10 prompts): verifiable questions with known answers; LLM judge scores 0-10
- **Safety / adversarial** (5 prompts): jailbreak attempts; judge scores refusal quality 0-10
- **Bias & fairness** (5 prompts): stereotype/sensitive prompts; judge scores neutrality 0-10
- **Judge model**: `llama-3.1-8b-instant` via Groq at temperature=0.1
- **Scoring**: structured JSON output from judge, averaged per category

Author : Sai Likhitha Gaddam
