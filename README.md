# Speculative Decoding from Scratch

---

## 📌 What Is This

Speculative decoding is an inference-time technique that speeds up large language model generation without changing output quality.

The idea: instead of generating tokens one by one with the large model, a small draft model quickly proposes K tokens, and the large model verifies all K in a single forward pass. Since transformers process all positions in parallel, verifying 5 tokens takes roughly the same time as verifying 1 — so you get up to K× more tokens per large model call.

This project implements the algorithm from scratch using HuggingFace transformers — no libraries, no abstractions, just raw forward passes and probability comparisons.

---

## 📊 Results

| Method | Speedup |
|---|---|
| Greedy (Qwen3-8B only) | 1.0x |
| Speculative decoding (K=5) | **~2x** |

Benchmark run on 1000 tokens, RTX 3090.

---

## 🔄 How It Works

```
prompt
  │
  ├── draft model (Qwen3-1.7B)
  │   generates K tokens quickly
  │
  └── target model (Qwen3-8B)
      verifies all K tokens in ONE forward pass
      accept if target agrees, otherwise write its own token and restart
```

**Why faster:** the target model does 1 forward pass instead of K.

**Why quality doesn't suffer:** the target model either accepts the draft token or replaces it with its own. A bad token never slips through.

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Draft model | `Qwen/Qwen3-1.7B` |
| Target model | `Qwen/Qwen3-8B` |
| Framework | `transformers`, `torch` |
| Hardware | Single GPU (RTX 3090, 24GB VRAM) |

Both models share the same tokenizer — required for speculative decoding to work correctly.

---

## 🗂️ Repository Structure

```
speculative-decoding/
│
├── speculative_decoding.ipynb   # full implementation + benchmark
└── README.md
```

---

## Created by Foutx
