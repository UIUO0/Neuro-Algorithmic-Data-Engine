<div align="center">

# 🧠 Neuro-Algorithmic-Data-Engine

### Text-to-SQL · Parameter-Efficient Fine-Tuning · Apple MLX

**Transforming natural language questions into complex SQL queries using a locally fine-tuned DeepSeek-8B model on Apple Silicon.**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![MLX](https://img.shields.io/badge/Apple-MLX-black?logo=apple&logoColor=white)](https://github.com/ml-explore/mlx)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon-orange?logo=apple)](https://support.apple.com/en-us/111902)

</div>

---

## 📋 Executive Summary

**Neuro-Algorithmic-Data-Engine** is an end-to-end machine learning pipeline that fine-tunes a large language model to translate natural language questions into structurally valid, executable SQL queries — a critical capability for **Big Data analysis** in enterprise environments.

The project demonstrates production-grade engineering across three core domains:

| Domain | Implementation |
|---|---|
| **NLP** | Fine-tuning an 8-billion-parameter LLM to understand natural language intent and map it to structured query language |
| **Data Engineering** | Streaming pipeline processing 78K+ training examples with O(1) memory — capable of scaling to millions without modification |
| **Algorithmic Optimization** | QLoRA (4-bit quantization + Low-Rank Adaptation) achieving **95.2% loss reduction** while fitting entirely within 16GB Unified Memory |

### 💼 Business Value & Enterprise Impact

- **Democratizing Big Data:** Empowers non-technical stakeholders (executives, product managers, marketing) to instantly extract complex insights directly from enterprise databases using conversational natural language, eliminating the bottleneck of waiting for data engineering teams.
- **Cost-Effective & Private AI:** Demonstrates how organizations can deploy powerful 8-billion-parameter LLMs strictly locally on consumer-grade hardware (Apple M4, 16GB RAM) using QLoRA. This ensures **100% data privacy** for sensitive schemas while bypassing expensive cloud GPU operational costs.
- **Scalable Infrastructure:** The O(1) memory streaming pipeline guarantees that the data ingestion engine can scale to process massive, enterprise-level datasets without requiring hardware upgrades or suffering from Out-Of-Memory (OOM) failures.

### Key Technical Highlights

- **Model**: [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit) — 4-bit quantized (QLoRA)
- **Framework**: [Apple MLX](https://github.com/ml-explore/mlx) — native Apple Silicon acceleration
- **Dataset**: [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) — 78,577 Text-to-SQL examples
- **Hardware**: MacBook M4, 16GB Unified Memory — **runs 100% locally**
- **Peak Memory**: 5.32 GB during training (67% headroom remaining)
- **Trainable Parameters**: 0.065% (5.24M / 8,030M) — extreme parameter efficiency

---

## 🏗️ Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    Neuro-Algorithmic-Data-Engine                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐   │
│  │ data_loader  │────▶│  train_mlx   │────▶│  inference.py   │   │
│  │    .py       │     │     .py      │     │  (Text-to-SQL)  │   │
│  └──────┬──────┘     └──────┬───────┘     └────────┬────────┘   │
│         │                   │                      │            │
│    HuggingFace         mlx_lm.lora           Interactive REPL   │
│    Streaming API       QLoRA Training         + Single Query    │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│  Hardware: Apple M4 · 16GB Unified Memory · Metal GPU            │
│  Framework: MLX (Apple Silicon Native)                           │
└──────────────────────────────────────────────────────────────────┘
```

### Memory Budget — Designed for 16GB Constraint

Every component is engineered to fit within a strict 16GB Unified Memory budget:

```
┌─────────────────────────────────────────────────────┐
│  16GB Unified Memory Budget                         │
│  ├── ~4.5 GB  → Quantized Model (4-bit, 8B params) │
│  ├── ~0.5 GB  → Gradients (8 LoRA layers)           │
│  ├── ~0.2 GB  → Optimizer States (Adam)             │
│  ├── ~0.2 GB  → Activations (batch=1, grad ckpt)   │
│  ├── ~0.1 GB  → LoRA Adapters (rank=8)              │
│  ├── ~0.5 GB  → Tokenizer + Overhead                │
│  ├── ~2.0 GB  → OS + System                         │
│  ├───────────────────────────────────────────────── │
│  │  TOTAL       ~8.0 GB                             │
│  │  HEADROOM    ~8.0 GB (50% safety margin)         │
│  └── ~50 KB   → Data Pipeline (O(1) streaming) ←    │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Algorithmic Complexity Analysis

Every function in the data pipeline is annotated with formal Time and Space complexity guarantees. This ensures predictable, scalable performance regardless of dataset size.

### Data Pipeline Complexity (`data_loader.py`)

| Function | Time Complexity | Space Complexity | Memory Impact |
|---|---|---|---|
| `_stream_raw_dataset()` | O(N) total, O(1) per example | O(1) — one example in memory | ~10-50 KB per example |
| `_build_user_prompt()` | O(L) — string concatenation | O(L) — one string allocated | ~1-5 KB per prompt |
| `_transform_to_chat_format()` | O(N) total, O(1) per example | O(1) — transforms in-place | ~1-5 KB per dict |
| `_transform_to_completions_format()` | O(N) total, O(1) per example | O(1) — one example at a time | Negligible |
| `_write_jsonl_streaming()` | O(N) writes, O(N/B) flushes | O(B) — batched I/O buffer | ~10 KB buffer |
| `prepare_dataset()` | O(N) — single pass | O(B) — I/O buffer only | Peak ~50 KB |
| `get_available_memory_gb()` | O(1) — single syscall | O(1) — one float | Negligible |
| `check_memory_safety()` | O(1) — comparison | O(1) | Negligible |
| `_validate_config()` | O(1) — arithmetic checks | O(1) — no allocations | Negligible |

> **Where:** N = dataset size, L = text length per example, B = `DISK_WRITE_BATCH_SIZE` (default: 100)

### Key Complexity Guarantees

- **O(1) Memory Pipeline**: The entire data pipeline uses constant memory regardless of dataset size. Processing 78K examples uses the same RAM as processing 78M.
- **O(N) Single-Pass Processing**: Each example is streamed, transformed, and written in a single pass — no multi-pass algorithms or random access.
- **O(N/B) I/O Optimization**: Disk writes are batched (B=100) to minimize system calls while keeping memory under ~10 KB.

---

## 🔬 Strategic Architecture Decisions

### 1. Streaming-First Data Processing

**Problem**: Loading the full `sql-create-context` dataset (~78K examples) into memory would consume significant RAM, competing with the model for the 16GB budget.

**Solution**: HuggingFace's streaming API provides a lazy iterator — data is fetched via HTTP chunked transfer and processed one example at a time. The pipeline uses Python generators end-to-end, ensuring zero materialization.

```python
# O(1) memory — only one example exists in memory at any time
dataset = load_dataset("b-mc2/sql-create-context", streaming=True)
for example in dataset:
    yield transform(example)  # Generator — no accumulation
```

### 2. QLoRA — 4-bit Quantization + Low-Rank Adaptation

**Problem**: Full fine-tuning of an 8B parameter model requires ~32 GB in fp16, far exceeding our 16GB budget.

**Solution**: QLoRA combines two techniques:

| Technique | Effect |
|---|---|
| **4-bit Quantization** | Compresses model weights from 16 bits → 4 bits (4× reduction: ~16 GB → ~4.5 GB) |
| **Low-Rank Adaptation** | Only trains small adapter matrices (rank=8) injected into 8 layers — 0.065% of total parameters |

### 3. Gradient Checkpointing

Trades compute for memory by recomputing activations during the backward pass instead of storing them. Reduces activation memory by ~60%, critical for fitting within the 16GB budget.

### 4. Memory-Safe Design

Built-in `psutil` monitoring with a 2GB safety threshold. The pipeline automatically halts if available memory drops below the threshold, preventing system-level OOM crashes:

```python
if count % 1000 == 0 and not check_memory_safety():
    logger.error("Memory safety threshold breached. Stopping stream.")
    break
```

### 5. Adapter Fusion for Deployment

Post-training, LoRA adapters are fused (merged) back into the base model weights, producing a standalone model that:
- Requires no adapter loading at inference time
- Has zero latency overhead compared to the base model
- Can be deployed as a single artifact

---

## 📁 Project Structure

```
Neuro-Algorithmic-Data-Engine/
├── data_loader.py          # Streaming data pipeline (O(1) memory)
├── train_mlx.py            # QLoRA training orchestrator
├── inference.py            # Interactive Text-to-SQL inference engine
├── requirements.txt        # Apple Silicon-optimized dependencies
├── LICENSE                 # MIT License
├── README.md               # This file
├── .gitignore              # Excludes model weights & checkpoints
│
├── data/                   # Generated training data (JSONL)
│   ├── train.jsonl         # 66,802 samples (85%)
│   ├── valid.jsonl         #  7,850 samples (10%)
│   └── test.jsonl          #  3,925 samples  (5%)
│
├── adapters/               # LoRA adapter checkpoints
│   ├── adapters.safetensors
│   ├── adapter_config.json
│   └── 0000*_adapters.safetensors  # Periodic saves
│
├── fused_model/            # Final merged model (ready for inference)
│   ├── model.safetensors
│   ├── config.json
│   └── tokenizer.*
│
└── lora_config.yaml        # Auto-generated training configuration
```

---

## 🚀 Quick Start

### Prerequisites

- **Hardware**: Mac with Apple Silicon (M1/M2/M3/M4) and ≥16 GB Unified Memory
- **OS**: macOS Sonoma 14.0+ recommended
- **Python**: 3.9+

### Step 1 — Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/Neuro-Algorithmic-Data-Engine.git
cd Neuro-Algorithmic-Data-Engine

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies (Apple Silicon optimized)
pip install -r requirements.txt
```

### Step 2 — Prepare Training Data

Stream and process the `sql-create-context` dataset from HuggingFace:

```bash
python3 data_loader.py
```

**Expected output:**
```
Processing: 78577 samples [00:10, 7443 samples/s]
  Train: 66,802 samples → data/train.jsonl
  Valid:  7,850 samples → data/valid.jsonl
  Test:   3,925 samples → data/test.jsonl
✅ Done! Processed 78,577 samples total.
```

**Custom options:**
```bash
# Limit to 5000 samples for quick testing
python3 data_loader.py --max-samples 5000

# Use completions format instead of chat
python3 data_loader.py --format completions
```

### Step 3 — Fine-Tune the Model (QLoRA)

Run the full training pipeline — includes pre-flight checks, memory estimation, training, evaluation, fusion, and test generation:

```bash
python3 train_mlx.py
```

> **⏱ Training Time**: ~45 minutes for 600 iterations on M4 16GB
> **📊 Peak Memory**: ~5.3 GB (well within the 16GB budget)

**Custom training options:**
```bash
# Quick test run (10 iterations)
python3 train_mlx.py --iters 10 --save-every 5 --steps-per-eval 5

# Custom learning rate and more iterations
python3 train_mlx.py --learning-rate 2e-5 --iters 1000

# Training only (skip fusion and test generation)
python3 train_mlx.py --skip-fusion --skip-test
```

### Step 4 — Fuse Adapters into Base Model

If fusion didn't run automatically, merge the LoRA adapters manually:

```bash
python3 -m mlx_lm fuse \
  --model mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit \
  --adapter-path adapters \
  --save-path fused_model
```

### Step 5 — Run Inference (Text-to-SQL)

Use the interactive inference engine to generate SQL queries:

```bash
# Interactive REPL mode — chat with your model
python3 inference.py

# Single query mode
python3 inference.py \
  --question "What is the average salary for each department?" \
  --schema "CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary DECIMAL);"

# JSON output (for scripting/pipelines)
python3 inference.py \
  --question "How many orders per customer?" \
  --schema "CREATE TABLE orders (id INT, customer_id INT, amount DECIMAL);" \
  --json
```

Or use `mlx_lm generate` directly:

```bash
python3 -m mlx_lm generate \
  --model fused_model \
  --prompt "### Table Schema:
CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary DECIMAL);

### Question:
What is the average salary for each department?" \
  --max-tokens 100
```

---

## 🏆 Results Showcase

Real examples of natural language questions translated to SQL by the fine-tuned model:

| # | Natural Language Question | Table Schema | Generated SQL |
|---|---|---|---|
| 1 | "What is the average salary for each department?" | `employees (id, name, department, salary)` | `SELECT department, AVG(salary) FROM employees GROUP BY department` |
| 2 | "How many students scored above 90?" | `students (id, name, score, grade)` | `SELECT COUNT(*) FROM students WHERE score > 90` |
| 3 | "What is the total revenue by region?" | `sales (id, region, revenue, date)` | `SELECT region, SUM(revenue) FROM sales GROUP BY region` |
| 4 | "Find the top 5 customers by order count" | `orders (id, customer_id, amount)` | `SELECT customer_id, COUNT(*) AS cnt FROM orders GROUP BY customer_id ORDER BY cnt DESC LIMIT 5` |
| 5 | "List employees who earn more than their department average" | `employees (id, name, department, salary)` | `SELECT name FROM employees e1 WHERE salary > (SELECT AVG(salary) FROM employees e2 WHERE e1.department = e2.department)` |

---

## 📈 Training Results

| Metric | Start (Iter 1) | End (Iter 600) | Change |
|---|---|---|---|
| **Train Loss** | 2.472 | 0.119 | **↓ 95.2%** |
| **Val Loss** | 2.495 | 0.245 | **↓ 90.2%** |
| **Best Val Loss** | — | 0.102 (Iter 250) | — |
| **Peak Memory** | — | 5.32 GB | 33% of budget |
| **Trainable Params** | — | 5.24M / 8,030M | **0.065%** |

### Training Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Model | DeepSeek-R1-Distill-Llama-8B-4bit | 4-bit quantized for 16GB memory |
| Fine-tune Type | LoRA (QLoRA) | Parameter-efficient, memory-safe |
| LoRA Rank | 8 | Balance of capacity vs. memory |
| LoRA Layers | 8 | Reduced from 16 for memory safety |
| Batch Size | 1 (effective: 4) | Minimum batch + gradient accumulation |
| Learning Rate | 1e-5 | Conservative for stable convergence |
| Max Seq Length | 512 tokens | Conservative for memory |
| Gradient Checkpointing | ✅ Enabled | Trades compute for ~60% activation memory reduction |
| Prompt Masking | ✅ Enabled | Loss computed only on SQL output tokens |

---

## 🔧 Data Engineering Pipeline

### Input–Output Mapping

The data engineering transforms the raw `b-mc2/sql-create-context` dataset into structured training examples:

```
┌──────────────────────────────────────────────────────────┐
│  Raw Dataset (HuggingFace)                               │
│  ├── question: "What is the total revenue by region?"    │
│  ├── context:  "CREATE TABLE sales (id INT, ...)"       │
│  └── answer:   "SELECT region, SUM(revenue) FROM ..."   │
├──────────────────────────────────────────────────────────┤
│  ▼ Data Processing (data_loader.py)                      │
├──────────────────────────────────────────────────────────┤
│  Merged Input Prompt:                                    │
│  "### Table Schema:                                      │
│   CREATE TABLE sales (id INT, region VARCHAR, ...)       │
│                                                          │
│   ### Question:                                          │
│   What is the total revenue by region?"                  │
├──────────────────────────────────────────────────────────┤
│  Training Target:                                        │
│  "SELECT region, SUM(revenue) FROM sales GROUP BY region"│
└──────────────────────────────────────────────────────────┘
```

### Output Format (mlx_lm.lora compatible)

```json
{
  "messages": [
    {"role": "system", "content": "You are a SQL expert. Given a user question and the relevant table schemas, generate the correct SQL query to answer the question."},
    {"role": "user", "content": "### Table Schema:\nCREATE TABLE sales ...\n\n### Question:\nWhat is the total revenue by region?"},
    {"role": "assistant", "content": "SELECT region, SUM(revenue) FROM sales GROUP BY region;"}
  ]
}
```

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|---|---|---|
| **ML Framework** | [Apple MLX](https://github.com/ml-explore/mlx) | Native Apple Silicon acceleration |
| **Fine-Tuning** | [mlx-lm](https://github.com/ml-explore/mlx-examples) | LoRA/QLoRA training & inference |
| **Base Model** | [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit) | 4-bit quantized LLM |
| **Dataset** | [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) | 78K Text-to-SQL examples |
| **Data Loading** | [HuggingFace Datasets](https://huggingface.co/docs/datasets) | Streaming API for O(1) memory |
| **Tokenization** | [Transformers](https://huggingface.co/docs/transformers) + [SentencePiece](https://github.com/google/sentencepiece) | LLaMA tokenizer |
| **Monitoring** | [psutil](https://github.com/giampaolo/psutil) | Real-time memory safety checks |
| **Inference** | Custom `inference.py` | Interactive REPL + single-query + JSON output |
| **Language** | Python 3.9+ | Core implementation |

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ on Apple Silicon**

*Neuro-Algorithmic-Data-Engine — Where NLP meets Data Analysis*

</div>
