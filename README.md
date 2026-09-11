# Next Word Prediction

A complete, ready-to-run next-word-prediction system with **three swappable
prediction engines**, a **FastAPI backend**, a **web interface**, and
**pretrained models included out of the box** — no training required to
try any of them.

This project was built as a consolidated, production-quality upgrade of
several smaller reference implementations (an n-gram/Markov-chain
predictor, a word-level LSTM, a RoBERTa masked-language-model predictor,
and a federated-learning research repo), and additionally includes a
**from-scratch modern decoder transformer** (RoPE, SwiGLU, grouped-query
attention, tied embeddings) — trained, evaluated, and quantized end to end
on nothing but a single CPU core, with every number below measured, not
assumed.

<p align="center">
  <img src="docs/screenshot.png" alt="Next Word web interface — a text box with live next-word suggestions shown as keycap-style buttons with probability bars, with a tab to switch between the Transformer, GPT (from scratch), and LSTM engines" width="700">
</p>

<p align="center"><sub>The web interface: type a sentence, get ranked next-word suggestions in real time, click or press <kbd>Tab</kbd> to insert.</sub></p>

---

## Contents

- [Overview](#overview)
- [Methodology](#methodology)
- [Results — the from-scratch GPT engine](#results--the-from-scratch-gpt-engine)
- [Technology stack](#technology-stack)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Design notes](#design-notes)

---

## Overview

Given a piece of text the user is typing, the system predicts the most
likely next word(s), ranks them by probability, and displays them for
one-click insertion — the same interaction pattern as a phone keyboard's
predictive-text bar, generalized to three different prediction strategies:

| | LSTM engine | GPT engine (from scratch) | Transformer engine |
|---|---|---|---|
| **Needs training?** | No — pretrained model included; retrain any time | No — pretrained + quantized checkpoint included; retrain any time | No — pretrained DistilBERT, works immediately |
| **Architecture** | 2-layer LSTM, word-level tokens | 6-layer modern decoder (RoPE, SwiGLU, GQA, RMSNorm, tied embeddings), byte-level BPE | 6-layer DistilBERT, WordPiece |
| **Learns from** | Whatever text you train it on (ships on Shakespeare) | Whatever text you train it on (ships on Shakespeare + public-domain novels) | A huge general-English corpus, already learned by DistilBERT |
| **Params** | ~1.4M | 5.77M total / 4.23M non-embedding | ~66M |
| **Shipped size** | ~22MB | **7.0MB** (int8, matmul+embedding quantized) | ~250MB |
| **Strength** | Simple, easy to read end to end | Modern architecture at a size that actually trains on a laptop CPU | Fluent, grammatically aware, zero training needed |

All three engines sit behind the same `/predict` API and the same UI
toggle, so they're directly comparable and interchangeable.

## Methodology

The prediction task is framed as **language modelling**: given a sequence
of preceding tokens, estimate a probability distribution over the
vocabulary for the next one, and return the top-k most likely words.

**LSTM engine — trained from scratch, word-level:**
1. **Corpus ingestion** — the training text is split into lines/sentences
   so the model never learns spurious transitions across unrelated chunks.
2. **Tokenization** — a Keras `Tokenizer` builds a word ↔ integer vocabulary
   (capped at 8,000 words; out-of-vocabulary words map to a single `<OOV>`
   token).
3. **Sequence generation** — every sentence is expanded into overlapping
   n-gram prefixes (`[w₁,w₂] → w₃`, `[w₁,w₂,w₃] → w₄`, ...), each capped
   at 20 tokens of context.
4. **Model** — `Embedding → LSTM(150) → Dropout → LSTM(100) → Dropout →
   Dense(softmax over vocab)`.
5. **Training** — Adam optimizer, sparse categorical cross-entropy (labels
   kept as plain integers rather than one-hot vectors, to keep memory use
   manageable). The bundled model was trained for 10 epochs on the full
   ~1.1MB corpus.
6. **Inference** — the input text is tokenized and padded the same way,
   passed through the network, and the top-k highest-probability words
   are returned.

**GPT engine — trained from scratch, modern architecture, sub-word level:**
See [Results](#results--the-from-scratch-gpt-engine) below for the full
design rationale, architecture, training run, and quantization numbers —
this one gets its own section because every choice in it is backed by a
measurement, not a default.

**Transformer engine — pretrained, zero-shot:**
Rather than training a model, the input text is appended with a `[MASK]`
token and passed to DistilBERT (a masked-language model pretrained on
BookCorpus + English Wikipedia). The model's existing knowledge of English
grammar and semantics is used directly — its predicted distribution over
the mask position *is* the next-word prediction, filtered to keep only
real, complete words and re-ranked to the requested `top_k`.

**Why include all three?** They sit on genuinely different points of the
same trade-off: a tiny model you fully own and train yourself (LSTM); a
small model with a modern, efficient architecture that you also train
yourself but that scales far better with more data/compute if you have
it (GPT engine); and a large model with far more general world/language
knowledge that you don't have to train at all (transformer). Comparing
them side by side is more useful than picking one and hiding the others.

## Results — the from-scratch GPT engine

**Model:** 6-layer decoder-only transformer, d=256, 6,000-token byte-level
BPE vocabulary, 128-token context. **5.77M parameters total (4.23M
non-embedding).**

| metric | held-out (val) | test |
|---|---|---|
| loss | 5.565 | 5.614 |
| perplexity | 261.1 | 274.2 |
| top-1 accuracy | 16.0% | 16.0% |
| top-3 accuracy | 25.8% | 25.4% |
| top-5 accuracy | 30.9% | 30.2% |

Trained for **1,000 steps** (batch size 24, ~3,072 tokens/step) on a
**single CPU core, no GPU** — about 26 minutes wall-clock. Loss went
8.76 → 3.75 (train) / 261 → 274 perplexity (held-out) over that run;
training loss kept falling past step ~650 while validation loss flattened
and then crept up, so the run was stopped there rather than continuing to
overfit a 2.1M-token corpus with a 5.77M-parameter model.

Measured Python/CPU inference latency (this environment, single core,
no batching): **~88ms median, ~100ms mean per prediction.** This is a
plain PyTorch forward pass, not a browser/WASM build — no ONNX/WASM
export is included in this project, so no in-browser latency number is
claimed here.

### Quantization

Shipping decision made by measurement, not by default (`gpt/quantize.py`):

| variant | size | test perplexity | test top-5 | Δ top-5 |
|---|---|---|---|---|
| fp32 | 22.0 MB | 274.2 | 30.2% | — |
| int8 (matmul only) | 11.4 MB | 276.8 | 30.1% | −0.14pp |
| int8 (matmul + embedding) | **7.0 MB** | 277.2 | 30.1% | −0.17pp |

Quantizing the tied embedding table on top of the matmuls costs another
0.03 percentage points of top-5 accuracy and saves another 4.4MB (a 3.1×
total size reduction vs. fp32) — worth it by the same logic the project
uses throughout: **top-5 is what a person experiences**, so that's the
gate (`TOP5_DROP_GATE_PP = 1.0` in `gpt/quantize.py`), not perplexity in
isolation. The `int8 (matmul + embedding)` variant is what `predict.py`
loads by default.

### Architecture — a modern-small decoder, not a GPT-2 reimplementation

- **RMSNorm** — no bias term, cheaper than LayerNorm
- **RoPE** — relative positions baked into attention via rotation, no
  learned position embedding table to store
- **SwiGLU** — gated feed-forward block, better quality per parameter
  than a plain GELU MLP
- **Grouped-query attention** — 8 query heads, 2 KV heads (4:1); shrinks
  the KV cache versus full multi-head attention
- **Tied embeddings** — the input embedding doubles as the output head,
  saving 1.54M parameters on a 6k vocabulary at this scale — the single
  biggest parameter-count lever available at this size

### Why this shape, and why these numbers are smaller than a bigger write-up's

If you've seen a larger writeup in this style report **top-5 ≈ 59-61%**
after **16,000 steps on a GPU/MPS accelerator (hours of wall-clock)** on
a much larger corpus and vocabulary — that's a materially bigger compute
and data budget than this project used, not a different technique. The
same architecture family (RoPE/SwiGLU/GQA/RMSNorm/tied embeddings), the
same measurement-driven quantization gate, and the same evaluation
methodology (held-out perplexity + top-k accuracy) are all here; what's
different is **1,000 steps on one CPU core against a 2.1M-token corpus**
versus **16,000 steps on an accelerator against a much bigger one.**
Model capacity, step count, and data are the actual levers behind that
gap — not a missing trick. `gpt/train.py` is fully resumable
(`python train.py --steps N` continues from the last checkpoint), so
pointing it at more data and letting it run longer on better hardware is
the direct path from these numbers to that report's.

## Technology stack

| Layer | Technology | Why |
|---|---|---|
| LSTM model | **TensorFlow / Keras** | Standard, well-documented deep learning framework for sequence models; `Tokenizer`/`pad_sequences` utilities handle text preprocessing cleanly. |
| GPT model (from scratch) | **PyTorch, no other ML dependency** | Full control over the architecture (RoPE/SwiGLU/GQA/RMSNorm/tied embeddings) — none of that is a stock `nn.Module`, so it's written directly against `torch.nn.functional`. |
| Tokenizer (GPT engine) | **Hugging Face `tokenizers`** | Fast, Rust-backed byte-level BPE trainer/encoder — the same tokenization family GPT-2 and its descendants use. |
| Quantization (GPT engine) | **`torch.quantization.quantize_dynamic` + a hand-rolled int8 embedding** | Dynamic quantization covers every `nn.Linear`; the tied embedding needed a manual symmetric int8 scheme since dynamic quantization doesn't touch `nn.Embedding`. |
| Transformer model | **PyTorch + Hugging Face `transformers`** | Gives direct access to thousands of pretrained masked-language models (DistilBERT by default, swappable to BERT/RoBERTa) with a few lines of code. |
| Backend API | **FastAPI** | Async, typed, automatic request validation via Pydantic, minimal boilerplate, serves both static files and the JSON `/predict` endpoint. |
| Server | **Uvicorn** | ASGI server that runs the FastAPI app. |
| Frontend | **Vanilla HTML / CSS / JavaScript** | No build step, no framework overhead — a single-purpose UI doesn't need one; keeps the whole project runnable with just `pip install` and no `npm`. |
| Data interchange | **JSON over HTTP** | Simple `POST /predict {text, engine, top_k}` → `{suggestions: [...]}` contract, easy to call from anything (curl, Python, the bundled UI). |
| Training corpora | **Public-domain Shakespeare + public-domain novels** | ~1.1MB for the LSTM, ~7.5MB (Shakespeare + several novels) for the GPT engine's larger, more diverse vocabulary needs — all license-free to redistribute. |

## Project structure

```
nwp_project/
├── data/
│   ├── corpus.txt              LSTM training corpus (Shakespeare, ~1.1MB, public domain)
│   └── corpus_large.txt        GPT training corpus (+ public-domain novels, ~7.5MB)
├── lstm/
│   ├── train_lstm.py           retrain (or train on your own text) whenever you want
│   ├── predict_lstm.py         load the trained model and predict
│   └── model/                  ALREADY TRAINED AND INCLUDED — ready to use out of the box
│       ├── lstm_model.h5       trained weights (10 epochs on the full corpus)
│       ├── tokenizer.pkl       fitted vocabulary
│       └── config.json         sequence length / vocab size metadata
├── gpt/
│   ├── model.py                 the modern decoder architecture (RoPE/SwiGLU/GQA/RMSNorm)
│   ├── train_tokenizer.py       trains the byte-level BPE tokenizer
│   ├── prepare_data.py          tokenizes the corpus, writes train/val/test .bin files
│   ├── train.py                 resumable training loop
│   ├── evaluate.py              loss / perplexity / top-1/3/5 accuracy on a split
│   ├── quantize.py              produces + measurement-gates the int8 variants
│   ├── predict.py               loads the shipped checkpoint, predicts whole words
│   └── model/                  ALREADY TRAINED, QUANTIZED, AND INCLUDED
│       ├── tokenizer.json               the trained BPE tokenizer
│       ├── gpt_config.json              architecture hyperparameters
│       ├── gpt_model_fp32.pt            fp32 weights
│       ├── gpt_model_int8.pt            int8, matmul-only
│       ├── gpt_model_int8_full.pt       int8, matmul+embedding — SHIPPED BY DEFAULT
│       ├── shipped_variant.json         which file predict.py loads by default
│       ├── quantization_report.json    the full fp32 vs int8 comparison
│       └── train.bin / val.bin / test.bin   tokenized corpus splits
├── transformer/
│   └── predict_transformer.py  pretrained DistilBERT predictor, no training needed
├── app/
│   ├── main.py                 FastAPI server, exposes /predict
│   └── static/                 the web UI (index.html / style.css / script.js)
├── docs/
│   └── screenshot.png          UI screenshot used above
├── requirements.txt
└── README.md
```

**All three models are already trained (and, for the GPT engine, quantized)
and included** — nothing needs to run before you use them. See
[Results](#results--the-from-scratch-gpt-engine) above for what the GPT
engine's numbers mean and how to improve on them with more
data/steps/hardware.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Run the web app

```bash
cd app
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000**. Type in the box; suggestions appear as you
type or on space, and <kbd>Tab</kbd> inserts the top suggestion. Switch
between "Transformer", "GPT (scratch)", and "LSTM" with the toggle at the
top — all three work immediately, no setup required.

### Call the predictors directly from Python

```python
# Transformer — pretrained, no training needed
from transformer.predict_transformer import TransformerPredictor
p = TransformerPredictor()
print(p.predict("The weather today is", top_k=5))

# GPT (from scratch) — already trained + quantized, ships with the repo
from gpt.predict import GPTPredictor
p = GPTPredictor()
print(p.predict("the weather today is", top_k=5))

# LSTM — already trained, ships with the repo
from lstm.predict_lstm import LSTMPredictor
p = LSTMPredictor()
print(p.predict("to be or not to", top_k=5))
```

All three return the same shape:
```python
[{"word": "sunny", "probability": 0.41}, {"word": "cold", "probability": 0.18}, ...]
```

### Call the API directly

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I would like to", "engine": "gpt", "top_k": 5}'
```

`engine` is one of `"transformer"`, `"gpt"`, or `"lstm"`.

### Retrain the GPT engine (on your own text, or just for longer)

```bash
cd gpt
python train_tokenizer.py --data ../data/corpus_large.txt --vocab-size 6000   # once
python prepare_data.py --data ../data/corpus_large.txt                       # once
python train.py --steps 1000 --total-steps 3000                              # resumable
python evaluate.py --split test                                             # honest numbers
python quantize.py                                                          # re-quantize + re-gate
```

`train.py` is fully resumable — re-running `python train.py --steps N`
continues from the last checkpoint rather than restarting, which is what
makes training in short bursts (e.g. on a laptop, between other things)
practical. Point `--data` at any UTF-8 `.txt` file to train on your own
text instead; with a GPU, the same code trains far faster and will get
much closer to (or past) the numbers cited in
[Results](#results--the-from-scratch-gpt-engine).

### Retrain the LSTM on your own text

```bash
cd lstm
python train_lstm.py --data ../data/corpus.txt --epochs 30
```

Point `--data` at any UTF-8 `.txt` file — chat exports, notes,
documentation, a specific author's work, anything — one sentence per line
works best. This overwrites `lstm/model/`. For a quick test on a smaller
slice first:

```bash
head -c 200000 ../data/corpus.txt > ../data/corpus_small.txt
python train_lstm.py --data ../data/corpus_small.txt --epochs 10
```

### Swap in a bigger/better pretrained transformer

In `transformer/predict_transformer.py`, change:
```python
MODEL_NAME = "distilbert-base-uncased"
```
to `"bert-base-uncased"`, `"roberta-base"`, or `"roberta-large"` for higher
quality at the cost of speed/memory. Any Hugging Face masked-language-model
checkpoint is a drop-in replacement.

## Design notes

For context, here's how the three implemented engines compare to other
common approaches, roughly in order of sophistication:

1. **Markov chains / n-gram frequency tables** *(not implemented — noted
   for completeness)* — simplest possible approach, no machine learning:
   just "what word most often followed this word (or pair of words) in
   the training text." No real sense of meaning, but instant and requires
   no model.
2. **Word-level LSTM** *(implemented, `lstm/`)* — a small recurrent
   network learns a probability distribution over the vocabulary from
   sequential context. Small, fast, and trainable on modest hardware and
   modest data — but limited to what it has seen.
3. **From-scratch modern decoder transformer** *(implemented, `gpt/`)* —
   the same architecture family used in current small/efficient LLMs
   (RoPE, SwiGLU, grouped-query attention, RMSNorm, tied embeddings),
   trained, evaluated, and quantized end to end in this project. More
   capable per parameter than the LSTM, and — because it's yours — scales
   with however much data and compute you're willing to give it.
4. **Pretrained transformer masked-language-model** *(implemented,
   `transformer/`)* — reuses a model already trained on a huge corpus, so
   it understands grammar and context far better with zero training on
   your end, at the cost of size and inference speed.
5. **Federated learning** *(not implemented — noted for completeness)* —
   a research-level extension where the model trains across many users'
   devices without raw text ever leaving the device, aggregating only
   model updates. Genuinely more complex (needs a client/server
   simulation, secure aggregation, etc.); out of scope for a single
   runnable project like this one, but a natural next step if privacy-
   preserving personalization is the goal.
