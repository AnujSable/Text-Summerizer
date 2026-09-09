# Summarizer-HF
Text Summarizer App - Transformer Minor Project (using HuggingFace &amp; FastAPI)
# Abstractive Dialogue Summarization with T5

An end-to-end Natural Language Processing (NLP) pipeline that fine-tunes a Sequence-to-Sequence (Seq2Seq) Transformer model (`t5-small`) to automatically summarize multi-turn conversations into concise, structured key points. Built with PyTorch, Hugging Face Transformers, and Accelerated computing.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Model Architecture](#model-architecture)
- [Dataset Specifications](#dataset-specifications)
- [Repository Structure](#repository-structure)
- [Environment Setup & Installation](#environment-setup--installation)
- [Data Preprocessing & Tokenization](#data-preprocessing--tokenization)
- [Training & Fine-Tuning Pipeline](#training--fine-tuning-pipeline)
- [Model Performance & Training Logs](#model-performance--training-logs)
- [Inference & Usage](#inference--usage)
- [Future Enhancements](#future-enhancements)

---

## Project Overview

In multi-turn chat conversations (e.g., customer support chats, team messaging, interview transcripts), extracting key information manually is time-consuming. Extractive summarization often fails because spoken dialog contains informal phrasing and fragmented sentences. 

This project implements an **Abstractive Dialogue Summarization** solution using `t5-small` fine-tuned on the **SAMSum** dataset. The model learns to rephrase, synthesize, and extract intent directly from unstructured dialogues into coherent, human-readable summaries.

---

## Key Features

- **Text Cleaning Pipeline**: Cleans multi-line chat strings, strips residual HTML, normalizes white spaces, and converts text to uniform casing.
- **Dynamic Device Selection**: Automatically detects and leverages `cuda` (NVIDIA GPUs), `mps` (Apple Silicon GPUs), or `cpu` fallback.
- **Hugging Face Trainer Engine**: Employs Hugging Face's `Trainer` class integrated with `accelerate` for optimized gradient updates, warmup schedules, and checkpoint saving.
- **Configurable Generation Parameters**: Uses Beam Search (`num_beams=4`) during inference to generate fluent summaries while preventing repetitive loops.
- **Model Checkpointing**: Automatically exports trained model weights, configuration, and vocabulary to disk for deployment.

---

## Model Architecture

The core model is **T5 (Text-to-Text Transfer Transformer)**, specifically `t5-small` (60M parameters). 

- **Encoder-Decoder Architecture**: Encoder processes input dialogue tokens; Decoder auto-regressively generates output summary tokens.
- **Pretrained Weights**: Leveraging standard T5 weights initialized from pre-training on the C4 dataset.
- **Token Embeddings**: Shared vocabulary space (`32,128` tokens) mapped across 512-dimensional hidden layers.

---

## Dataset Specifications

The pipeline uses the **SAMSum Dataset**, containing over 16,000 messenger-style conversations written by fluent English speakers along with corresponding human-written summaries.

- **Data Fields**:
  - `id`: Unique identifier for each conversation.
  - `dialogue`: Unstructured raw chat transcripts (e.g., `"Amanda: Do you want cookies?\r\nJerry: Sure!"`).
  - `summary`: Reference ground-truth summary.
- **Data Split & Sampling**:
  - **Full Dataset Size**: ~14,732 training samples and 818 validation samples.
  - **Sampled Subset (Fast Experimentation)**: 4,000 training conversations and 500 validation conversations.

---

## Repository Structure

```text
├── text_summarizer.ipynb      # Main Jupyter Notebook (data cleaning, training, inference)
├── samsum-train.csv           # Raw training dataset
├── samsum-validation.csv      # Raw validation dataset
├── saved_summary_model/       # Saved fine-tuned weights and tokenizer files
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── README.md                  # Project documentation
