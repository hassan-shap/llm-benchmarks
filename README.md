# LLM Benchmarks

This repository provides tools and scripts for benchmarking various open-weight pretrained large language models (LLMs) using the Hugging Face API. The benchmarks are designed to evaluate model performance across a variety of tasks, including cosine similarity analysis and model evaluations on benchmark datasets like BoolQ and MMLU.

## Overview

The main goal of this repository is to offer a streamlined interface for researchers and practitioners to assess the performance of open-weight pretrained LLMs. Specifically, the repository contains:

1. **Cosine Similarity Analysis**:
   - Code to compute and visualize cosine similarity plots, reproducing **Figure 1(c)** and **Figure 4** from the paper [*Open-Weight LLM Benchmarks*](https://arxiv.org/abs/2403.17887).
  
2. **Model Evaluations**:
   - Evaluation scripts for the following datasets:
     - **BoolQ**: A question-answering benchmark.
     - **MMLU**: The Massive Multitask Language Understanding benchmark.

## Pretrained LLMs via Hugging Face API

All pretrained models used in this repository are accessed through the [Hugging Face API](https://huggingface.co/models), ensuring up-to-date and readily accessible model weights. The evaluation pipeline supports any model available on Hugging Face, including well-known models like `GPT-Neo`, `T5`, and `BERT`.

## Getting Started

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/hassan-shap/llm-benchmarks
   cd llm-benchmarks

@article{hassan2024llm,
  title={Open-Weight LLM Benchmarks},
  author={Hassan, Shap et al.},
  journal={arXiv preprint arXiv:2403.17887},
  year={2024}
}
