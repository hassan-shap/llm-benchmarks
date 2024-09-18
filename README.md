# LLM Benchmarks

This repository contains scripts for analyzing and evaluating various open-weight pretrained large language models (LLMs) using the Hugging Face API. The results are presented in [*The Unreasonable Ineffectiveness of the Deeper Layers*](https://arxiv.org/abs/2403.17887)

## Overview

We have two components here:

1. **Cosine Similarity Analysis**:
   - Code to compute and visualize cosine similarity plots, reproducing **Figure 1(c)** and **Figure 4** of the paper.
  
2. **Model Evaluations**:
   - Evaluation scripts for the following datasets:
     - **BoolQ**: A question-answering benchmark.
     - **MMLU**: The Massive Multitask Language Understanding benchmark.

## Pretrained LLMs via Hugging Face API

All pretrained models used in this repository are accessed through the [Hugging Face API](https://huggingface.co/models), ensuring up-to-date and readily accessible model weights. The evaluation pipeline supports any model available on Hugging Face.

## Citation
If you use ideas from our paper in your work, please cite it as follows:

@article{,
  title={The Unreasonable Ineffectiveness of the Deeper Layers},
  author={A. Gromov, K. Tirumala, H. Shapourian, P. Glorioso, D. A. Roberts},
  journal={arXiv preprint arXiv:2403.17887},
  year={2024}
}
