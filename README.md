# Understanding Large Language Models: Visualization and Training Concepts

<!--- Removed top badges for less redundancy -->

## Overview ✨

Welcome! This repository offers a two-part guide designed to demystify the internal workings and training lifecycle of modern Large Language Models (LLMs), focusing on the Transformer architecture. We aim to bridge the gap between abstract concepts and concrete examples by visualizing a real model's parameters and explaining how such models learn.

*   **Part 1** explores and visualizes the architecture, parameters, and dynamic attention mechanisms of the `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` model to build intuition.
*   **Part 2** provides a conceptual overview of the LLM training lifecycle, including pre-training, fine-tuning strategies (SFT, Alignment with RLHF/GRPO/DPO), knowledge distillation, and parameter-efficient techniques (PEFT/LoRA).

## Target Audience 🎯

This guide is intended for:

*   Students learning about AI, Machine Learning, and Natural Language Processing (NLP).
*   Developers curious about the models they interact with.
*   Researchers looking for practical ways to inspect model internals or understand training paradigms.
*   Anyone seeking a deeper understanding of how LLMs function and learn.

Basic familiarity with Python is assumed. Key concepts are explained within the notebooks.

## Content Overview 📚

This guide covers the following key areas across two notebooks:

**Part 1: Architecture & Visualization (`LLM_Architecture_Visualization.ipynb`)**
*   **Foundations:** Core ML/ANN concepts, parameters.
*   **Input:** Tokenization, Token Embeddings (visualized).
*   **Transformer Blocks:** Self-Attention (QKV, Multi-Head, GQA context), Position-wise Feed-Forward Networks (FFN using SwiGLU), Layer Normalization (RMSNorm), Residual Connections (components visualized for Layer 0, Middle, Last).
*   **Output:** Final Normalization, Language Modeling Head (visualized, weight tying checked).
*   **Analysis:** Parameter statistics across layers, dynamic attention pattern heatmaps, aggregate weight visualizations (Q, K, V, O, FFN projections across all layers).

**Part 2: Training & Fine-tuning Concepts (`Understanding_LLM_Training_Lifecycle_Part2.ipynb`)**
*   **Pre-training:** Building foundational knowledge (Next-Token Prediction).
*   **Knowledge Distillation:** Context for the specific `DeepSeek-R1-Distill` model.
*   **Fine-tuning:** Supervised Fine-tuning (SFT) / Instruction Tuning.
*   **Alignment Tuning:** Concepts of RLHF/PPO, GRPO, DPO.
*   **Efficiency:** Parameter-Efficient Fine-tuning (PEFT), focusing on LoRA.
*   **Tools:** Overview of relevant Hugging Face libraries (`datasets`, `transformers`, `peft`, `trl`) and practical examples (like Unsloth).

## Notebooks

1.  **Part 1: Architecture & Visualization**
    *   **File:** `LLM_Architecture_Visualization.ipynb`
    *   **Focus:** Dissects the architecture of `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` and provides code to visualize its parameters and attention mechanisms.
    *   [![Open Part 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hercules45/Understanding-LLM-Internals/blob/main/LLM_Architecture_Visualization.ipynb)

2.  **Part 2: Training & Fine-tuning Concepts**
    *   **File:** `Understanding_LLM_Training_Lifecycle_Part2.ipynb`
    *   **Focus:** Provides a conceptual explanation of the LLM training lifecycle relevant to the model in Part 1. Contains no runnable training code.
    *   [![Open Part 2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hercules45/Understanding-LLM-Internals/blob/main/Understanding_LLM_Training_Lifecycle_Part2.ipynb)
    

## Model Used (for Visualization in Part 1)

*   **Model ID:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
*   **Link:** [Hugging Face Model Card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)

## How to Use

1.  **Open in Colab:** Click the "Open In Colab" badges in the **Notebooks** section above.
2.  **Select Runtime (Part 1):** Use a `GPU` accelerator in Colab (`Runtime` -> `Change runtime type` -> `T4 GPU`) for Part 1 for best performance with model loading and visualization. Part 2 is conceptual.
3.  **Run Cells Sequentially:** Execute the notebook cells in order. 
4.  **Explore:** Read the explanations and observe the generated outputs and visualizations in Part 1. *Note: The aggregate weight plots near the end of Part 1 can be very resource-intensive (RAM/CPU) and may take significant time to render or cause slowdowns.*

## Requirements (for Part 1)

*   Python libraries: `transformers`, `torch`, `accelerate`, `matplotlib`, `seaborn`, `numpy`. (Installed by the notebook).
*   Internet connection (for model download).
*   Sufficient RAM (>= 12GB recommended) and GPU VRAM (>= 8GB recommended).

## Limitations

*   **Part 1 Focus:** Primarily architecture, parameters, attention visualization. Excludes runnable training, activation analysis. Aggregate plots are resource-heavy.
*   **Part 2 Focus:** Conceptual explanations only; no runnable training code.
*   **Model Specificity:** While core concepts are general, some implementation details relate to the specific Qwen/DeepSeek model.


