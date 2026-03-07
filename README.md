# Poetry Author Classification

A machine learning project for classifying poems by author using classical NLP method and transformer-based model.

---

## Project Goals

- Compare classical ML approache with transformer-based model for text classification;

**Learning goals:**

- Practice parameter-efficient fine-tuning with LoRA;
- Organize a hyperparameter search pipeline;
- Explore how much quality can be achieved from a small model trained locally;
- Create a training loop for LoRa manually;
- Prepare the model for optimized inference.

---

## Task Overview

The task is a **multi-class text classification** problem:

> Given a poem, predict its author.

The dataset is relatively small and contains stylistically nuanced texts.  
Evaluation focuses on **macro F1-score** due to class imbalance.

Also I was interested in checking out my own poems in terms of famous poets' styles.

---

## Models

### Baseline

**TF–IDF + Linear SVM**

Used as a strong classical reference point.

---

### Main Model

**BERT-based sequence classifier with LoRA fine-tuning**

Instead of updating all transformer weights, LoRA adapters are trained inside attention layers.

Reasons for this setup:
- Enables training on limited hardware;
- Allows faster experimentation;
- Provides hands-on experience with parameter-efficient fine-tuning.

A small pretrained backbone was intentionally chosen to allow local training and to evaluate how much performance can be extracted from a lightweight model.

---

## Experiments & Results

### Compared Approaches

1. TF–IDF + Linear SVM;
2. BERT encoder with LoRA fine-tuning and hyperparameter search.

---

### Metric

**Macro F1-score** (primary metric due to class imbalance).

---

### Results

| Model                    | Macro F1 |
|--------------------------|----------|
| TF–IDF + Linear SVM      | 0.83     |
| BERT + LoRA              | **0.94** |

---

## Tech Stack

- Python;
- PyTorch;
- Hugging Face Transformers;
- PEFT (LoRA);
- Optuna;
- Scikit-learn.
