# 🧠 AI Sentiment Analysis

This repository contains three sentiment classification models developed for the course **Artificial Intelligence II – Deep Learning for NLP** (Spring 2024–2025). The models classify English-language tweets into positive or negative sentiment using increasingly advanced techniques.

---

## 📂 Project Structure

```
AI-Sentiment-Analysis/
├─ Assignment1/                # TF-IDF + Logistic Regression
│  ├─ instructions.pdf    
│  ├─ report.pdf
│  └─ tfidf-notebook.ipynb
├─ Assignment2/                # Neural Network + Word2Vec
│  ├─ instructions.pdf    
│  ├─ report.pdf
│  └─ neural-networks-notebook.ipynb
├─ Assignment1/                # Fine-tuned Transformers (BERT & DistilBERT)
│  ├─ instructions.pdf    
│  ├─ report.pdf
│  ├─ distilbert-notebook.ipynb
│  └─ bert-notebook.ipynb
├─ README.md                   # Project overview and documentation
```
**Reports** within the assignments are documents that include details on data processing steps, such as preprocessing, analysis, and vectorization, followed by experiments, hyperparameter tuning, optimization techniques, and evaluation. Each report concludes with an overall analysis of the results and a summary of the best-performing trials, including a comparison with previous approaches (assignments).

**Instructions** refer to the official assignment guidelines provided for each task.

The **.ipynb files** are notebooks that include the code, experiments, and the process leading to the final selected model for each assignment.

---

## 📝 Assignments Overview

### 🔹 Assignment 1: TF-IDF + Logistic Regression
A classical machine learning approach using TF-IDF vectorization and logistic regression. Serves as a strong baseline for sentiment classification.

### 🔹 Assignment 2: Neural Network + Word2Vec
A feedforward neural network built with PyTorch, leveraging Word2Vec embeddings for semantic understanding of tweet content.

### 🔹 Assignment 3: Fine-tuned Transformers (BERT & DistilBERT)
State-of-the-art transformer-based models fine-tuned using HuggingFace’s `transformers` library for high-accuracy sentiment classification.

---

## 🧰 Technologies Used

- Python ≥ 3.8
- `scikit-learn`
- `PyTorch`
- `HuggingFace Transformers`
- `gensim`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
