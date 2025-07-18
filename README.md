# UF-FAE: Union-Find based Feature-Augmented Embedding for Scalable Anti-Money Laundering in Blockchain Technology

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

UF-FAE is a unified algorithmic framework that combines **Union-Find with Path Compression** and **machine learning models** to detect money laundering behavior on large-scale **blockchain transaction graphs**. This project is designed to enhance the **throughput (TPS)** and **security** of IOTA-like DAG structures by introducing fast, interpretable graph-based features for downstream AML classification tasks.

---

## 🔍 Motivation

Despite increasing adoption of blockchain technologies, financial institutions still face the **"data silos"** problem, which limits cross-institutional surveillance. Moreover, traditional AML detection relies on rules or static transaction features, ignoring **network structures**.

We propose **UF-FAE**, an efficient and interpretable pipeline leveraging **graph theory, distributed ledgers, and AI** to detect suspicious financial flows in decentralized settings.

---

## 🧠 Key Concepts

- **Union-Find with Path Compression**: Efficient disjoint-set algorithm used for grouping accounts into weakly connected components (WCCs).
- **Graph Feature Aggregation**: Extract group-level and node-level statistics such as degree, betweenness, closeness, and bidirectionality.
- **Multi-modal ML Pipeline**: Combine graph structure with transactional metadata for classification via Logistic Regression, Decision Tree, Random Forest, and SVM.
- **Time-aware Evaluation**: Simulates real-world constraints with time-based data splits (no future leakage).

---

## 🗂 Dataset

- **Name**: [SAML-D](https://www.kaggle.com/datasets/oztasyusuf/synthetic-aml-data-saml-d)  
- **Size**: ~9.5 million transactions  
- **Label**: `Is_laundering` (binary)  
- **Features**: 12 fields including `Sender_account`, `Receiver_account`, `Amount`, `Time`, `Payment_type`, etc.

---
