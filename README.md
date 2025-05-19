[English](README.md) | [简体中文](README.zh.md)

# Mini-NanoGPT - `database` Branch

## 📌 Project Overview

**Mini-NanoGPT** is a lightweight visual training platform based on Karpathy’s [nanoGPT](https://github.com/karpathy/nanoGPT), designed to help deep learning beginners, researchers, and developers quickly grasp the GPT model training workflow through an intuitive graphical interface.

This `database` branch builds upon the original features by introducing **model management database support**, enabling unified storage and tracking of model metadata, training configurations, inference parameters, and execution history — greatly enhancing systematic and scalable experiment management.

---

## 🚀 Branch Highlights: Database Features

### ✅ Model Registration & Persistent Tracking

* Introduces an **SQLite database**. A `DBManager` component automatically assigns a unique ID to each new model and records metadata such as model name, creation time, and file path.
* Centralized management of all model metadata, enabling easy lookup and state maintenance.

### ✅ Configuration, Execution, and Files

* Automatically stores **hyperparameter configurations**, **log paths**, **inference parameters**, and **generation history** during training.
* Supports parameter rollback, experiment resumption, and auto-filled UI forms for reproducible training.
* Automatically creates structured directories (e.g., `out/{model_name}_{model_id}`, `data/{model_name}_{model_id}`).

### ✅ Visual Model Management Interface

* A new **Model Management** tab added to the frontend:

  * Visual browsing of all models (Name + ID)
  * Supports adding, switching, refreshing, and deleting models — all operations are fully synchronized with the database.
* All frontend actions use `DBManager` to maintain consistency between backend data and UI state.

> ⚠️ Note: This branch only adds model database and management functionality. The training and inference processes remain unchanged — see the main branch documentation for details.

---

## 🧪 Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Model registration, configuration storage, and related operations during training and inference are implemented in the `DBManager` source file.

---
