[English](README.md) | [简体中文](README.zh.md)

# Mini NanoGPT 🚀

## This is the original *Mini nanoGPT* project, which is no longer maintained.

**The new version offers a more modern, real-time loss-curve plotting UI and capabilities, drastically reducing the time required for model training.**

**It also introduces a database to store information, making the model easier to manage, enabling storage of historical training data, and allowing you to switch between and load different training sessions at any time.**

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone --depth 1 --branch old https://github.com/ystemsrx/mini-nanoGPT.git
cd mini-nanogpt

# Install dependencies (Python 3.7+)
pip install -r requirements.txt
```

### 2. Launch the Project

```bash
python main.py
```

Open your browser and navigate to the URL shown in the terminal (usually [http://localhost:7860](http://localhost:7860)) to access the training interface!

## 🎮 Usage Guide

### Step 1: Prepare Your Data

1. Go to the **Data Processing** page.
2. Select or paste your training text and choose a tokenization method. For better results, check “Use tokenizer” to automatically build a vocabulary based on your text.
3. If you don’t want to use a validation set for now, check “Skip validation set.”
4. Click **Start Processing** when you’re ready.

Here’s a small example:

![image](https://github.com/user-attachments/assets/667d1fb4-9f9a-4d3a-8574-894be7c71bc6)

### Step 2: Train the Model

1. Switch to the **Training** page. Adjust parameters as needed (the defaults are fine for a quick test).
2. The UI will display live loss curves for both the training set and validation set. If you created a validation set in Step 1, you’ll see two curves: blue for training loss and orange for validation loss.
3. If you only see one curve, check the console for an error like:

   ```txt
   Error while evaluating val loss: Dataset too small: minimum dataset(val) size is 147, but block size is 512. Either reduce block size or add more data.
   ```

   This means your block size is larger than your validation set—try reducing it (e.g., to 128).
4. Once both curves appear and update dynamically, click **Start Training** and wait for the process to complete.

![image](https://github.com/user-attachments/assets/61b1f55e-5a9e-45e4-af9e-0c58f8a2be7e)

#### Evaluation-Only Mode

To run in evaluation-only mode (i.e., compute loss on validation set without training), set **Number of Evaluation Seeds** to a positive integer. The UI will then show the loss for different random seeds.

### Step 3: Generate Text

1. Go to the **Inference** page.
2. Enter a prompt (starting text).
3. Click **Generate** to see what the model writes!

![image](https://github.com/user-attachments/assets/dcebc48a-69c2-4008-b6b4-3fec060a75fb)

## 📁 Project Structure

```
mini-nanogpt/
├── main.py          # Entry point
├── config/          # Configuration files
├── data/            # Data processing and storage
├── modules/         # Model definitions, inference & API
└── trainer/         # Training logic
```

## 📝 License

This project is released under the [MIT License](LICENSE).

---

🎉 **Start your GPT journey now!**
