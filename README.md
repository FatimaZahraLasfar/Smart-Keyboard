# ⌨️ Smart Keyboard V2 — Interactive N-Gram Playground

An interactive NLP application built with **Python and Streamlit** that explores next-word prediction, language modeling, and text generation using statistical N-gram models trained on the WikiText-2 corpus.

🔗 **Live Demo:** [Smart Keyboard V2](https://smart-keyboard.streamlit.app/)

## 🎯 Overview

Smart Keyboard V2 is an educational project designed to demonstrate how statistical language models can predict the next word in a sequence.

The application implements **unigram, bigram, and trigram models** and provides an interactive environment for experimenting with context, probability estimation, smoothing, and model behavior.

Rather than treating language modeling as a black box, the application allows users to inspect predictions, compare different N-gram orders, generate text, and evaluate sentences using perplexity.

## ✨ Features

### 1. Next-Word Prediction

- Enter a text prompt and receive the four most likely next-word suggestions.
- Display the probability associated with each suggestion.
- Explore the top 10 predicted words through a probability distribution chart.
- Use backoff to fall back to a lower-order model when a higher-order context has not been observed.

### 2. N-Gram Model Comparison

Compare three statistical language models:

- **Unigram:** predicts words without considering previous words.
- **Bigram:** uses one previous word as context.
- **Trigram:** uses two previous words as context.

Experiment with different contexts to understand how additional context affects the predicted probabilities.

### 3. Interactive Text Generation

Generate text using a Markov-chain-style process in which each predicted word becomes part of the context for the next prediction.

- Choose an initial prompt.
- Set the maximum number of words to generate.
- Generate text using the selected model.
- Download the generated text as a `.txt` file.

### 4. Smoothing and Model Configuration

Adjust the N-gram order and the smoothing constant \(k\) directly from the sidebar.

The application uses **add-k (Laplace when \(k=1\)) smoothing** to assign nonzero probabilities to unseen word transitions when \(k>0\).

### 5. Perplexity Evaluation

Evaluate a sentence using perplexity to examine how well the language model predicts its word sequence.

- Calculate sentence perplexity.
- Inspect the proportion of tokens assigned zero probability.
- Experiment with smoothing to observe its effect on the evaluation.

A lower perplexity generally indicates that a model assigns higher probability to the evaluated text.

### 6. Vocabulary Explorer

- Search for a word in the model vocabulary.
- Check whether the word was observed during training.
- Display its frequency in the training corpus.
- Handle out-of-vocabulary words using an unknown-token representation.

### 7. User Interface

- Interactive Streamlit interface.
- Light and dark modes.
- Dynamic visualizations using Altair.
- Cached data and model objects to reduce repeated computation.

## 🧠 How It Works

The application follows a statistical NLP pipeline:

1. **Dataset acquisition:** Load the WikiText-2 raw training corpus from Hugging Face.
2. **Text preprocessing:** Remove empty lines and section headings, split text into sentences, normalize selected formatting artifacts, lowercase text, and tokenize words.
3. **Corpus preparation:** Save the processed sentences in a CSV file grouped by document identifiers.
4. **N-Gram construction:** Count word occurrences and transitions for unigram, bigram, and trigram contexts.
5. **Probability estimation:** Estimate next-word probabilities using observed counts and add-k smoothing.
6. **Prediction:** Rank candidate words by their estimated probabilities.
7. **Backoff:** When using trigram prediction, fall back to a bigram or unigram model if the current context is unavailable.
8. **Evaluation and generation:** Calculate perplexity, explore vocabulary statistics, and generate text interactively.

## 🛠️ Technologies

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| NLTK | Sentence and word tokenization |
| Hugging Face Datasets | Loading the WikiText-2 corpus |
| Pandas | Data handling and tabular representations |
| Altair | Probability distribution visualizations |
| `collections.Counter` | Counting word transitions |
| `math` and `random` | Probability calculations and stochastic text generation |

## 📁 Project Structure

```text
Smart-Keyboard/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignored files and datasets
│
├── src/
│   └── pipeline.py         # N-Gram models and language modeling utilities
│
├── data/
│   ├── load_hf_data.py     # Dataset download and preprocessing
│   ├── corpus_hf.csv      # Generated processed corpus (not committed)
│   └── corpus.csv          # Optional local corpus (not committed)
│
├── results/                # Project results and outputs
└── .devcontainer/          # Development container configuration
```

The generated corpus files are excluded from Git to avoid committing large datasets. The application prepares the WikiText-2 corpus automatically when the processed CSV file is missing.

## 🚀 Installation

### Prerequisites

- Python 3.10 or later recommended.
- Git.
- An internet connection for the initial dataset download.

### 1. Clone the repository

```bash
git clone https://github.com/FatimaZahraLasfar/Smart-Keyboard.git
cd Smart-Keyboard
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**

```cmd
.venv\Scripts\activate.bat
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

**First launch:** The application downloads and preprocesses WikiText-2 if `data/corpus_hf.csv` does not exist. This may take a few minutes and requires internet access.

## 📐 Mathematical Background

### 1. N-Gram Language Models

An N-gram model estimates the probability of a word based on the preceding words.

**General formula:**

<p align="center">
  <code>P(wᵢ | w₁, ..., wᵢ₋₁) ≈ P(wᵢ | wᵢ₋ₙ₊₁, ..., wᵢ₋₁)</code>
</p>

Examples:

- **Unigram:** P(wᵢ)
- **Bigram:** P(wᵢ | wᵢ₋₁)
- **Trigram:** P(wᵢ | wᵢ₋₂, wᵢ₋₁)

### 2. Add-k Smoothing

Add-k smoothing prevents unseen word transitions from automatically receiving a zero probability.

<p align="center">
  <code>P(w | c) = (C(c, w) + k) / (C(c) + k × |V|)</code>
</p>

Where:

- `C(c, w)` is the number of times word `w` follows context `c`.
- `C(c)` is the total number of occurrences of context `c` followed by a word.
- `k` is the smoothing parameter.
- `|V|` is the vocabulary size.

When `k = 1`, the method is called **Laplace smoothing**. When `k = 0`, unseen transitions can receive a probability of zero.

### 3. Perplexity

Perplexity measures how well a language model predicts a sequence of words. Lower perplexity generally indicates better predictive performance on the same evaluation data.

**Formula:**

<p align="center">
  <code>PP(W) = exp(−(1/N) × Σᵢ₌₁ᴺ log P(wᵢ | w₁, ..., wᵢ₋₁))</code>
</p>

Where:

- `N` is the number of predicted tokens.
- `P(wᵢ | w₁, ..., wᵢ₋₁)` is the probability assigned to token `wᵢ` given its preceding context.
- `log` is the natural logarithm.
- `exp` is the exponential function.

**Important:** Perplexity comparisons are meaningful when models use the same evaluation corpus, tokenization, and probability conventions.


## 📚 Dataset

This project uses **WikiText-2**, a language modeling dataset derived from Wikipedia articles.

- **Source:** [WikiText-2 on Hugging Face](https://huggingface.co/datasets/Salesforce/wikitext)
- **Configuration:** `wikitext-2-raw-v1`
- **Split used for corpus preparation:** `train`

The preprocessing script performs sentence segmentation, lowercasing, tokenization, filtering of selected formatting artifacts, and removal of sentences containing fewer than four retained tokens.

The processed corpus is stored locally as `data/corpus_hf.csv` and is generated automatically when needed.

The dataset itself is not included in the Git repository.

## ⚠️ Limitations

- N-gram models rely on local word-count statistics rather than deep semantic understanding.
- Predictions are limited by the training vocabulary and observed contexts.
- Sparse contexts can lead to unreliable estimates.
- Text generation can become repetitive or grammatically inconsistent.
- Results depend on preprocessing, vocabulary size, and smoothing parameters.
- The current implementation is intended primarily for learning and experimentation, not as a production-ready mobile keyboard.

## 🔮 Future Improvements

Potential extensions include:

- Add quantitative evaluation on a held-out test set.
- Compare add-k smoothing with other smoothing techniques, such as Good-Turing or Kneser-Ney.
- Explore interpolated N-gram models alongside backoff.
- Add word-level and sentence-level evaluation reports.
- Improve text generation with repetition controls and stopping conditions.
- Compare statistical models with neural language models.
- Add multilingual support and richer text preprocessing.

## 👩‍💻 Author

**Fatima Zahra Lasfar**

- GitHub: [@FatimaZahraLasfar](https://github.com/FatimaZahraLasfar)
- Project repository: [Smart-Keyboard](https://github.com/FatimaZahraLasfar/Smart-Keyboard)
- Live application: [Smart Keyboard V2](https://smart-keyboard.streamlit.app/)

---

*Built as an NLP learning project to explore statistical language modeling, next-word prediction, and text generation through an interactive application.*
