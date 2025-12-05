# Neuro-Symbolic Fake News Detector (Part 1)

> **Project Context:** AIROST UTM Internship Project (Part 1 of 2)  
> **Status:** Active Research Prototype  
> **Architecture:** Dual-Encoder DistilBERT + Cross-Attention + Live Web Search  
> **Performance:** 70.0% Validation Accuracy on LIAR Benchmark  
> **Training Scale:** 5,885 Samples (30% of Data - Resource Optimized)

![Neuro-Symbolic Architecture Diagram](https://github.com/user-attachments/assets/7547a8e8-0687-47a2-aac7-2cc38c838fae)
*(Figure: Technical Schematic of the Dual-Encoder Attention Network)*

---

## 📖 Executive Summary
Developed as part of my internship initiative at **AIROST UTM**, this project implements a **Neuro-Symbolic AI system** designed to verify claims by cross-referencing them with real-time evidence fetched from the web.

Current state-of-the-art models often act as "Black Boxes," memorizing the training data without understanding external context. I addressed this failure mode by creating an agentic system that mimics human fact-checking:
1.  **Reads the Claim** (Neural perception).
2.  **Searches the Web** for live evidence (Symbolic tool use).
3.  **Reasons** about the relationship between the claim and the evidence using a custom Cross-Attention Mechanism.

---

## 📚 Primary Research Papers (The Theory)
My architecture is not arbitrary; it is grounded in recent advancements in hybrid AI systems. I utilized the following papers to build the theoretical foundation for my implementation:

### A. The Architecture Blueprint
* **Paper:** *Fake news detection based on shared information and cross-feature ambiguity learning (SI-CFAL)*
* **Authors:** Cui, ShaoDong, et al. (2025)
* **Implementation:** This paper provided the core concept of **"Ambiguity Learning."** I implemented this via my **Fusion Layer**, which dynamically weighs the "Neural" signal (Text Semantics) against the "Symbolic" signal (Evidence/Context). This allows the model to resolve conflicts, preventing it from relying solely on one modality.

### B. The Dataset Source
* **Paper:** *An Enhanced Fake News Detection System With Fuzzy Deep Learning*
* **Authors:** Xu, Cheng, and M-Tahar Kechadi (2024)
* **Implementation:** I utilized their **LIAR2 Dataset** structure. Specifically, I analyzed the metadata columns (e.g., `pants_on_fire_counts`, `mostly_true_counts`) to understand the distribution of "Symbolic Context." This guided my decision to strip the "Justification" column (which causes data leakage) and replace it with live web search results to ensure the model learns robust features rather than memorizing annotator explanations.

### C. The Baseline Validation
* **Paper:** *Fake News Detection Using Machine Learning and Deep Learning Algorithms: A Comprehensive Review*
* **Authors:** Alshuwaier, Faisal A., and Fawaz A. Alsulaiman (2025)
* **Implementation:** This comprehensive review identified **"Black Box Models"** and **"Static Data"** as the two critical failure points in current AI fact-checking. I used this insight to justify my **Neuro-Symbolic approach** (White Box logic via Attention maps) and **Live Search** (Dynamic Data), ensuring the model remains relevant even as news events evolve daily.

---

## 🧠 The Architecture: Evidence-Aware Dual-Encoder
I utilized a **Dual-Encoder Architecture** with a custom **Cross-Attention Fusion Layer**.

### 1. Symbolic Pillar (The "Eyes")
A `SmartSearcher` module uses the **DuckDuckGo API** to fetch real-time context for every claim. 
* **Dynamic Information Retrieval:** Unlike static datasets, my model sees the web as it exists *today*.
* **Caching Mechanism:** To handle the latency of web requests, I implemented a persistent JSON-based caching layer (`evidence_cache_v3.json`) that stores retrieval results, preventing API rate limits and accelerating training epochs.

### 2. Neural Pillar (The "Brain")
I used **DistilBERT** (a distilled version of BERT) as the backbone for semantic understanding.
* **Claim Encoder:** A distinct BERT stream encodes the input statement into high-dimensional vectors.
* **Evidence Encoder:** A separate BERT stream processes the retrieved web snippets. This separation ensures the model treats the claim and evidence as distinct entities before fusion.

### 3. Cognitive Fusion (The "Logic")
Instead of simple concatenation, I implemented a **Multi-Head Cross-Attention Layer**:
* **Query ($Q$):** The Claim Vector.
* **Key/Value ($K,V$):** The Evidence Vector.
* **Mechanism:** The model calculates attention scores to determine which specific parts of the retrieved evidence correspond to the claim. This filters out noise from the search results and focuses the decision-making on relevant proof.

---

## 📊 Data & Computational Constraints

### The Dataset: LIAR Benchmark
I trained the model on the **LIAR Dataset**, a collection of 12.8K manually labeled short statements from PolitiFact. The data covers various subjects including **economy, healthcare, taxes, federal budget, and education**.

### 📉 Exploratory Data Analysis (EDA) & Cleaning
Before training, I performed extensive EDA to understand the data distribution. I discovered two critical issues that required intervention:

1.  **The Ambiguity Problem (Removing "Half-True"):**
    * **Discovery:** A significant portion of the dataset was labeled "Half-True."
    * **Action:** I removed this category entirely.
    * **Reasoning:** "Half-True" statements contain elements of truth mixed with falsehoods. Training a binary classifier on such ambiguous data confuses the decision boundary, leading to "hallucinations" or low confidence. By removing it, I sharpened the model's focus on clear **Real (True/Mostly True)** vs **Fake (False/Pants on Fire)** signals.

2.  **Class Imbalance (The "Liar" Bias):**
    * **Discovery:** The dataset was heavily skewed toward "False" claims (Ratio approx 2:1).
    * **Impact:** Without intervention, the model would simply guess "Fake" every time and achieve ~60% accuracy without learning anything.
    * **Action:** I implemented **Weighted Cross-Entropy Loss** ($PosWeight=2.0$). This penalized the model 2x more for missing a "Real" claim, forcing it to actually learn the features of truth rather than playing the odds.

### Resource Optimization (The 30% Constraint)
For this specific demonstration, I strictly curated the training set to a high-quality subset of **5,885 examples** (approximately **30%** of the full dataset).
* **Reasoning:** This constraint was intentional to optimize for available computational resources (GPU memory and training time) while preserving statistical significance. 
* **Impact:** Even with this 30% subset, the model achieved **70% accuracy**, demonstrating that the **Neuro-Symbolic architecture** is highly data-efficient. It learns to reason with fewer examples because it has access to external knowledge (the internet) rather than relying solely on memorizing patterns within a massive text corpus.

---
# 📉 Deep Learning Fake News Classifier (Part 2)

> **Project Context:** AIROST UTM Internship Project (Part 2 of 2)  
> **Status:** Complete  
> **Architecture:** Word2Vec (Static Embeddings) + LSTM (Recurrent Neural Network)  
> **Performance:** 99.2% Accuracy on ISOT/Kaggle Dataset

![LSTM Architecture Diagram](https://github.com/user-attachments/assets/3747ae61-baf3-4d40-af5b-1b2dcc58438c)
*(Figure 2: Schematic of the Sequential LSTM Architecture with Pre-trained Embeddings)*

---

## 📖 Executive Summary
As the second phase of my internship at **AIROST UTM**, I developed a pure Deep Learning classifier focused on **Semantic Pattern Recognition**.

While Part 1 (Neuro-Symbolic) focused on *external verification* via web search, this model (Part 2) focuses on *internal consistency*. It analyzes the linguistic style, word choice, and sequential structure of news articles to detect patterns characteristic of misinformation (e.g., sensationalism, emotional language) without needing internet access.

---

## 🧠 The Architecture: Semantic Sequential Learning
This model moves away from "Bag of Words" approaches and utilizes **Sequence Modeling** to understand the context of an entire article.

### 1. Vector Space Modeling (Word2Vec)
Instead of training embeddings from scratch, I implemented a **Domain-Specific Word2Vec** model.
* **Training:** The Word2Vec model was trained on the entire corpus (~45,000 articles) to learn the specific vocabulary of political news.
* **Dimensionality:** Words are converted into 100-dimensional dense vectors ($d=100$).
* **Freeze Strategy:** These weights were injected into the Neural Network as a **Non-Trainable (Frozen) Layer**. This ensures the model relies on the established semantic relationships (e.g., "President" is close to "White House") rather than overfitting to specific keywords during the classification phase.

### 2. Sequential Processing (LSTM)
The core engine is a **Long Short-Term Memory (LSTM)** network.
* **Why LSTM?** News articles are long sequences. Traditional RNNs suffer from the "Vanishing Gradient" problem, forgetting the start of a sentence by the time they reach the end. LSTMs maintain a "Cell State" (Memory) that allows them to track context across long paragraphs.
* **Configuration:** 128 Memory Units with a Dropout rate of 0.2 to prevent overfitting.

---

## 📊 Data & Performance

### The Dataset
I utilized the **"Fake and Real News Dataset"** (Clément Bisaillon/ISOT), consisting of approximately **44,000 articles**.
* **True News:** Scraped from Reuters (High journalistic standards).
* **Fake News:** Scraped from flagged misinformation sites (Sensationalist style).

### Preprocessing Pipeline
1.  **Publisher Cleaning:** Removed "Reuters" metadata tags to prevent the model from learning "Source Bias" shortcuts.
2.  **Noise Removal:** Stripped special characters and URL artifacts.
3.  **Tokenization:** Converted text to integers, padded to a sequence length of **1,000 tokens** to capture the full body of most articles.

### 📈 Results
The model achieved near-perfect separation between the two classes on the test set.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **99.26%** |
| **Precision** | 0.99 |
| **Recall** | 0.99 |
| **F1-Score** | 0.99 |

*Note: While the accuracy is exceptionally high, this suggests strong stylistic differences between the "Real" (Reuters) and "Fake" sources in this specific dataset. This model serves as a high-performance baseline for detecting linguistic anomalies.*

---
