# Fake News Detection: Hybrid BERT vs. Baseline Analysis

## 1. Introduction

### Problem Description
* **Motivation:** Disinformation has evolved from simple "clickbait" to sophisticated narratives that mimic professional journalism. Detecting these requires more than keyword matching; it requires deep semantic understanding.
* **Target Audience:** Fact-checkers, AI developers, and social media platforms looking for robust automated verification tools.
* **Benefits:** A high-precision model that can differentiate between stylistic "veneer" and factual legitimacy, reducing false positives in real-world scenarios.

### Proposed Solution
* **Approach:** A **Hybrid Committee of Experts** using **BERT (Transformers)**, compared against a traditional Machine Learning Baseline.
* **Computational Challenges:** Managing the high resource demand of BERT. We transitioned from "brute-force" training to **stratified sampling**, reducing training time from hours to minutes while maintaining statistical integrity.
* **Summary of Results:** The **Hybrid Expert** achieved a **97.1% accuracy**. While BERT shows superior contextual intelligence, the choice of dataset (WELFake) proved vital to maintaining a skeptical edge.

---

## 2. Proposed Method

### Solution Choice
* **Alternatives:** Traditional ML (SVM/Random Forest) was considered as a Baseline, but **BERT (`bert-base-uncased`)** was selected for its **Attention mechanism**, which allows the model to "rectify" a judgment by weighing the body text against the title.
* **Methodology:** Performance was measured using Accuracy and Confidence Scores across three modes: *Title Only*, *Body Only*, and *Full Context*. An **Ablation Study** was conducted to isolate the impact of different training datasets (ISOT vs. WELFake).

---

## 3. Experimental Results

### Demonstration and Technologies
* **Environment:** Google Colab with GPU acceleration (T4/A100).
* **Stack:** Python, PyTorch, Hugging Face Transformers.
* **Reproducibility:** Stratified sampling of 10,000 samples per dataset to ensure balanced and efficient fine-tuning.

### Results
* **Best Configuration:** The **Master Fusion Model** (ISOT + WELFake) provided the most stable results with **97.1% Accuracy**.
* **Ablation Study:** * **ISOT Expert:** Near-perfect on formal news (99.6%) but vulnerable to sophisticated fakes.
    * **WELFake Expert:** More resilient against "high-quality" disinformation.
    * **Full Context vs. Title:** Accuracy consistently increased by ~3-5% when the model had access to the full body text.

---

## 4. Discussion and Conclusions

### Results Discussion
BERT outperformed the Baseline in **contextual rectification**. In cases like the 2026 Fusion Breakthrough, BERT flipped a "FAKE" title label to a "TRUE" verdict (99.9%) by recognizing the institutional tone of the body, whereas the Baseline remained biased by the "future" vocabulary.

### Method Validity
The hybrid approach is highly valid. By combining ISOT and WELFake, we mitigated the "style bias" of individual datasets. However, we discovered that **WELFake is the superior training source** for preventing "blind trust" in well-written but false news.

### Limitations and Maturity
* **Over-Sophistication:** BERT can be "tricked" by high-quality pseudo-science (e.g., the Lunar Snowfall case) because it prioritizes structural perfection over scientific laws.
* **Bias:** The model remains slightly influenced by historical data associations (e.g., "Trump" in titles).
* **Maturity:** The system is a functional prototype (TRL 4) ready for integration into multi-layered verification pipelines.

### Future Works
* **Fact-Checking Integration:** Connecting the model to real-time Knowledge Graphs (API-based fact-checking).
* **Multilingual Expansion:** Testing the Hybrid Expert's resilience in different languages.
* **Adversarial Training:** Specifically training on "High-Level Deepfakes" to close the gap in scientific disinformation detection.
