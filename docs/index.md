# Fake News Detection: Hybrid BERT vs. Baseline Analysis

## 1. Introduction

### Problem Description
* **Motivation:** Disinformation has evolved from simple "clickbait" to sophisticated narratives that mimic professional journalism. Detecting these requires more than keyword matching; it requires deep semantic understanding.
* **Target Audience:** Fact-checkers, AI developers, and social media platforms looking for robust automated verification tools.
* **Benefits:** A high-precision model that can differentiate between stylistic "veneer" and factual legitimacy, reducing false positives in real-world scenarios.

### Proposed Solution
* **Approach:** A **Hybrid Committee of Experts** using **BERT (Transformers)**, compared against a traditional Machine Learning Baseline.
* **Computational Challenges:** Transitioned from "brute-force" training to **stratified sampling in Google Colab**, reducing training time from hours to minutes while maintaining statistical integrity.
* **Summary of Results:** The **Hybrid Expert** achieved a **97.1% accuracy**. BERT demonstrated a superior "Temporal Resilience" compared to the Baseline.

---

## 2. Proposed Method

### Solution Choice
* **Alternatives:** Traditional ML (SVM/Random Forest) was considered as a Baseline, but **BERT (`bert-base-uncased`)** was selected for its **Attention mechanism**, which allows the model to "rectify" a judgment by weighing the body text against the title.
* **Methodology:** Performance was measured using Accuracy and Confidence Scores across three modes: *Title Only*, *Body Only*, and *Full Context*. An **Ablation Study** was conducted to isolate the impact of different training datasets (ISOT vs. WELFake).

---

## 3. Experimental Results

### Demonstration and Technologies
* **Environment:** Google Colab with GPU acceleration (T4).
* **Stack:** Python, PyTorch, Hugging Face Transformers.
* **Reproducibility:** Stratified sampling of 10,000 samples per dataset to ensure balanced and efficient fine-tuning.

### Results
* **The "Ablation Jump":** Accuracy consistently increased by ~3-5% when the model had access to the full body text, proving BERT's ability to override headline bias through contextual analysis.
* **Specialist Performance:**
    * **ISOT Expert:** Near-perfect on formal news (99.6%) but vulnerable to sophisticated, well-written fakes.
    * **WELFake Expert:** The most resilient specialist, providing a superior defense against "high-quality" disinformation and satire.

---

## 4. Discussion and Conclusions

### Results Discussion
Our "2026 Stress Test" revealed a critical divide. The **Baseline model suffered a total collapse** when facing the **2026 Nuclear Fusion breakthrough**, labeling it as **FAKE (94% confidence)** due to the "Knowledge Cutoff". However, **BERT successfully identified the news as TRUE (99.9%)**. This proves that semantic depth allows BERT to recognize professional journalistic standards even when the topic is outside its original training window.

### Method Validity
The hybrid approach is highly valid. By combining ISOT and WELFake, we mitigated the "style bias" of individual datasets. **WELFake** emerged as the superior training source for developing a "healthy skepticism" against authoritative-sounding disinformation.

### Limitations and Maturity
* **Institutional Mimicry:** While resilient to temporal drift, BERT can still be "tricked" by high-quality satire (e.g., the **Lunar Snowfall** case) if the prose perfectly mimics scientific jargon.
* **Maturity:** The system is a functional prototype (**TRL 4**) ready for integration into real-world verification pipelines.

### Future Works
* **RAG Integration:** Connecting the model to real-time fact-checking (Knowledge Graphs) to verify physical facts alongside stylistic analysis.
* **User Interface:** Deploying the model into a practical application accessible to any citizen.

