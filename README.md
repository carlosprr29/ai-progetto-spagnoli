# Fake News Detector Pro: BERT-Based Committee of Experts
> **An Ablation Study on Temporal Decay and Semantic Depth** > **Authors:** Carlos Padilla Ruano & Antonio José Osuna Montes

---

## 📌 Project Overview
This project engineers an advanced disinformation classification framework using **BERT (Bidirectional Encoder Representations from Transformers)**. Moving beyond traditional frequency-based Machine Learning (BoW/TF-IDF), this system focuses on **contextual intent** and **syntactic legitimacy** to detect evolving fake news patterns in a decadal window (2016–2026).

### **The Committee of Experts Architecture**
We implemented a **Triple-Specialist Ensemble** to analyze news through different stylistic lenses:
* **WELFake Specialist:** Optimized for high-variance digital data (social media, clickbait, and diverse web sources).
* **ISOT Specialist:** Specialized in formal agency reporting (e.g., Reuters) vs. structured political propaganda.
* **Hybrid Fusion (Master) Model:** A cross-domain ensemble designed to reconcile conflicting writing styles and maximize generalization.

---

## 🛠️ Applied Methodology (Technical Rigor)

To ensure the system moves beyond **"counting words"** and focuses on **"understanding stories,"** our methodology follows three strategic pillars designed to maximize semantic depth:

* **Semantic De-biasing:** We implemented a systematic removal of agency headers (e.g., "Reuters", "Washington") and metadata tags. This forces BERT to ignore "structural shortcuts" and focus purely on the **linguistic intent** and **narrative style** of the news, preventing the model from simply memorizing source names.
* **Ablation & Contextual Framework:** We developed a dual-input analysis comparing **Title-Only vs. Full-Text**. This allows the model to avoid the **"Style Trap"**—where a sensationalist headline might trigger a false positive—by allowing BERT to "rectify" its judgment through the analysis of the body text.
* **Execution & Scalability (Google Colab):** All experiments were conducted within a **Google Colab environment** using **Stratified Sampling**. This allowed us to reduce training latency from hours to minutes without losing statistical significance, enabling rapid stress-testing across both **2016 and 2026 news cycles**.

---

## 🧪 Key Findings: The 2026 Frontier

Testing our models against real-world news from **2016 and February 2026** within our **Google Colab environment** revealed four critical insights into modern NLP:

### 1. The "Ablation Jump" (Contextual Rectification)
BERT demonstrates a superior ability to override headline bias. In various test cases, while the **Title-Only** mode flagged news as **FAKE** due to sensationalist headers, the **Full-Context** analysis successfully "rectified" the verdict to **TRUE**, proving that the model uses the body text to validate or debunk the headline.

### 2. The Temporal Drift (Baseline vs. BERT)
Our stress tests revealed a massive gap in temporal resilience. The **Baseline model** suffered a total collapse when facing the **2026 Nuclear Fusion breakthrough**, labeling this true story as **FAKE with 94% confidence**. This proves that frequency-based models (BoW/TF-IDF) are "trapped" in their training year. In contrast, **BERT-based models** successfully identified the news as **TRUE**, proving that semantic understanding is far more effective than simple keyword matching when facing future events.

### 3. The "Satire Wall" & Linguistic Mimicry
Even the most advanced models have limits. We discovered that high-quality satire can "hack" BERT's logic. When a false story is written with perfect institutional jargon (e.g., the **"Lunar Snowfall"** case), the **Hybrid Fusion** model prioritized **Authoritative Rhetoric (Form)** over **Physical Possibility (Substance)** and failed. 

### 4. The WELFake Resilience (The Ultimate Expert)
The most significant finding of this project was the absolute robustness of the **WELFake Specialist**. While the Baseline failed the 2026 test and the Fusion model failed the Satire test, **WELFake correctly identified 100% of the cases** in our Google Colab environment. This suggests that training on diverse, "noisy" digital data creates a superior form of skepticism, making it our most reliable defense against professionalized disinformation.

---

## ⚖️ Final Verdict: Efficiency vs. Depth

| Feature | Baseline (Logistic Regression) | BERT (Committee of Experts) |
| :--- | :--- | :--- |
| **Computational Cost** | Ultra-Low (Efficient Sentry) | High (Requires GPU Acceleration) |
| **Detection Logic** | Keyword & Pattern Recognition | Semantic & Structural Analysis |
| **Best Performance** | Catching obvious, low-quality fake news | Validating complex, professional-grade news |
| **Temporal Resilience** | **Fails in 2026 tests** (Knowledge Cutoff) | **Success in 2026 tests** (Semantic Depth) |
| **Core Weakness** | High False Positive rate in modern news | Vulnerable to "Institutional Mimicry" (Satire) |

---

## 📂 Project Structure
> ⚠️ **Note on Model Weights**
>
> Due to GitHub's file size limitations, all trained model weights are hosted on Google Drive and linked within the `models/` directory for direct use.
>
> 🔗 [Access Trained Models Here](https://drive.google.com/drive/folders/17rkrKPKLdCeyllKBnYh1QMlE8Q3CcFMQ?usp=sharing)

```plaintext
├── docs/                # Project documentation
│   └── MEMORIA.pdf      # Documentation of the project
│   └── index.md         # Documentation entry point
├── models/              # Saved model weights and checkpoints
├── notebooks/           # Research and development workflow
│   ├── 01_1_Exploratory_Data_Analysis...
│   ├── 01_2_Exploratory_Data_Analysis...
│   ├── 01_3_Exploratory_Data_Analysis...
│   ├── 02_Baseline_Model.ipynb
│   ├── 03_1_BERT_Training_WELFake.ipynb
│   ├── 03_2_BERT_Training_ISOT.ipynb
│   ├── 03_3_BERT_Training_Maestro.ipynb
│   └── 04_Testing_and_Evaluation.ipynb
├── src/                 # Modular Source Code
│   ├── data_loader.py   # Stratified ingestion (WELFake/ISOT)
│   ├── eda_utils.py     # Helper functions for data visualization
│   ├── evaluation.py    # Performance metrics and Ablation logic
│   └── modelling.py     # Committee of Experts & BERT architecture
├── .gitignore           # Files excluded from version control
├── LICENSE              # Project licensing information
├── README.md            # Main project documentation
└── requirements.txt     # Python dependencies


> ⚠️ **Note on Model Weights:** Due to GitHub's file size limitations, the trained weights for the **WELFake**, **ISOT**, and **Hybrid Fusion** models are hosted on Google Drive. 
 
[Access Trained Models Here](https://drive.google.com/drive/folders/17rkrKPKLdCeyllKBnYh1QMlE8Q3CcFMQ?usp=sharing)
 
