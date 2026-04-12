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
To ensure the model prioritizes **semantics** over **structural shortcuts**, we implemented:

* **Source De-biasing:** Systematic removal of agency headers ("Reuters", "Washington") and metadata tags that act as artificial proxies for truth.
* **Linguistic Data Augmentation:** Strategies to force the AI to learn deep context rather than memorizing fixed paragraph distributions.
* **Ablation Framework:** A dual-input analysis comparing **Title-Only vs. Full-Text** to identify the "Style Trap" in modern reporting.
* **Stratified Efficiency:** Implementation of stratified sampling to reduce training latency from hours to minutes without sacrificing statistical significance.

---

## 🧪 Key Findings: The 2026 Frontier
Testing our models against real-world and synthetic news from **February 2026** revealed four critical frontiers in NLP:

### 1. The "Ablation Jump" (Contextual Rectification)
BERT demonstrates a superior ability to override headline bias. In cases like the **2026 Fusion Breakthrough**, while the *Title-Only* mode flagged the news as **FAKE (99%)**, the **Full-Context** analysis reached a **99.9% TRUE** verdict, successfully identifying the professional coherence of the body text.

### 2. The Fact-Checking Paradox
In complex "debunking" reports (e.g., Pope Francis tests), BERT correctly identified the text as **TRUE (99%)** by recognizing its formal journalistic structure and citations. Traditional Baselines failed here, flagging the content as FAKE by only detecting "negative" keywords without understanding the intent.

### 3. The "Satire Wall" & Linguistic Mimicry
Our most significant discovery: **High-quality satire can "hack" BERT.** If a lie is written with perfect institutional jargon (e.g., NASA-style prose), BERT prioritizes **Authoritative Rhetoric (Form)** over **Physical Possibility (Substance)**, maintaining a dangerously high confidence in false reports.

### 4. The Fusion Paradox (Domain Interference)
We discovered that the **WELFake Specialist** often provides more resilient defenses than the Fusion Model. Combining datasets with conflicting styles (social media vs. formal agencies) can lead to **"Domain Interference"**, suggesting that specialization is often more effective than raw data volume.

---

## ⚖️ Final Verdict: Efficiency vs. Depth

| Feature | Baseline (Logistic Regression) | BERT (Committee of Experts) |
| :--- | :--- | :--- |
| **Computational Cost** | Ultra-Low (Efficient Sentry) | High (Requires GPU Acceleration) |
| **Detection Logic** | Keyword & Pattern Recognition | Semantic & Structural Analysis |
| **Best Performance** | Catching obvious sensationalism | Validating complex, professional-grade news |
| **Weakness** | High False Positive rate in 2026 news | Vulnerable to "Institutional Mimicry" |

---

## 📂 Project Structure

```plaintext
├── docs/                # Project documentation
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
│   ├── main.py          # Execution entry point
│   └── modelling.py     # Committee of Experts & BERT architecture
├── .gitignore           # Files excluded from version control
├── LICENSE              # Project licensing information
├── README.md            # Main project documentation
└── requirements.txt     # Python dependencies (Transformers, Torch, etc.)
