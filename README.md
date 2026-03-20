# Fake News Detector Pro: BERT-Based Committee of Experts
An Ablation Study on Temporal Decay and Semantic Depth (2016-2026)
Authors: Carlos Padilla Ruano and Antonio José Osuna Montes

# 📌 Project Description
This project develops an advanced fake news classification system using BERT (Bidirectional Encoder Representations from Transformers). The core objective is to move beyond simple word-counting to understand contextual intent, comparing it against traditional Machine Learning baselines in the face of evolving misinformation.

  The Committee of Experts Architecture
  We implemented a triple-specialist system to analyze news from different angles:

  WELFake Specialist: Trained on diverse web-based data (social media, clickbait).

  ISOT Specialist: Trained on formal agency news (Reuters) vs. political disinformation.

  Fusion (Master) Model: An integrated ensemble that combines both domains to maximize generalization.

# 🛠️ Applied Methodology (Technical Rigor)
To ensure the model learns semantics rather than shortcuts, we implemented:

  Bias Cleaning: Systematic removal of agency headers ("Reuters", "Washington") and location tags that provide artificial "clues".

  Data Augmentation: Random sentence shuffling to force the AI to learn context instead of memorizing fixed paragraph structures.

  Ablation Testing: A dual-mode analysis comparing Title-Only vs. Full-Text to identify the "Style Trap" in news reporting.

# 🧪 Key Findings: The 2026 Battle
Testing our models against real-world and synthetic news from February 2026 revealed three critical frontiers:

  1. The "Ablation Jump" (Self-Correction)
BERT successfully uses body text to override sensationalist headline bias. In cases like 2026 Fusion Breakthroughs, while the Title-Only expert flagged it as FAKE (99%), the Full-Text analysis reached a near-perfect 99.9% REAL verdict, correcting the Baseline's failure.

  2. The Fact-Checking Paradox
In the Pope Francis test, BERT correctly identified a debunking report as REAL (99%) by recognizing its formal journalistic structure and citations. The Baseline failed here, incorrectly flagging it as FAKE by only detecting "negative" keywords without understanding the intent.

  3. The "Satire Wall" & Linguistic Mimicry
Our most significant discovery: high-quality satire (e.g., Lunar Snowfall) can deceive BERT because the model prioritizes authoritative rhetoric (form) over physical possibility (substance). If a lie is written with perfect NASA-style jargon, BERT's confidence in "REAL" remains dangerously high.

  4. The Fusion Paradox (Data Dilution)
We discovered that the Fusion Model does not significantly outperform individual specialists. Combining datasets with conflicting styles (social media vs. formal agencies) can lead to "Domain Interference," proving that specialization is more effective than raw data volume.

# ⚖️ Final Verdict: Efficiency vs. Depth
The Baseline (Logistic Regression): An "Efficient Sentry." Excellent at catching obvious sensationalism at near-zero computational cost.

BERT: The "Semantic Expert." Essential for validating complex truths and modern geopolitical events, though vulnerable to professional-grade mimicry.

# 📂 Project Structure
Plaintext
├── notebooks/          # Google Colab experiments & scrapers
├── src/                # Modular Source Code
│   ├── data_loader.py  # Dataset ingestion (WELFake/ISOT)
│   ├── preprocessing.py# Bias cleaning & tokenization
│   ├── models.py       # Committee of Experts logic
│   └── testing.py      # Ablation & 2026 test battery
├── requirements.txt    # Dependencies (transformers, torch, sklearn)
└── README.md           # Documentation

# 🚀 How to Run
Environment: Connect to Google Colab and mount your Drive.

Dependencies: Install via pip install -r requirements.txt.

Execution: Use the Control Panel in the main notebook to test individual news items or run the full Ablation Test Battery.
