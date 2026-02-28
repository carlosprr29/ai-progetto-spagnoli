# 📰 Fake News Detector Pro: BERT-Based Ablation Study
Project Status: In Development 

Authors: Carlos Padilla Ruano and Antonio José Osuna Montes


# 📝 Project Description
This project develops a fake news classification system based on the BERT (Bidirectional Encoder Representations from Transformers) language model. The primary objective is to conduct an ablation study to determine how training on different data sources influences performance and how accuracy varies when analyzing the headline only versus the full text.

  # Training Architecture:
  We have trained and compared three main systems:
  
    1. WELFake Model: Trained on a diverse web-based dataset (social media news, clickbait).
    2. ISOT Model: Trained on official agency news (Reuters) and political disinformation sites.
    3. Fusion (Master) Model: An integrated model that combines both datasets to improve generalization.

    
# 🚀 Applied Methodology
To mitigate common biases found in public datasets, we implemented:
  - Bias Cleaning: 
  Removal of agency headers (e.g., "Reuters", "Washington") and city names that provided artificial "clues" to the model.
  - Data Augmentation:
  Random sentence shuffling within the training set to ensure the AI learns contextual patterns rather than memorizing specific paragraphs.
  - Thresholding System:
  An interactive control that manages uncertainty by flagging news as "Doubtful/Inconclusive" if they do not meet a minimum percentage of confidence.
  
  
# ⚠️ Identified Issues & Critical Points
Despite achieving high precision in validation metrics (Accuracy > 90%), real-world tests with news from February 2026 have revealed significant challenges:
  1. The "Satire Wall": The ISOT model incorrectly classifies satirical news (e.g., The Onion) as "Real" because they perfectly mimic the formal style of a news agency.
  2. Lack of World Knowledge: The system detects "how it is written" (form) but not "what is being said" (substance). It fails to identify logical absurdities if they are well-redacted.
  3. Stylistic Bias: Real news using highly emotional language are sometimes flagged as Fake, while fake news written with a sober, professional tone successfully deceive the system.


# 🛠️ How to Run this Notebook
1. Connect to Google Drive.
2. Load the models from the paths specified in /Project_IA/.
3. Use the Control Panel to test individual news items.
4. Execute the Test Battery to view the triple comparative results.
