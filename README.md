# 💼 Financial Sentiment Classification in Text  
*A Comparative Study of ML, DL, and Generative AI Approaches*

## 📘 Overview

This project addresses the challenge of **sentiment classification in financial texts**, a domain known for its specialized terminology, compact expressions, and subtle sentiment cues. We conduct a **comprehensive evaluation** of various modeling strategies, ranging from traditional machine learning to deep learning and generative AI methods.

---

## ⚙️ Methodology

Our approach follows a structured **six-stage machine learning pipeline**:

1. **Exploratory Data Analysis (EDA)**
2. **Multi-variant Preprocessing**
3. **Feature Engineering**
4. **Model Development**
5. **Model Evaluation**
6. **Deployment**

We evaluate **over 45 experimental configurations** to uncover performance trends and identify the most effective strategies for financial sentiment analysis.

---

## 📊 Results Summary

| Model Type               | Best Model                     | Macro F1 | Precision | Notes                                     |
|--------------------------|--------------------------------|----------|-----------|-------------------------------------------|
| Traditional ML           | SVM / Logistic Regression + TF-IDF | 0.73     | –         | Strong baseline but prone to overfitting |
| Deep Learning            | BiLSTM / MLP + GloVe           | ~0.75    | –         | Better generalization, moderate on minority classes |
| Transformer-based / GenAI| GPT-4o Mini                    | **0.81** | **0.83**  | Best overall, high resource consumption   |
| Balanced Trade-off       | SBERT + SVM                    | ~0.78    | ~0.80     | Good compromise between accuracy and efficiency |

---

## 💡 Key Insights

- **Traditional models** can perform competitively with good feature engineering, but tend to overfit.
- **Deep learning** offers better generalization but struggles with minority class performance.
- **Transformer and generative models** (like GPT-4o Mini) yield state-of-the-art results but require significant compute resources.
- **SBERT + SVM** offers a practical balance for real-world deployment when efficiency is a concern.

---

## 🚀 Deployment Considerations

While large pre-trained models are excellent for capturing domain-specific sentiment nuances, real-world applications may favor lighter, faster models when **computational efficiency** is a constraint.

---

## 📁 Project Structure

```bash
📂 src/                 # Source code for preprocessing, training, evaluation
📂 data/                # Raw and processed datasets
📂 models/              # Trained model checkpoints
📊 results/             # Evaluation reports and plots
README.md              # Project overview
requirements.txt       # Python dependencies
