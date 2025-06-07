# 🍔 McDonald's Healthier Alternative Recommendation System

**An ML + NLP + DL-powered approach to suggesting better food choices**

---

## 🧠 Overview

**McDonalds-Healthier-Alternative-Recommendation** is a hybrid machine learning and deep learning project designed to suggest **healthier alternatives** to popular McDonald’s menu items. It uses a combination of:

* **K-Nearest Neighbors (KNN)** for nutritional similarity,
* **TF-IDF + Cosine Similarity** for name-based similarity,
* **BERT** for healthiness classification from food descriptions.

By combining these AI techniques, this system provides smarter food alternatives—balancing taste familiarity with nutritional improvement.

---

## 🔍 Problem Statement

Fast food is convenient, but not always healthy. The goal of this project is to:

1. Recommend similar food items based on **nutrition** (e.g., protein and fat).
2. Recommend similar items based on **name/description** (textual similarity).
3. **Classify** whether a food item is *healthy* or *unhealthy* based on its description.

---

## 🛠️ Techniques Used

### 1️⃣ **Nutritional Similarity with KNN (ML)**

* **Algorithm**: K-Nearest Neighbors (KNN)
* **Features Used**: `protein`, `fat` (normalized and scaled)
* **Goal**: Find top-k items from a healthy food database that are closest in nutritional profile to a McDonald’s item.
* **Result**: Suggests 3 nearest matches, e.g., for *Spicy Chicken Wrap*, alternatives might be: `Nuts`, `Milkshake`, and `Coffee`.

---

### 2️⃣ **Textual Similarity with TF-IDF + Cosine Similarity (NLP)**

* **Technique**:

  * TF-IDF Vectorizer to encode food item names
  * Cosine Similarity to compare text embeddings
* **Goal**: Suggest food items with **similar names** or descriptions (e.g., user searches for “Chicken Wrap”, system suggests “Banana Wrap” or “Spiced Oats”).
* **Limitation**: Does *not* consider nutrition, only text similarity.
* **Result**: For *Spicy Chicken Wrap*, results might be: `Tomato`, `Banana`, and `Oats`.

---

### 3️⃣ **Healthiness Classification with BERT (DL)**

* **Model**: Pre-trained BERT, fine-tuned on a binary classification task (`Healthy` vs `Unhealthy`)
* **Input**: Food item description (text)
* **Output**: Prediction of healthiness
* **Example**:

  * Input: "Spicy Chicken Wrap"
  * Output: `Unhealthy`

---

## 🔄 Combined Flow

```
             +---------------------+
             | McDonald's Food Item|
             +---------------------+
                       |
       +-----------------------------+
       | Nutritional Vector (Protein/Fat)
       +-----------------------------+
                       |
          [KNN] --> Nearest Healthy Matches
                       |
               ----------------------------------
               | TF-IDF Vector (Name/Description)
               ----------------------------------
                               |
       [Cosine Similarity] --> Textual Alternatives
                               |
               ----------------------------------
               | Food Description Input (Text)
               ----------------------------------
                               |
            [BERT Classifier] --> Health Prediction
```

---

## 💻 Project Structure

```bash
McDonalds-Healthier-Alternative-Recommendation/
│
├── data/                   # Food datasets (McDonald's, healthy foods)
├── models/                 # Saved BERT models / TF-IDF pickles
├── knn_recommender.py      # KNN logic for nutritional similarity
├── nlp_similarity.py       # TF-IDF + Cosine similarity script
├── bert_classifier.py      # BERT fine-tuning and inference
├── utils.py                # Data preprocessing, scaling, etc.
├── app.py                  # Optional streamlit or flask frontend
├── requirements.txt        # Dependencies
└── README.md               # Project info
```

---

## 📊 Example Output

**Input**: `"Spicy Chicken Wrap"`

| Technique       | Output                        |
| --------------- | ----------------------------- |
| KNN (Nutrition) | `Nuts`, `Milkshake`, `Coffee` |
| NLP (Text)      | `Tomato`, `Banana`, `Oats`    |
| BERT (Health)   | `Prediction: Unhealthy`       |

---

## ⚙️ How to Run

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/McDonalds-Healthier-Alternative-Recommendation.git
cd McDonalds-Healthier-Alternative-Recommendation
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Individual Scripts

**KNN Recommender**

```bash
python knn_recommender.py --item "Spicy Chicken Wrap"
```

**NLP-Based Similarity**

```bash
python nlp_similarity.py --item "Spicy Chicken Wrap"
```

**BERT Classifier**

```bash
python bert_classifier.py --description "Spicy Chicken Wrap with crispy chicken and mayo"
```

---

# License

This repository is proprietary and all rights are reserved. No usage, modification, or distribution is allowed without permission.
