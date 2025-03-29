# -McDonalds-Healthier-Alternative-Recommendation

1. Healthier Alternatives: (Using ML - KNN)
Method: K-Nearest Neighbors (KNN) is used here.

Explanation: KNN is a classical machine learning algorithm used for finding items that are "nearest" to the selected food item based on certain features—in this case, nutritional information (protein and fat).

How it works:

You select a McDonald's food item (e.g., Spicy Chicken Wrap), and its nutritional information (protein and fat content) is scaled and transformed.

The KNN algorithm finds the top 3 food items from the Daily Food & Nutrition dataset that are most similar in terms of protein and fat content.

The system suggests food items that are healthier alternatives based on the nutritional similarity (although these may not always be the healthiest options, they are nutritionally similar).

Result: In your case, it suggests Nuts, Milkshake, and Coffee as healthier alternatives, based on similar protein and fat content.

2. NLP-Based Similar Food Items: (Using NLP - Cosine Similarity with TF-IDF)
Method: TF-IDF (Term Frequency-Inverse Document Frequency) for text-based similarity and Cosine Similarity for comparing the similarity of food item names.

Explanation: Here, Natural Language Processing (NLP) techniques are used to find food items that have similar names or descriptions.

TF-IDF vectorizes the food item names from both McDonald's and the Daily Food datasets, which helps to transform the textual data (food item names) into numerical form.

Cosine Similarity measures how similar two text vectors (food item names) are, based on their "distance" in a high-dimensional space. The cosine similarity score is between 0 and 1 (where 1 means identical).

This technique doesn’t consider the nutritional content but looks for items with similar names or keywords in the dataset.

Result: For your selected item "Spicy Chicken Wrap", the similar items found by the model are Tomato, Banana, and Oats based on the text-based similarity of their names.

3. Predicted Healthiness: (Using DL - BERT for Sequence Classification)
Method: BERT (Bidirectional Encoder Representations from Transformers) for sequence classification (using a pre-trained model for text classification).

Explanation: BERT, a state-of-the-art deep learning model for Natural Language Processing (NLP), is used here to classify whether a given food item is healthy or unhealthy. This model is trained on textual data and can perform tasks like sentiment analysis, classification, and more.

In your case, BERT classifies the Spicy Chicken Wrap as unhealthy or healthy based on its description.

The BERT model is fine-tuned on labeled data (e.g., healthy or unhealthy), which means it has been trained to predict these labels based on the food item descriptions.

Result: For Spicy Chicken Wrap, the model predicts Unhealthy.

Breakdown of Each Part in the Flow:
KNN (ML): Finds healthier alternatives based on nutritional similarities (protein and fat content).

TF-IDF + Cosine Similarity (NLP): Finds text-based similar food items by comparing food item names.

BERT (DL): Predicts the healthiness of the selected food item using deep learning-based text classification.

To summarize:
KNN (ML) predicts nutritionally similar food items.

TF-IDF and Cosine Similarity (NLP) predicts text-based similarity between food items' names.

BERT (DL) predicts whether the selected food item is healthy or unhealthy based on its description
