# DistilBERT-IMDb
Sentiment analysis on IMDb reviews.

This repository contains the code to:
- Perform domain adaptation of DistilBERT on IMDb movie reviews using DAPT.
- Train a linear classifier head on top of the above model with fine-tuning for sentiment analysis.
- Create a Gradio web application for interactive sentiment analysis with SHAP explanations.

Performing Domain-Adaptive Pretraining (DAPT) results in better representations for the target domain (movie reviews) compared to using the original DistilBERT, leading to improved sentiment classification performance.

Using only a linear classifier, opposed to using it with a pre classifier as in the transformers library, provides similar performance with slight gains (~0.5%) with less parameters.


## Structure
- `dapt.py`: Code for domain-adaptive pretraining of DistilBERT on IMDb reviews.
- `train.py`: Code for training the linear classifier head on top of the DAPT model.
- `inference.py`: Script for performing inference on new reviews using the trained model.
- `app.py`: Gradio web application for interactive sentiment analysis with SHAP explanations.
- `utils/model.py`: Contains the DistilBERTClassifier class and inference function.
- `utils/mlm.py`: Class definition for masked language modeling.
- `utils/dataset.py`: Dataset loading and preprocessing utilities.
- `utils/utils.py`: Miscellaneous utility functions.
