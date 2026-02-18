import torch
from utils.model import DistilBERTClassifier, infer_reviews
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safe_load_file

REPO_ID = "ayushshah/distilbert-dapt-imdb-sentiment"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = DistilBERTClassifier().to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(REPO_ID)

model_path = hf_hub_download(repo_id=REPO_ID, filename="model.safetensors", local_dir="./assets")
model.load_state_dict(safe_load_file(model_path))
model.eval()

# For single review inference
sample_review = "Great movie"
_, label, conf = infer_reviews(sample_review, model, tokenizer)
print(f"Review: {sample_review}\nPredicted Sentiment: {label} (Confidence: {conf:.4f})\n")

# For batch inference
reviews = ["I loved this film!", "This was a terrible movie."]
results = infer_reviews(reviews, model, tokenizer)
for review, (_, label, conf) in zip(reviews, results):
    print(f"Review: {review}\nPredicted Sentiment: {label} (Confidence: {conf:.4f})\n")