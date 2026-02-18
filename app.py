# gradio for huggingface spaces
import gradio as gr
import torch
from torch.amp import autocast
from transformers import DistilBertTokenizerFast
from utils.model import DistilBERTClassifier, infer_reviews
import shap

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safe_load_file

REPO_ID = "ayushshah/distilbert-dapt-imdb-sentiment"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = DistilBERTClassifier().to(DEVICE)
tokenizer = DistilBertTokenizerFast.from_pretrained(REPO_ID)

model_path = hf_hub_download(repo_id=REPO_ID, filename="model.safetensors")
model.load_state_dict(safe_load_file(model_path))
model.eval()

def predict_logits(texts):
    if isinstance(texts, str):
        texts = [texts]
    elif not isinstance(texts, list):
        texts = list(texts)

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        with autocast(device_type=DEVICE):
            outputs = model(**inputs)

    return outputs.cpu().numpy()

def predict_sentiment(review):
    _, class_label, confidence = infer_reviews(review, model, tokenizer)
    
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_logits, masker, output_names=['Negative', 'Positive'])
    
    shap_values = explainer([review])
    shap_plot = shap.plots.text(shap_values[0, :, 0], display=False)
    
    return class_label, confidence, shap_plot

# Create Gradio interface
with gr.Blocks(title="DistilBERT Sentiment Analysis") as iface:
    gr.Markdown("# DistilBERT IMDb Sentiment Analysis")
    gr.Markdown("Analyze the sentiment of movie reviews with confidence scores and SHAP explanations. Check out the [model card](https://huggingface.co/ayushshah/distilbert-dapt-imdb-sentiment) for more details.")
    gr.Markdown("""
- Supports only English language.
- Not suitable for tasks other than sentiment analysis on movie reviews.
- It is running on CPU so responses might be slow.
""")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Movie Review",
                placeholder="Enter your movie review here...",
                lines=5
            )
            submit_btn = gr.Button("Analyze Sentiment", variant="primary", interactive=False)

            input_text.change(
                fn=lambda x: gr.update(interactive=False) if x.strip() == "" else gr.update(interactive=True),
                inputs=input_text,
                outputs=submit_btn
            )
        
        with gr.Column():
            label = gr.Label(label="Predicted Sentiment")
            confidence = gr.Number(label="Confidence Score", precision=4)
            
    with gr.Row():
        output_plot = gr.HTML(container=True, value="<div style='padding:10px;'>SHAP Explanation will appear here.</div>")
    
    # Examples
    with gr.Row():
        gr.Examples(
            examples=[
                ["This movie was absolutely amazing! The acting was superb and the plot kept me engaged throughout."],
                ["Terrible film. Waste of time and money. Poor acting and boring storyline."],
                ["It was okay, nothing special but not terrible either."]
            ],
            inputs=input_text
        )
    
    submit_btn.click(fn=predict_sentiment, inputs=input_text, outputs=[label, confidence, output_plot])


iface.launch()