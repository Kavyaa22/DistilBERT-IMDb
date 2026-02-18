import torch
from transformers import DistilBertForMaskedLM
from utils.mlm import MaskedLanguageModeling
from utils.dataset import IMDbReviewsDataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from utils.utils import setup


def main():
    setup(23)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataloader = IMDbReviewsDataLoader(
        split='unsupervised',
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    EPOCHS = 3
    grad_accum_steps = 4

    num_update_steps_per_epoch = len(dataloader) // grad_accum_steps
    num_train_steps = num_update_steps_per_epoch * EPOCHS
    warmup_steps = int(0.05 * num_train_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased').to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    mlm = MaskedLanguageModeling(tokenizer=dataloader.dataset.tokenizer)

    scaler = GradScaler()
    optimizer.zero_grad(set_to_none=True)

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        accum_steps = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False, unit="batch")
        for batch in progress_bar:
            inputs = {key: val.to(DEVICE) for key, val in batch.items() if key != 'labels'}
            mlm_inputs = mlm(inputs)
            labels = mlm_inputs.pop('labels').to(DEVICE)

            with autocast(device_type=DEVICE):
                outputs = model(**mlm_inputs, labels=labels)
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()
            accum_steps += 1

            if accum_steps == grad_accum_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                scheduler.step()
                accum_steps = 0

            epoch_loss += loss.item() * grad_accum_steps
            progress_bar.set_postfix(loss=loss.item() * grad_accum_steps, lr=scheduler.get_last_lr()[0])

        if accum_steps > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

    # Save the domain-adapted model
    model.save_pretrained('distilbert-base-uncased-imdb-dapt')

if __name__ == "__main__":
    main()