import os
import torch
from torch.utils.data import random_split
from utils.dataset import IMDbReviewsDataLoader, IMDbReviews
from utils.model import DistilBERTClassifier
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import json
from utils.utils import setup, plot_curves, calculate_and_save_metrics, test_model


def main():
    setup(23)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 64

    trainloader = IMDbReviewsDataLoader(
        split='train',
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    test_dataset = IMDbReviews(split='test')
    val_size = int(0.2 * len(test_dataset))
    test_size = len(test_dataset) - val_size
    generator = torch.Generator().manual_seed(23)
    val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size], generator=generator)

    valloader = IMDbReviewsDataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    testloader = IMDbReviewsDataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    
    EPOCHS = 3
    ROOT = 'assets'
    os.makedirs(ROOT, exist_ok=True)

    num_train_steps = len(trainloader) * EPOCHS
    warmup_steps = int(0.1 * num_train_steps)

    model = DistilBERTClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW([
        {'params': model.distilbert.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 5e-5},
    ])

    # Linear decay with warmup
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(1e-8, float(num_train_steps - current_step) / float(max(1, num_train_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = torch.nn.CrossEntropyLoss()

    scaler = GradScaler()
    best_val_acc = 0.0

    # Loss tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(EPOCHS):
        # Training
        model.train()
        epoch_loss = 0.0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False, unit="batch")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=DEVICE):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_size = labels.size(0)
            epoch_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()
            train_correct += correct
            train_total += batch_size

            history['train_loss'].append(epoch_loss / train_total)
            history['train_acc'].append(train_correct / train_total)

            progress_bar.set_postfix(loss=loss.item(), acc=correct / batch_size)

        avg_train_loss = history['train_loss'][-1]
        train_acc = history['train_acc'][-1]

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(valloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False, unit="batch")
        with torch.no_grad():
            for batch in loop:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                with autocast(device_type=DEVICE):
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)

                batch_size = labels.size(0)
                val_loss += loss.item() * batch_size
                preds = torch.argmax(logits, dim=1)
                c = (preds == labels).sum().item()
                correct += c
                total += batch_size

                loop.set_postfix(loss=loss.item(), acc=c / batch_size)

                history['val_loss'].append(val_loss / total)
                history['val_acc'].append(correct/total)

        avg_val_loss = history['val_loss'][-1]
        val_acc = history['val_acc'][-1]

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'{ROOT}/best_model.pt')

    # Save loss history to JSON
    with open(f'{ROOT}/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    plot_curves(history, ROOT)

    # Final evaluation on test set
    all_preds, all_labels, all_probs = test_model(model, f'{ROOT}/best_model.pt', testloader, DEVICE)
    calculate_and_save_metrics(all_preds, all_labels, all_probs, ROOT)


if __name__ == "__main__":
    main()