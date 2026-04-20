import torch
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu


def train_one_epoch(model, dataloader, optimizer, scheduler, device):

    model.train()

    total_loss = 0

    progress = tqdm(dataloader)

    for batch in progress:

        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        scheduler.step()

        total_loss += loss.item()

        progress.set_description(f"loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)

    return avg_loss



def evaluate(model, dataloader, processor, device):

    model.eval()

    bleu_scores = []

    with torch.no_grad():

        for batch in dataloader:

            pixel_values = batch["pixel_values"].to(device)

            outputs = model.generate(pixel_values=pixel_values)

            preds = processor.batch_decode(outputs, skip_special_tokens=True)

            labels = processor.batch_decode(
                batch["labels"], skip_special_tokens=True
            )

            for pred, label in zip(preds, labels):

                score = sentence_bleu([label.split()], pred.split())

                bleu_scores.append(score)

    return sum(bleu_scores) / len(bleu_scores)



def save_checkpoint(model, optimizer, epoch, path):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }

    torch.save(checkpoint, path)



def load_checkpoint(model, optimizer, path, device):

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]

    return epoch