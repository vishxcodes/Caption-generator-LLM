import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import pandas as pd


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_NAME = "Salesforce/blip-image-captioning-base"


TRAIN_DATASET_PATH = "dataset/train.csv"
IMAGE_FOLDER = "dataset/images"


OUTPUT_DIR = "finetuned_model"


BATCH_SIZE = 4
EPOCHS = 3
LR = 5e-5
MAX_LEN = 40



class CaptionDataset(Dataset):

    def __init__(self, dataframe, processor, image_dir):
        self.df = dataframe
        self.processor = processor
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = os.path.join(self.image_dir, row["image"])
        caption = row["caption"]

        image = Image.open(image_path).convert("RGB")
        # TOKENIZE
        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        encoding = {k: v.squeeze() for k, v in encoding.items()}

        encoding["labels"] = encoding["input_ids"]

        return encoding



def load_dataset():

    df = pd.read_csv(TRAIN_DATASET_PATH)

    print("Dataset size:", len(df))

    return df


# Batches
def build_dataloaders(processor):

    df = load_dataset()

    dataset = CaptionDataset(df, processor, IMAGE_FOLDER)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    return dataloader



def train():

    processor = BlipProcessor.from_pretrained(MODEL_NAME)

    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)

    model.to(DEVICE)


    train_loader = build_dataloaders(processor)


    optimizer = AdamW(model.parameters(), lr=LR)


    total_steps = len(train_loader) * EPOCHS


    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )


    model.train()


    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        total_loss = 0

        progress = tqdm(train_loader)

        for batch in progress:

            pixel_values = batch["pixel_values"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)


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


        avg_loss = total_loss / len(train_loader)

        print("Average Loss:", avg_loss)


    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print("Model saved to:", OUTPUT_DIR)



def generate_example():

    processor = BlipProcessor.from_pretrained(OUTPUT_DIR)
    model = BlipForConditionalGeneration.from_pretrained(OUTPUT_DIR)

    model.to(DEVICE)

    model.eval()

    image = Image.open("sample.jpg").convert("RGB")

    inputs = processor(image, return_tensors="pt").to(DEVICE)

    out = model.generate(**inputs, max_length=30)

    caption = processor.decode(out[0], skip_special_tokens=True)

    print("Generated Caption:", caption)



if __name__ == "__main__":

    print("Starting BLIP fine-tuning pipeline...")

    train()

    print("Training complete")

    generate_example()