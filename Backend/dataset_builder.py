import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


RAW_IMAGE_DIR = "raw_images"
CAPTION_FILE = "captions.csv"

OUTPUT_DIR = "dataset"

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1


def clean_caption(text: str) -> str:
    text = text.lower()
    text = text.strip()
    text = text.replace("\n", " ")
    return text


def validate_image(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except Exception:
        return False


def load_raw_dataset():

    df = pd.read_csv(CAPTION_FILE)

    rows = []

    for _, row in df.iterrows():

        image_path = os.path.join(RAW_IMAGE_DIR, row["image"])

        if not os.path.exists(image_path):
            continue

        if not validate_image(image_path):
            continue

        caption = clean_caption(row["caption"])

        rows.append({
            "image": row["image"],
            "caption": caption
        })

    return pd.DataFrame(rows)


def split_dataset(df):

    train_df, temp_df = train_test_split(df, test_size=(1 - TRAIN_SPLIT))

    val_size = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)

    val_df, test_df = train_test_split(temp_df, test_size=(1 - val_size))

    return train_df, val_df, test_df


def save_dataset(train_df, val_df, test_df):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)


def print_stats(train_df, val_df, test_df):

    print("\nDataset Statistics")
    print("------------------")
    print("Train samples:", len(train_df))
    print("Validation samples:", len(val_df))
    print("Test samples:", len(test_df))


def main():

    print("Loading raw dataset...")

    df = load_raw_dataset()

    print("Total valid samples:", len(df))

    train_df, val_df, test_df = split_dataset(df)

    save_dataset(train_df, val_df, test_df)

    print_stats(train_df, val_df, test_df)

    print("Dataset successfully prepared.")


if __name__ == "__main__":
    main()