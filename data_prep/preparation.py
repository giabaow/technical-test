import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import unicodedata
from bs4 import BeautifulSoup
import os


raw_path = "data/dataset.csv"
output_dir = os.getenv("PROCESSED_DATA_DIR", "/data/processed")
test_size = float(os.getenv("TEST_SIZE", "0.2"))
random_seed = int(os.getenv("SEED", "42"))


df = pd.read_csv(raw_path)

df = df.drop_duplicates(subset='Text')
df = df.reset_index(drop=True)

def clean_text(text):

    if "<" in text:
        text = BeautifulSoup(text, 'html.parser').get_text()

    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

df["Text"] = df["Text"].apply(clean_text)


def split_and_save(df: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_seed, stratify=df["language"])

    train_path = os.path.join(output_dir, "train.csv")
    test_path  = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)

split_and_save(df, output_dir=output_dir)