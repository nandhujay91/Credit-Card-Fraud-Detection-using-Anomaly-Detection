import pandas as pd

from src.preprocessing.duplicate_handler import DuplicateHandler

df = pd.read_csv("data/interim/validated_creditcard.csv")

print("Original Shape :", df.shape)

print("Duplicate Rows Before :", df.duplicated().sum())

clean_df = DuplicateHandler.remove_duplicates(df)

print("Clean Shape :", clean_df.shape)

print("Duplicate Rows After :", clean_df.duplicated().sum())