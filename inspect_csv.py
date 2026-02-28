import pandas as pd
import os

CSV_PATH = r'C:\Users\spide\OneDrive\Documents\Handwritten-character-recognition-using-Convolutional-Neural-Networks-CNN-\Dataset\written_name_train_v2.csv'

out = open(r'C:\Users\spide\OneDrive\Documents\Handwritten-character-recognition-using-Convolutional-Neural-Networks-CNN-\inspect_out.txt', 'w')

df = pd.read_csv(CSV_PATH)
out.write(f"Total rows: {len(df)}\n")
out.write(f"Columns: {list(df.columns)}\n")
out.write(f"Unique IDENTITY values: {df['IDENTITY'].nunique()}\n")
out.write(f"Top IDENTITY values:\n{df['IDENTITY'].value_counts().head(20)}\n\n")
out.write(f"Sample rows:\n{df.head(10).to_string()}\n\n")

# Check if 'UNREADABLE' or NaN in IDENTITY
out.write(f"NaN in IDENTITY: {df['IDENTITY'].isna().sum()}\n")
if 'UNREADABLE' in df['IDENTITY'].values:
    out.write(f"UNREADABLE count: {(df['IDENTITY']=='UNREADABLE').sum()}\n")

# Check FILENAME prefix
out.write(f"\nFILENAME samples:\n{df['FILENAME'].head(5).to_list()}\n")

out.close()
print("Done. Check inspect_out.txt")
