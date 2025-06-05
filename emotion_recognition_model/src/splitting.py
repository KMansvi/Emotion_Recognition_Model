import pandas as pd
from sklearn.model_selection import train_test_split

# Load your original dataset
df = pd.read_csv('data/goemotions_augmented.csv')

# First split into train_val (80%) and test (20%)
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Then split train_val into train (75% of train_val) and val (25% of train_val)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)

# Save to csv files
train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

print("Data split done:")
print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")
print(f"Test size: {len(test_df)}")
