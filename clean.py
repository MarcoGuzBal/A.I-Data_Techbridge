import pandas as pd
import numpy as np
import argparse

def load_data(file_path, sheet_lines, sheet_header):
    """Load data from an Excel file."""
    df = pd.read_excel(file_path, sheet_name=sheet_lines)
    df2 = pd.read_excel(file_path, sheet_name=sheet_header)
    return df, df2

def clean_and_merge_data(df, df2):
    """Clean and merge the dataset by removing unnecessary columns and handling missing values."""
    df.columns = df.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    
    # Convert date columns to datetime and drop NaN values
    if 'DONATIONDATE' in df2.columns:
        df2['DONATIONDATE'] = pd.to_datetime(df2['DONATIONDATE'], errors='coerce')
        df2 = df2.dropna(subset=['DONATIONDATE'])
    if 'EXPIRATIONDATE' in df.columns:
        df['EXPIRATIONDATE'] = pd.to_datetime(df['EXPIRATIONDATE'], errors='coerce')
    
    # Drop columns with more than 70% missing values
    null_percentage = df.isnull().mean() * 100
    columns_to_drop = null_percentage[null_percentage > 70].index
    df = df.drop(columns=columns_to_drop)
    
    # Drop specific columns
    if 'BILLOFLADINGNUMBER' in df.columns:
        df = df.drop(columns='BILLOFLADINGNUMBER')
    
    # Filter the second dataframe
    df2_filtered = df2[['DONATIONNUMBER', 'DONATIONDATE']]
    df = df.merge(df2_filtered, on='DONATIONNUMBER', how='inner')
    
    # Drop rows where DONATIONDATE is NaN
    df = df.dropna(subset=['DONATIONDATE'])
    
    # Drop rows where SIZEUOM is blank or NaN
    df = df.dropna(subset=['SIZEUOM'])
    df = df[df['SIZEUOM'].astype(str).str.strip() != '']
    
    # Ensure expiration date column exists
    if 'EXPIRATIONDATE' in df.columns:
        df['TIME'] = (df['EXPIRATIONDATE'] - df['DONATIONDATE']).dt.days
        df = df.drop(columns=['DONATIONDATE', 'EXPIRATIONDATE'])
    else:
        print("Warning: EXPIRATIONDATE column not found in dataset.")
    
    # Select only relevant columns
    independent_variables = ["DONATIONREASON", "SHELFLIFE", "PACKAGINGTYPE", "SIZE_", "SIZEUOM", "STORAGEREQUIREMENT", "PACK", "UNITOFMEASURE"]
    df = df[independent_variables + ['TIME']]

    # Fill missing values using nearest neighbor method
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    print("Dropped columns:", list(columns_to_drop))
    print("Final dataset shape:", df.shape)
    
    return df

def convert_to_one_hot(df):
    """Convert categorical columns to one-hot encoded columns and drop the original columns."""
    # Specify the columns to be one-hot encoded
    categorical_columns = ['DONATIONREASON', 'PACKAGINGTYPE', 'SIZEUOM', 'STORAGEREQUIREMENT', 'UNITOFMEASURE']
    
    # Apply one-hot encoding to the categorical columns
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)  # drop_first avoids multicollinearity

    print("Dropping original categorical columns...")
    
    print(f"One-hot encoded columns: {categorical_columns}")
    print(f"Data shape after one-hot encoding: {df.shape}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Clean and merge dataset for analysis.")
    parser.add_argument('file_path', type=str, help="Path to the Excel file.")
    parser.add_argument('--sheet_lines', type=str, default='AMX_DONATION_LINES', help="Sheet name for donation lines.")
    parser.add_argument('--sheet_header', type=str, default='AMX_DONATION_HEADER', help="Sheet name for donation header.")
    args = parser.parse_args()
    
    df, df2 = load_data(args.file_path, args.sheet_lines, args.sheet_header)
    df_cleaned = clean_and_merge_data(df, df2)
    df_cleaned.to_csv("cleaned_data.csv", index=False)
    print("Cleaned data returned")

    df_one_hot = convert_to_one_hot(df_cleaned)
    
    # Save cleaned and one-hot encoded data
    df_one_hot.to_csv("cleaned_data_one_hot.csv", index=False)
    print("One-hot encoded data saved successfully.")

if __name__ == "__main__":
    main()