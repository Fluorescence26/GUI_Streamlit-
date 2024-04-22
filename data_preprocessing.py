
import pandas as pd
from category_encoders import TargetEncoder
from imblearn.under_sampling import RandomUnderSampler
from utils import encode_targets

def preprocess_dataframe(df):

    # Select only SMILES and Target columns
    print("selectiong [smiles] and [Fluorescent] column...")

    df = df[['Smiles', 'Fluorescent labeling']]

    # Drop duplicates(repeated smiles ,etc)
    
    print("droping repeated smiles...")
    df = df.drop_duplicates()

    print("adding new column where [ fluroscent=1 ] and [ Non-fluroscent=0 ]")


    # Target encoding
    df = encode_targets(df)

    print(f"after preprocessing: rows=[{df.shape[0]}] , columns=[{df.shape[1]}]")

    return df

def preprocess_dataframe_for_regression(df):
    # Select only SMILES and Target columns
    df = df[['Smiles', 'Fluorescent labeling']]

    # Drop duplicates
    df = df.drop_duplicates()

    # Target encoding
    df = encode_targets(df)
    

    # Undersampling
    
    # df = undersampler(df)

    return df