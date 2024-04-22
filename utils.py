import numpy as np
import pandas as pd

def encode_targets(df):
    """
    Encodes the target column of the dataframe.
    as well as handle imbalanced data
    """

    # create "target" column where if 'Fluorescent labeling' column is "yes" the add "1" otherwise "0"
    df['target'] = [ 1 if typ == 'Yes' else 0 for typ in df['Fluorescent labeling']]

    # how many molecules are fluoresscent
    active = len(df[df['Fluorescent labeling'] == "Yes"])

    # how many molecules are not fluorescent
    inactive = df[df['Fluorescent labeling'] == "No "].index

    # randomly sample inactive molecules
    random_indices = np.random.choice(inactive,active, replace=False)

    # get indexes of active molecules
    active_indices = df[df['Fluorescent labeling'] == "Yes"].index

    # add indexes of active and inactive
    under_sample_indices = np.concatenate([active_indices,random_indices])

    # make df according to indexes we got earlier 
    df = df.loc[under_sample_indices]

    #  select imp columns
    df= df[['Smiles',"target"]]
    
    return df