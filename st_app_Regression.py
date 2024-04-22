# Import libraries
import streamlit as st
from rdkit import Chem
import deepchem as dc
from rdkit.Chem.Draw import MolToImage
import joblib
import pandas as pd  # Make sure to import pandas

# Load the regression model
MODEL_PATH = "best_models/new_best_regressor.joblib"
model_regression = joblib.load(MODEL_PATH)

# Calculate RDKit descriptors from SMILES string
def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    featurizer = dc.feat.MACCSKeysFingerprint()
    descriptors = featurizer.featurize([mol])
    return pd.DataFrame(data=descriptors)

# Predict absorption max
def predict_absorption_max(model, smiles, solvent):
    smiles_desc = smiles_to_descriptors(smiles)
    solvent_desc = smiles_to_descriptors(solvent)
    X = pd.concat([smiles_desc, solvent_desc], axis=1)
    y_pred = model.predict(X)
    absorption_max = y_pred[0]
    return absorption_max

# Title
st.title("Regression Model")

# Input for regression model
smiles_input = st.text_input("Enter a SMILES string for the molecule:")
solvent_input = st.text_input("Enter a SMILES string for the solvent:")

# Check if the input is valid and not empty
if smiles_input and solvent_input:
    try:
        # Predict using the regression model
        result = predict_absorption_max(model_regression, smiles_input, solvent_input)

        # Draw molecule structure from SMILES string
        st.image(MolToImage(Chem.MolFromSmiles(smiles_input)), caption="Molecule Structure", width=100, use_column_width=True)

        # Display the prediction result on the app
        st.write(f"Predicted Absorption Max: {result}")

    except Exception as e:
        # Display an error message if the input is invalid or cannot be processed 
        st.error(f"Error in Regression Model: {e}")
