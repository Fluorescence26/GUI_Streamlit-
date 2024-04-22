# Import libraries
import streamlit as st
from rdkit import Chem
import deepchem as dc
from rdkit.Chem.Draw import MolToImage
import joblib
import pandas as pd

# Load models
model_path_fluorescence = "C:/Users/26pde/Downloads/Fluorescent_compound_prediction/new_best_regressor.joblib"
model_fluorescence = joblib.load(model_path_fluorescence)

model_path_regression = "new_best_regressor.joblib"
model_regression = joblib.load(model_path_regression)

# Calculate Morgan fingerprints from SMILES string
def smiles_to_morgan(smiles):
    mol = Chem.MolFromSmiles(smiles) 
    featurizer = dc.feat.CircularFingerprint(radius=3, size=1024)
    fp = featurizer.featurize([mol])
    return fp[0]  # Take the features for the first (and only) sample


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


# Predict some value using a model
def predict(model, fp):
    try:
        print(f"Input shape: {fp.shape}")

        # Check if the model is a deepchem model
        if hasattr(model, 'predict'):
            # If it's a deepchem model, assume it's a classification model
            pred = model.predict([fp])[0]
        else:
            # If not a deepchem model, assume it's a traditional sklearn model
            pred = model.predict(fp)[0]

        print(f"Prediction: {pred}")
        return pred
    except Exception as e:
        print(f"Error in predict: {e}")
        return None

# Draw structure
def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    img = MolToImage(mol)
    return img
# Title
st.title("Model Selector")

# Sidebar for selecting the model
model_selector = st.sidebar.selectbox("Select Model", ["Classification Model", "Regression Model"])

# Tab selection
if model_selector == "Classification Model":
    st.title("Fluorescence Classifier")

    # Input for fluorescence model
    smiles_input = st.text_input("Enter a SMILES string:")

    # Check if the input is valid and not empty
    if smiles_input:
        try:
            # Calculate Morgan fingerprints from SMILES string
            fp = smiles_to_morgan(smiles_input)

            # Predict using the fluorescence model
            result = predict(model_fluorescence, fp)

            # Draw molecule structure from SMILES string
            image = draw_molecule(smiles_input)

            # Display the molecule structure and the prediction result on the app
            st.image(image, caption="Molecule Structure", width=100, use_column_width=True)
            
            # Check the result and display the prediction
            if result is not None:
                st.write(f"Prediction: {'Fluorescent Molecule' if result == 1 else 'Non-fluorescent Molecule'}")

        except Exception as e:
            # Display an error message if the input is invalid or cannot be processed 
            st.error(f"Error in Fluorescence Model: {e}")

else:
    st.title("Regression Model")

    # Input for regression model
    smiles_input = st.text_input("Enter a SMILES string for the molecule:")
    solvent_input = st.text_input("Enter a SMILES string for the solvent:")

    if smiles_input and solvent_input:
        try:
            # Predict using the regression model
            result = predict_absorption_max(model_regression, smiles_input, solvent_input)

            # Draw molecule structure from SMILES string
            st.image(MolToImage(Chem.MolFromSmiles(smiles_input)), caption="Molecule Structure", width=100, use_column_width=True)

            # Display the prediction result on the app
            st.write(f"Predicted Absorption Max: {round(result, 2)}")

        except Exception as e:
            # Display an error message if the input is invalid or cannot be processed 
            st.error(f"Error in Regression Model: {e}")
