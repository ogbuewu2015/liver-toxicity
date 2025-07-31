import pandas as pd
import numpy as np
import joblib
import csv
from preprocessing_car import scale_new_data
import seaborn as sn
import pickle
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
#importing the cheminformatics module needed
import molvs

import matplotlib.pyplot as plt
from preprocessing_mito import scale_new_data
from molvs import Standardizer, normalize
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm
import rdkit    
from Mold2_pywrapper import Mold2
import streamlit as st
import pandas as pd
import joblib
from preprocessing_car import scale_new_data
from kpca_utils_car import split_columns, fit_kernel_pca, transform_with_kernel_pca, concat_transformed_with_nonfloat
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import streamlit as st
import pandas as pd
import ast
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize as Standardizer
from rdkit.Chem.SaltRemover import SaltRemover
from Standardizer import MyStandardizer
from molvs import Standardizer as MolVSStandardizer
from kpca_utils_car import split_columns, fit_kernel_pca, transform_with_kernel_pca, concat_transformed_with_nonfloat
from kpca_utils_mito import split_columns, fit_kernel_pca, transform_with_kernel_pca, concat_transformed_with_nonfloat


with open('car_model.pkl', 'rb') as f:
    car_model = pickle.load(f)

with open('mito_model.pkl', 'rb') as f:
    mito_model = pickle.load(f)

with open('calibrated_car_model.pkl', 'rb') as f:
    cal_car_model = pickle.load(f)

with open('calibrated_mito_model.pkl', 'rb') as f:
    cal_mito_model = pickle.load(f)


kpca_model_car = joblib.load("kpca_model_car.pkl")
kpca_model_mito = joblib.load("kpca_model_mito.pkl")

    



def get_weighted_predictions(model_mito, model_car, X_mito, X_car, logloss_mito, logloss_car, threshold=0.5):
    """
    Returns the weighted prediction probabilities and class labels 
    for two calibrated models based on their log loss.

    Parameters:
    - model1, model2: Calibrated sklearn models
    - X1, X2: Feature sets for model1 and model2 respectively
    - logloss1, logloss2: Log loss values (lower is better)
    - threshold: Decision threshold for classification

    Returns:
    - weighted_proba: Weighted average of class 1 probabilities
    - weighted_pred: Binary predictions based on threshold
    - weights: (model1_weight, model2_weight)
    """

    # Compute inverse log loss weights
    w_mito = 1 / logloss_mito
    w_car = 1 / logloss_car
    total = w_mito + w_car
    model_mito_weight = w_mito / total
    model_car_weight = w_car / total

    # Predict probabilities
    proba_mito = model_mito.predict_proba(X_mito)[:, 1]
    proba_car = model_car.predict_proba(X_car)[:, 1]

    # Weighted average
    weighted_proba = (model_mito_weight * proba_mito) + (model_car_weight * proba_car)
    weighted_pred = (weighted_proba >= threshold).astype(int)

    return weighted_proba, weighted_pred, (model_mito_weight, model_car_weight)



# === Heavy Metals List ===
heavy_metals_atomic_nums = [
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
    82, 83, 84, 85, 86, 87, 88,
    64, 33, 50
]

# === Preprocessing Functions ===

def remove_heavy_metals(df):
    def has_no_heavy_metals(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        return not any(atom.GetAtomicNum() in heavy_metals_atomic_nums for atom in mol.GetAtoms())
    return df[df['smiles'].apply(has_no_heavy_metals)]

def strip_isotopes(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    return Chem.MolToSmiles(mol, isomericSmiles=True)

metal_disconnector = MolVSStandardizer().disconnect_metals

def disconnect_metals_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = metal_disconnector(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)

# === Solvent Definitions ===
solvents = {
    'Water': 'O', 'Methanol': 'CO', 'Ethanol': 'CCO', 'Isopropanol': 'CC(O)C',
    'Acetic Acid': 'CC(=O)O', 'Formic Acid': 'C(=O)O', 'DMSO': 'CS(=O)C', 'Acetonitrile': 'CC#N',
    'DMF': 'CN(C)C=O', 'Acetone': 'CC(=O)C', 'THF': 'C1CCOCC1', 'Ethyl Acetate': 'CCOC(=O)C',
    'MEK': 'CCC(=O)C', 'Hexane': 'CCCCCC', 'Toluene': 'CC1=CC=CC=C1', 'Diethyl Ether': 'CCOCC',
    'Chloroform': 'ClC(Cl)Cl', 'Benzene': 'C1=CC=CC=C1', 'Cyclohexane': 'C1CCCCC1',
    'Dichloromethane': 'ClCCl', 'Carbon Tetrachloride': 'ClC(Cl)(Cl)Cl', '1,2-Dichloroethane': 'ClCCCl'
}

solvent_mols = {name: Chem.MolFromSmiles(smiles) for name, smiles in solvents.items()}

def remove_solvents(compound_smiles, solvent_mols):
    mol = Chem.MolFromSmiles(compound_smiles)
    if not mol:
        return None

    for solvent_mol in solvent_mols.values():
        if Chem.MolToSmiles(mol) == Chem.MolToSmiles(solvent_mol):
            return compound_smiles

    fragments = Chem.GetMolFrags(mol, asMols=True)
    non_solvent_fragments = []
    for fragment in fragments:
        is_solvent = any(Chem.MolToSmiles(fragment) == Chem.MolToSmiles(solv) for solv in solvent_mols.values())
        if not is_solvent:
            non_solvent_fragments.append(fragment)

    if not non_solvent_fragments:
        return None

    rwmol = Chem.RWMol(Chem.CombineMols(non_solvent_fragments[0], non_solvent_fragments[0]))
    for frag in non_solvent_fragments[1:]:
        rwmol = Chem.CombineMols(rwmol, frag)

    try:
        Chem.SanitizeMol(rwmol)
    except Exception:
        return None

    return Chem.MolToSmiles(rwmol)

# === Salt Removal and Sanitization ===
def remove_salts_and_sanitize(smiles):
    salt_remover = SaltRemover()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    cleaned_mol = salt_remover.StripMol(mol, dontRemoveEverything=True)
    if cleaned_mol is None or cleaned_mol.GetNumAtoms() == 0:
        return smiles
    try:
        Chem.SanitizeMol(cleaned_mol)
    except Exception:
        return smiles
    return Chem.MolToSmiles(cleaned_mol)

from rdkit.Chem import Descriptors

# Filter function to validate carbon count and molecular weight
def filter_dataframe(df):
    def is_valid_smiles(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False  # Invalid SMILES
        
        num_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
        if num_carbons < 4:
            return False
        
        if Descriptors.MolWt(mol) > 909:
            return False
        
        return True
    
    return df['smiles'].apply(is_valid_smiles)



# Initialize custom and MolVS standardizers
Standardizer = MyStandardizer()
standardizer = MolVSStandardizer()

# Define the final preprocessing function
def preprocess_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    uncharged_mol = standardizer.uncharge(mol)
    canonical_tautomer_mol = standardizer.canonicalize_tautomer(uncharged_mol)
    return Chem.MolToSmiles(canonical_tautomer_mol)


# Define the columns to convert after descriptor generation
columns_to_convert = ['D018', 'D326', 'D330', 'D340']

def convert_columns_to_int64(df):
    for col in columns_to_convert:
        if col in df.columns:
            df[col] = df[col].astype('int64')

# Define the columns to convert after descriptor generation
columns_to_convert = ['D018', 'D326', 'D330', 'D340']


# Function to check if a SMILES contains disconnected parts
def check_disconnected_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # Check if the molecule has multiple fragments
        fragments = Chem.GetMolFrags(mol, asMols=False)
        return len(fragments) > 1
    else:
        print(f"Invalid SMILES: {smiles}")
        return False

def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    
    if mol is not None:
        # Convert the sanitized molecule to canonical SMILES
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        return canonical_smiles
    else:
        print(f"Unable to parse SMILES: {smiles}")
        # Log problematic SMILES
        return None
    
    
def generate_mold2_descriptors(smiles_series):
    # Initialize Mold2
    mold2 = Mold2()

    # Convert SMILES to Mol objects
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_series]

    # Calculate descriptors
    descriptors = mold2.calculate(mols)

    # Convert the result into a DataFrame if needed
    descriptors_df = pd.DataFrame(descriptors)

    return descriptors_df


# Load preprocessing objects
@st.cache_resource
def load_preprocessing_objects_car():
    selector = joblib.load("selector_car.pkl")
    scaler = joblib.load("scaler_car.pkl")
    float_cols = joblib.load("float_cols_car.pkl")
    retained_features_final = joblib.load("retained_features_final_car.pkl")
    return selector, scaler, float_cols, retained_features_final

# Define preprocessing function for new data
def preprocess_new_data_car(raw_input_df):
    selector, scaler, float_cols, retained_features_final = load_preprocessing_objects_car()

    # Step 1: Apply variance threshold selection
    retained_columns = raw_input_df.columns[selector.get_support(indices=True)]
    var_filtered_df = raw_input_df[retained_columns]

    # Step 2: Scale float features
    scaled_df = scale_new_data(var_filtered_df, scaler, float_cols)

    # Step 3: Apply correlation-based feature selection
    final_df = scaled_df[retained_features_final]

    return final_df


# Load preprocessing objects
@st.cache_resource
def load_preprocessing_objects_mito():
    selector = joblib.load("selector_mito.pkl")
    scaler = joblib.load("scaler_mito.pkl")
    float_cols = joblib.load("float_cols_mito.pkl")
    retained_features_final = joblib.load("retained_features_final_mito.pkl")
    return selector, scaler, float_cols, retained_features_final

# Define preprocessing function for new data
def preprocess_new_data_mito(raw_input_df):
    selector, scaler, float_cols, retained_features_final = load_preprocessing_objects_mito()

    # Step 1: Apply variance threshold selection
    retained_columns = raw_input_df.columns[selector.get_support(indices=True)]
    var_filtered_df = raw_input_df[retained_columns]

    # Step 2: Scale float features
    scaled_df = scale_new_data(var_filtered_df, scaler, float_cols)

    # Step 3: Apply correlation-based feature selection
    final_df = scaled_df[retained_features_final]

    return final_df




        

# Custom list of important features to retain after preprocessing
important_features_car = [
    'D460', 'D282', 'D746', 'D300', 'D279', 'D748', 'D452', 'D729', 'D715',
    'D515', 'D005', 'D468', 'D187', 'D485', 'D547', 'D507', 'D499', 'D131',
    'D508', 'D713', 'D588', 'D372', 'D325', 'D252', 'D336', 'D053', 'D034',
    'D027', 'D339', 'D001', 'D768', 'D708', 'D394', 'D380', 'D718', 'D389',
    'D745', 'D556', 'D004', 'D582', 'D338', 'D502', 'D680', 'D589', 'D014',
    'D552', 'D596', 'D186', 'D539', 'D546', 'D550', 'D494', 'D477', 'D505'
]




important_features_mito = ['D461', 'D547', 'D450', 'D279', 'D469', 'D508', 'D499', 'D459', 'D199', 'D484', 'D506', 'D501', 'D556', 'D562', 'D503', 'D509', 'D505', 'D500', 'D492', 'D476', 'D451', 'D510', 'D470', 'D477', 'D460', 'D282', 'D493', 'D447', 'D462', 'D561', 'D453', 'D777', 'D197', 'D502', 'D173', 'D468', 'D551', 'D535', 'D265', 'D540', 'D507', 'D582', 'D719', 'D123', 'D480', 'D738', 'D478', 'D542', 'D454', 'D597', 'D533', 'D534', 'D739', 'D485', 'D680', 'D195', 'D187', 'D560', 'D732', 'D713', 'D709', 'D559', 'D765', 'D588', 'D483', 'D550', 'D775', 'D338', 'D754', 'D367', 'D259', 'D552', 'D186', 'D715', 'D351', 'D272', 'D360', 'D176', 'D604', 'D131', 'D532', 'D366', 'D600', 'D300', 'D016', 'D337', 'D486', 'D675', 'D379', 'D323', 'D005', 'D599', 'D714', 'D590', 'D752', 'D650', 'D475', 'D721', 'D744', 'D689', 'D668', 'D677', 'D004', 'D717', 'D377', 'D252', 'D413', 'D375', 'D335', 'D716', 'D130', 'D724', 'D648', 'D742', 'D768', 'D393', 'D411', 'D325', 'D397', 'D575', 'D339', 'D399', 'D398', 'D330', 'D382', 'D327', 'D373', 'D725', 'D381', 'D322', 'D380', 'D601', 'D308', 'D679', 'D750', 'D392', 'D376', 'D034', 'D390', 'D606', 'D571', 'D536', 'D774', 'D385', 'D511', 'D589', 'D035', 'D576', 'D602', 'D537', 'D389', 'D610', 'D621', 'D539', 'D674', 'D378', 'D544', 'D122', 'D546', 'D001', 'D545', 'D541', 'D384', 'D538', 'D570']



def compute_conformal_prediction(calib_proba, test_proba):
    calib_nonconformity = 1 - calib_proba
    test_nonconformity = 1 - test_proba

    p_values = np.array([
        (np.sum(calib_nonconformity >= score) + 1) / (len(calib_nonconformity) + 1)
        for score in test_nonconformity
    ])

    threshold = np.max(calib_nonconformity)
    accepted = test_nonconformity <= threshold

    return p_values, accepted

#function to generate csv file
def file_download(data, file):
    df = data.to_csv(index=False)
    f=base64.b64encode(df.encode()).decode()
    link= f'<a href = "data:file/csv: base64,{f}" download={file}> Download{file} file</a>'
    return link






st.set_page_config(page_title='Drug Induced Hepatotoxicity Prediction App', layout='wide')

# Styled sidebar header
st.sidebar.markdown(
    '<h2 style="color: white; background-color: #6A1B9A; padding: 12px; border-radius: 10px; text-align: center;">'
    'Use this Sidebar for Hepatotoxicity Prediction</h2>',
    unsafe_allow_html=True
)

# Developer credit section
st.markdown(
    """
    <div style="background-color:#F3E5F5; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
        <p style="color:#4A148C; font-size: 16px;">
            This Web Application was developed by 
            <a href="https://www.linkedin.com/in/emmanuel-ogbuewu-18a3a4117/" target="_blank" style="color:#1A237E; text-decoration: none;">
                Emmanuel I. Ogbuewu
            </a>, a PhD student of Dr. Jeremy S. Edwards at the University of New Mexico.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Description of the app’s purpose
st.markdown(
    """
    <div style="background-color:#E8EAF6; padding: 15px; border-radius: 10px;">
        <p style="color:#1A237E; font-size: 16px;">
            Hepatotoxicity, or drug-induced liver injury, is a major reason for drug withdrawals and clinical trial failures. 
            Predicting liver toxicity early is critical for ensuring patient safety and improving drug development success. 
            Our tool helps identify potential hepatotoxic risks of drug candidates using advanced machine learning. 
            See the chemical space visualization of the training compounds below.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


def plot_pca_subplots(train_df_1, dili_df_1,
                      train_df_2, dili_df_2,
                      title1="Subplot 1", title2="Subplot 2"):
    """
    Plots two subplots showing PCA_1 vs PCA_2 for two dataset pairs.

    Parameters:
    - train_df_1: DataFrame with PCA_1 and PCA_2 columns (e.g., X_train_vis_mito)
    - dili_df_1: DataFrame with PCA_1 and PCA_2 columns (e.g., X_DILI_train_vis_mito)
    - train_df_2: DataFrame with PCA_1 and PCA_2 columns (e.g., X_train_vis_car)
    - dili_df_2: DataFrame with PCA_1 and PCA_2 columns (e.g., X_DILI_train_vis_car)
    - title1: str, title for the first subplot
    - title2: str, title for the second subplot
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1 (e.g., Mito)
    ax1.scatter(train_df_1["PCA_1"], train_df_1["PCA_2"], color='purple', label="Train", alpha=0.7)
    ax1.scatter(dili_df_1["PCA_1"], dili_df_1["PCA_2"], color='red', label="DILI", alpha=0.7)
    ax1.set_title(title1)
    ax1.set_xlabel("Kernel PCA 1")
    ax1.set_ylabel("Kernel PCA 2")
    ax1.legend()

    # Subplot 2 (e.g., CAR)
    ax2.scatter(train_df_2["PCA_1"], train_df_2["PCA_2"], color='blue', label="Train", alpha=0.7)
    ax2.scatter(dili_df_2["PCA_1"], dili_df_2["PCA_2"], color='red', label="DILI", alpha=0.7)
    ax2.set_title(title2)
    ax2.set_xlabel("Kernel PCA 1")
    ax2.set_ylabel("Kernel PCA 2")
    ax2.legend()

    plt.tight_layout()
    plt.show()


    
    
    
    
    
X_train_vis_mito = pd.read_csv('X_train_vis_mito.csv')
X_DILI_train_vis_mito = pd.read_csv('X_DILI_train_vis_mito.csv')
X_train_vis_car= pd.read_csv('X_train_vis_car.csv')
X_DILI_train_vis_car = pd.read_csv('X_DILI_train_vis_car.csv')



    
plot_pca_subplots(
    X_train_vis_mito, X_DILI_train_vis_mito,
    X_train_vis_car, X_DILI_train_vis_car,
    title1="Mitotox Train Data vs DILI Train Data", title2="CAR Antagonist Train Data vs DILI Train Data"
)





one_or_few_SMILES = st.sidebar.text_input('Enter single SMILE strings in single or double quotation separated by comma:', "['CCO']")
st.sidebar.markdown('''"or upload SMILE strings in CSV format, note that the SMILE strings of the molecules should be in 'smiles' column:"''')
many_SMILES = st.sidebar.file_uploader("================================")

st.sidebar.markdown("""**if you upload the csv file, click the button below to get the hepatotoxicity prediction**""")
predict_button = st.sidebar.button("Predict Drug Induced Hepatotoxicity")

    




# === Main Logic ===
if one_or_few_SMILES != "['CCO']":
    try:
        smiles_list = ast.literal_eval(one_or_few_SMILES)
        df = pd.DataFrame(smiles_list, columns=['smiles'])

        # Step 1-11: Preprocess input SMILES
        df = run_full_preprocessing_pipeline(df, solvent_mols)

        # Step 12: Generate Mold2 descriptors
        descriptors_df = generate_mold2_descriptors(df['smiles'])
        convert_columns_to_int64(descriptors_df)
        st.success("✅ Mold2 descriptors generated.")
        st.dataframe(descriptors_df.head())

        # Step 13: Preprocess for CAR and MITO
        processed_car_df = preprocess_new_data_car(descriptors_df)
        processed_mito_df = preprocess_new_data_mito(descriptors_df)
        st.success("✅ Preprocessing complete for CAR and MITO.")

        # Step 14: Select important features
        selected_car_df = processed_car_df[[col for col in important_features_car if col in processed_car_df.columns]]
        selected_mito_df = processed_mito_df[[col for col in important_features_mito if col in processed_mito_df.columns]]
        st.success("✅ Selected important features for CAR and MITO.")

        # Step 15: Predict class 1 probabilities
        car_probs = car_model.predict_proba(selected_car_df)[:, 1]
        mito_probs = mito_model.predict_proba(selected_mito_df)[:, 1]

        df['CAR_Toxicity_Probability'] = car_probs
        df['MITO_Toxicity_Probability'] = mito_probs

        st.success("✅ Cytoxicity probabilities predicted.")
        # Step 15: KernelPCA transformation
        car_float, car_nonfloat = split_columns(selected_car_df)
        mito_float, mito_nonfloat = split_columns(selected_mito_df)

        car_kpca_transformed = transform_with_kernel_pca(car_float, kpca_model_car)
        mito_kpca_transformed = transform_with_kernel_pca(mito_float, kpca_model_mito)

        selected_car_data = concat_transformed_with_nonfloat(car_kpca_transformed, car_nonfloat, index=selected_car_df.index)
        selected_mito_data = concat_transformed_with_nonfloat(mito_kpca_transformed, mito_nonfloat, index=selected_mito_df.index)
        st.success("✅ Cytotoxicity data aligned with hepatotoxic data using kernel PCA.")
        calibrate_mito = pd.read_csv(X_DILI_calibrate_mito.csv)
        calibrate_car = pd.read_csv(X_DILI_calibrate_mito.car)
                
        weighted_calibrate_proba, weighted_calibrate_pred, _ = get_weighted_predictions(cal_mito_model, cal_car_model,
                                                                                                calibrate_mito_data, calibrate_car_data,
                                                                                                logloss_mito=0.293462, logloss_car=0.443863)
        weighted_test_proba, weighted_test_pred, weights = get_weighted_predictions(cal_mito_model, cal_car_model,
                                                                                            selected_mito_data, selected_car_data,
                                                                                            logloss_mito=0.293462, logloss_car=0.443863)
        df['Weighted_Toxicity_Probability'] = weighted_test_proba
        df['Predicted_Label'] = weighted_test_pred

                # Step 18: Conformal prediction
        p_values, accepted = compute_conformal_prediction(weighted_calibrate_proba, weighted_valid_proba)
        df['Conformal_PValue'] = p_values
        df['Accepted'] = accepted

        st.success("✅ Weighted predictions using calibrated models and conformal prediction completed.")
        st.dataframe(df2[['smiles','CAR_Toxicity_Probability', 'MITO_Toxicity_Probability', 'Weighted_Toxicity_Probability',
                                  'Predicted_Label', 'Conformal_PValue', 'Accepted']])
        output = df2[['smiles','CAR_Toxicity_Probability', 'MITO_Toxicity_Probability', 'Weighted_Toxicity_Probability',
                                  'Predicted_Label', 'Conformal_PValue', 'Accepted']]
        st.sidebar.markdown('''## See your output in the following table:''')
        st.sidebar.write(output)
        st.sidebar.markdown(file_download(output, "hepatotoxicity_prediction.csv"), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"⚠️ Failed to process input: {e}")

elif predict_button:
    if many_SMILES is not None:
        try:
            df2 = pd.read_csv(many_SMILES)
            if 'smiles' not in df2.columns:
                st.error("❌ Uploaded CSV must contain a 'smiles' column.")
            else:
                # Step 1-11: Preprocess uploaded SMILES
                df2 = run_full_preprocessing_pipeline(df2, solvent_mols)

                # Step 12: Generate Mold2 descriptors
                descriptors_df2 = generate_mold2_descriptors(df2['smiles'])
                convert_columns_to_int64(descriptors_df2)
                st.success("✅ Mold2 descriptors generated for uploaded data.")
                st.dataframe(descriptors_df2.head())

                # Step 13: Preprocess for CAR and MITO
                processed_car_df = preprocess_new_data_car(descriptors_df2)
                processed_mito_df = preprocess_new_data_mito(descriptors_df2)
                st.success("✅ Preprocessing complete for CAR and MITO.")

                # Step 14: Select important features
                selected_car_df = processed_car_df[[col for col in important_features_car if col in processed_car_df.columns]]
                selected_mito_df = processed_mito_df[[col for col in important_features_mito if col in processed_mito_df.columns]]
                st.success("✅ Selected important features for CAR and MITO.")

                # Step 15: Predict class 1 probabilities
                car_probs = car_model.predict_proba(selected_car_df)[:, 1]
                mito_probs = mito_model.predict_proba(selected_mito_df)[:, 1]

                df2['CAR_Toxicity_Probability'] = car_probs
                df2['MITO_Toxicity_Probability'] = mito_probs

                st.success("✅ Class 1 toxicity probabilities predicted.")
                # Step 15: KernelPCA transformation
                car_float, car_nonfloat = split_columns(selected_car_df)
                mito_float, mito_nonfloat = split_columns(selected_mito_df)

                car_kpca_transformed = transform_with_kernel_pca(car_float, kpca_model_car)
                mito_kpca_transformed = transform_with_kernel_pca(mito_float, kpca_model_mito)

                selected_car_data = concat_transformed_with_nonfloat(car_kpca_transformed, car_nonfloat, index=selected_car_df.index)
                selected_mito_data = concat_transformed_with_nonfloat(mito_kpca_transformed, mito_nonfloat, index=selected_mito_df.index)
                st.success("✅ Cytotoxicity data aligned with hepatotoxic data using kernel PCA.")
                
                calibrate_mito = pd.read_csv(X_DILI_calibrate_mito.csv)
                calibrate_car = pd.read_csv(X_DILI_calibrate_mito.car)
                
                weighted_calibrate_proba, weighted_calibrate_pred, _ = get_weighted_predictions(cal_mito_model, cal_car_model,
                                                                                                calibrate_mito_data, calibrate_car_data,
                                                                                                logloss_mito=0.293462, logloss_car=0.443863)
                weighted_test_proba, weighted_test_pred, weights = get_weighted_predictions(cal_mito_model, cal_car_model,
                                                                                            selected_mito_data, selected_car_data,
                                                                                            logloss_mito=0.293462, logloss_car=0.443863)
                df2['Weighted_Toxicity_Probability'] = weighted_test_proba
                df2['Predicted_Label'] = weighted_test_pred

                # Step 18: Conformal prediction
                p_values, accepted = compute_conformal_prediction(weighted_calibrate_proba, weighted_valid_proba)
                df2['Conformal_PValue'] = p_values
                df2['Accepted'] = accepted

                st.success("✅ Weighted predictions using calibrated models and conformal prediction completed.")
                st.dataframe(df2[['smiles','CAR_Toxicity_Probability', 'MITO_Toxicity_Probability', 'Weighted_Toxicity_Probability',
                                  'Predicted_Label', 'Conformal_PValue', 'Accepted']])
                output = df2[['smiles','CAR_Toxicity_Probability', 'MITO_Toxicity_Probability', 'Weighted_Toxicity_Probability',
                                  'Predicted_Label', 'Conformal_PValue', 'Accepted']]
                st.sidebar.markdown('''## See your output in the following table:''')
                st.sidebar.write(output)
                st.sidebar.markdown(file_download(output, "hepatotoxicity_prediction.csv"), unsafe_allow_html=True)
 

        except Exception as e:
            st.error(f"⚠️ Failed to process uploaded CSV: {e}")
    else:
        st.warning("⚠️ Please upload a CSV file to continue.")

else:
    st.info("ℹ️ Please input or upload SMILES data and Click on 'Predict Drug Induced Hepatotoxicity'.")

    st.markdown("""
    <div style="border: 2px solid #4A148C; border-radius: 15px; padding: 20px; text-align: center; background-color: #f4f1fa;">
        <h5 style="color: #2d0a61;">
            This predictive framework consists of three models: one for mitochondrial toxicity and another for CAR antagonist cytotoxicity, both trained on Tox21 datasets.
            The third model is built on DILIrank data (548 training and 117 calibration compounds, verified from literature).
            The Tox21-derived models are independently aligned with the DILIrank dataset using an unsupervised KernelPCA approach to support hepatotoxicity prediction.
        </h5>
        <h5 style="color: white; background-color: #6A1B9A; border-radius: 10px; padding: 15px; margin-top: 20px; opacity: 0.9;">
            The optimal classification thresholds, based on the Youden J statistic from validation data, are 0.59 for the mitochondrial toxicity model and 0.62 for the CAR antagonist cytotoxicity model.
            DILIrank hepatotoxicity predictions are considered more reliable when the associated p-value is accepted, indicating the prediction lies within the applicability domain of the three models.
        </h5>
    </div>
    """, unsafe_allow_html=True)



