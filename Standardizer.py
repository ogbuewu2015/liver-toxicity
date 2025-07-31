from rdkit import Chem
from molvs import Standardizer, normalize



class MyStandardizer:
    def __init__(self):
        self.standardizer = Standardizer()

        # Modify the normalization rules
        norms = list(normalize.NORMALIZATIONS)
        for i in range(len(norms) - 1, 0, -1):
            if norms[i].name == "Sulfoxide to -S+(O-)":
                del norms[i]
        norms.append(normalize.Normalization("[S+]-[O-] to S=O", "[S+:1]([O-:2])>>[S+0:1](=[O-0:2])"))

        # Set the modified normalization rules
        self.standardizer.normalizations = norms

        # Set the prefer_organic option to True
        self.standardizer.prefer_organic = True


    
    def standardize_mol(self, smiles_list):
        standardized_mol_list = []  
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is not None:
                # Standardize the molecule
                standardized_mol = self.standardizer.standardize(mol)
                standardized_mol_list.append(standardized_mol)
               
            else:
                print("Invalid RDKit Mol object.")
        return standardized_mol_list
    
    
    
    
    

    def standardize_smiles(self, smiles_list):
        standardized_smiles_list = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is not None:
                # Standardize the molecule
                standardized_mol = self.standardizer.standardize(mol)
                # Get the canonical tautomer
                standardized_smiles = Chem.MolToSmiles(standardized_mol)
                standardized_smiles_list.append(standardized_smiles)
            else:
                print(f"Unable to parse SMILES: {smiles}")

        return standardized_smiles_list