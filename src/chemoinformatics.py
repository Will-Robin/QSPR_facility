from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=False,
    )


def get_elements(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return set()

    return {atom.GetSymbol() for atom in mol.GetAtoms()}
