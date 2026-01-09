"""
Data loading utilities for phonon band prediction
"""

import os
import json
import pandas as pd
import numpy as np
from ase import Atoms
from pymatgen.core.structure import Structure
import glob

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def load_band_structure_data(data_dir, raw_dir, data_file):
    """
    Load band structure data from pkl file or create from JSON files.
    
    Args:
        data_dir (str): Directory to save/load data files.
        raw_dir (str): Directory containing raw JSON files.
        data_file (str): Name of the data file.
        
    Returns:
        pd.DataFrame: DataFrame containing band structure data.
    """
    data_path = os.path.join(data_dir, data_file)
    
    if len(glob.glob(data_path)) == 0:
        # Load from JSON files and save to pkl
        df = pd.DataFrame({})
        for file_path in glob.glob(os.path.join(raw_dir, '*.json')):
            Data = dict()
            with open(file_path) as f:
                data = json.load(f)
            
            structure = Structure.from_str(data['metadata']['structure'], fmt='cif')
            atoms = Atoms(
                list(map(lambda x: x.symbol, structure.species)),
                positions=structure.cart_coords.copy(),
                cell=structure.lattice.matrix.copy(), 
                pbc=True
            )
            
            Data['id'] = data['metadata']['material_id']
            Data['structure'] = [atoms]
            Data['qpts'] = [np.array(data['phonon']['qpts'])]
            Data['real_band'] = [np.array(data['phonon']['ph_bandstructure'])]
            
            dfn = pd.DataFrame(data=Data)
            df = pd.concat([df, dfn], ignore_index=True)
        
        df.to_pickle(data_path)
        print(f"Saved data to {data_path}")
        return df
    else:
        # Load from existing pkl file
        return pd.read_pickle(data_path)