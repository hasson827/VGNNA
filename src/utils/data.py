"""
Data processing utilities for kMVN
Integrates qpoints and bands processing from Bandformer
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from ase.neighborlist import neighbor_list
import mendeleev as md
from ase import Atom
import math

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Initialize atomic properties
class MD:
    """Class to store atomic properties"""
    def __init__(self):
        self.radius, self.pauling, self.ie, self.dip = {}, {}, {}, {}
        for atomic_number in range(1, 119):
            ele = md.element(atomic_number)
            self.radius[atomic_number] = ele.atomic_radius
            self.pauling[atomic_number] = ele.en_pauling
            ie_dict = ele.ionenergies
            self.ie[atomic_number] = ie_dict[min(list(ie_dict.keys()))] if len(ie_dict)>0 else 0
            self.dip[atomic_number] = ele.dipole_polarizability

md_class = MD()


class GaussianBasis(nn.Module):
    """Gaussian basis for edge length encoding (50 dimensions, reference Bandformer)"""
    def __init__(self, start=0.0, end=8.0, step=0.2):
        super().__init__()
        self.start = start
        self.end = end
        self.step = step
        self.num_basis = int((end - start) / step) + 1  # 50 dimensions

    def forward(self, x):
        values = torch.linspace(self.start, self.end, self.num_basis, dtype=x.dtype, device=x.device)
        diff = (x[..., None] - values) / self.step
        return diff.pow(2).neg().exp().div(1.12)


def atom_feature(atomic_number: int, descriptor):
    """Get atomic features based on descriptor."""
    if descriptor == 'mass':  # Atomic Mass (amu)
        feature = Atom(atomic_number).mass
    elif descriptor == 'number':  # atomic number
        feature = atomic_number
    else:
        if descriptor == 'radius':    # Atomic Radius (pm)
            feature = md_class.radius[atomic_number]
        elif descriptor == 'en': # Electronegativity (Pauling)
            feature = md_class.pauling[atomic_number]
        elif descriptor == 'ie':  # Ionization Energy (eV)
            feature = md_class.ie[atomic_number]
        elif descriptor == 'dp':  # Dipole Polarizability (Å^3)
            feature = md_class.dip[atomic_number]
        else:   # no feature
            feature = 1
    return feature


def create_node_input(atomic_numbers: list, n=None, descriptor='mass'):
    """Create node input features for a list of atomic numbers."""
    x = []
    temp = []
    for atomic_number in atomic_numbers:
        atomic = [0.0] * 118         
        atomic[atomic_number - 1] = atom_feature(int(atomic_number), descriptor)
        x.append(atomic)
        temp += [atomic] * len(atomic_numbers)
    if n is not None:
        x += temp * n
    return torch.from_numpy(np.array(x, dtype=np.float64))


def doub(array):
    """Concatenate an array with itself along axis 0."""
    return np.concatenate([array]*2, axis=0)


class UnitCellShift:
    """Class to handle unit cell shifts based on provided shift lengths."""
    def __init__(self, shift_lengths):
        self.shift_lengths = shift_lengths
        self.shift_arrays = [np.array(list(range(length))[int((length-1)/2):]+list(range(length))[:int((length-1)/2)]) - int((length-1)/2) for length in shift_lengths]
        self.shift_indices = np.array(list(range(np.prod(self.shift_lengths)))).reshape(self.shift_lengths)
        self.shift_reverse = np.meshgrid(*self.shift_arrays, indexing='ij')
        self.shift_reverse = np.concatenate([shift.reshape((-1, 1)) for shift in self.shift_reverse], axis=1)


def create_virtual_nodes_kmvn(edge_src, edge_dst, edge_shift, edge_vec, edge_len):
    """Create virtual nodes for 'kmvn' method."""
    N = max(edge_src) + 1
    shift_lengths = np.max(edge_shift, axis=0) - np.min(edge_shift, axis=0) + 1
    ucs = UnitCellShift(shift_lengths)
    shift_dst = ucs.shift_indices[tuple(np.array(edge_shift).T.tolist())]
    return doub(edge_src), np.concatenate([edge_dst, N * (edge_dst + 1) + edge_src + N ** 2 * shift_dst], axis=0), doub(edge_shift), doub(edge_vec), doub(edge_len), ucs


def get_node_deg(edge_dst, n):
    """Compute node degrees from destination edges."""
    node_deg = np.zeros((n, 1), dtype=np.float64)
    for dst in edge_dst:
        node_deg[dst] += 1
    node_deg += node_deg == 0
    return torch.from_numpy(node_deg)


def build_data_kmvn(mpid, structure, real, qpts, r_max, descriptor='mass', factor=1000, **kwargs):
    """
    Build data object for kMVN graph-based learning model.
    Integrates qpoints and bands processing from Bandformer.
    
    Args:
        mpid (str): Material project ID.
        structure (ase.atoms.Atoms): Atomic structure.
        real (np.ndarray): Real values (band structure).
        qpts (np.ndarray): q-points [num_qpts, 3].
        r_max (float): Cutoff radius for neighbor list.
        descriptor (str): Descriptor for node features.
        factor (int): Scaling factor for real values.
        
    Returns:
        torch_geometric.data.Data: Data object for PyTorch Geometric.
    """
    # Build graph structure
    edge_src, edge_dst, edge_shift, edge_vec, edge_len = neighbor_list(
        "ijSDd", a=structure, cutoff=r_max, self_interaction=True
    )
    
    # Create virtual nodes for kMVN
    edge_src, edge_dst, edge_shift, edge_vec, edge_len, ucs = create_virtual_nodes_kmvn(
        edge_src, edge_dst, edge_shift, edge_vec, edge_len
    )
    
    numb = len(structure)
    len_usc = len(ucs.shift_reverse)
    
    # Create node features
    z = create_node_input(structure.arrays['numbers'], len_usc, descriptor='one_hot')
    x = create_node_input(structure.arrays['numbers'], len_usc, descriptor=descriptor)
    
    # Node degrees
    node_deg = get_node_deg(edge_dst, len(x))
    
    # Process qpoints: keep original format for kMVN (no resampling)
    # qpts shape: [num_qpts, 3] - original format for kMVN
    qpts_tensor = torch.from_numpy(qpts.copy())
    
    # Process bands: keep original format to match qpts dimension
    # bands shape: [num_qpts, num_bands] - kept as-is
    # Note: get_spectra() output dimension is determined by qpts, so bands must match
    bands_tensor = torch.from_numpy(real/factor)  # [num_qpts, num_bands]
    
    # Build data dict
    # Note: kMVN requires original qpts and bands to maintain dimensional consistency
    # get_spectra(Hs, shifts, qpts) -> output shape: [num_qpts, 3*numb]
    # y shape must be: [num_qpts, 3*numb] for loss computation
    data_dict = {
        'id': mpid,
        'pos': torch.from_numpy(structure.positions.copy()),
        'lattice': torch.from_numpy(structure.cell.array.copy()).unsqueeze(0),
        'symbol': structure.symbols,
        'z': z,
        'x': x,
        'y': bands_tensor.unsqueeze(0),  # [1, num_qpts, 3*numb]
        'node_deg': node_deg,
        'edge_index': torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
        'edge_shift': torch.tensor(edge_shift, dtype=torch.float64),
        'edge_vec': torch.tensor(edge_vec, dtype=torch.float64),
        'edge_len': torch.tensor(edge_len, dtype=torch.float64),
        'qpts': qpts_tensor,  # [num_qpts, 3] - original format for kMVN
        'r_max': r_max,
        'numb': numb,
        'ucs': ucs
    }
    
    data = Data(**data_dict)
    return data


def generate_data_dict(data, r_max, descriptor='mass', factor=1000, **kwargs):
    """
    Generate a dictionary of band structure data for kMVN.
    
    Args:
        data (dict): Dictionary containing data to process.
        r_max (float): Cutoff radius for neighbor list.
        descriptor (str): Descriptor for node features.
        factor (int): Scaling factor for real values.
        
    Returns:
        dict: Data dictionary containing band structure information.
    """
    data_dict = dict()
    ids = data['id']
    structures = data['structure']
    qptss = data['qpts']
    reals = data['real_band']
    
    print(f"Processing {len(ids)} samples...")
    for id, structure, real, qpts in zip(ids, structures, reals, qptss):
        data_dict[id] = build_data_kmvn(id, structure, real, qpts, r_max, descriptor, factor, **kwargs)
    
    print(f"Generated {len(data_dict)} data samples")
    return data_dict
    
