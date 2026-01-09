"""
Q-points sampling functionality using seekpath for phonon band path generation.

This module provides functions to compute high-symmetry k-point paths
and generate q-points for phonon band structure visualization.
"""

import numpy as np
import seekpath

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


Greek_letters = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 
                'Eta', 'Theta', 'Iota', 'Kappa', 'Lambda', 'Mu', 'Nu', 
                'Xi', 'Omicron', 'Pi', 'Rho', 'Sigma', 'Tau', 'Upsilon', 
                'Phi', 'Chi', 'Psi', 'Omega']


def get_structure(astruct):
    """
    Extracts structure data (cell, positions, atomic numbers) from an ASE atoms object.
    
    Args:
        astruct (ase.Atoms): ASE crystal structure.
    
    Returns:
        tuple: Cell, positions, and atomic numbers as lists.
    """
    return astruct.cell.tolist(), astruct.get_positions().tolist(), astruct.numbers.tolist()


def symbol_latex(symbol):
    """
    Converts a chemical symbol to LaTeX format.
    
    Args:
        symbol (str): Chemical symbol.
    
    Returns:
        str: LaTeX formatted chemical symbol.
    """
    return f'\\{symbol.capitalize()}' if symbol.split('_')[0].capitalize() in Greek_letters else symbol


def get_qpts(astruct, res=0.05, threshold=1e-07, symprec=1e-05, angle_tolerance=-1.0):
    """
    Computes high-symmetry k-point path and q-points for a given structure.
    
    This function uses seekpath to calculate the high-symmetry path through
    the Brillouin zone and generates discrete q-points along that path for
    phonon band structure plotting.
    
    Args:
        astruct (ase.Atoms): ASE crystal structure.
        res (float, optional): Resolution of q-points (default: 0.05).
        threshold (float, optional): Symmetry threshold (default: 1e-07).
        symprec (float, optional): Symmetry precision (default: 1e-05).
        angle_tolerance (float, optional): Angle tolerance (default: -1.0).
    
    Returns:
        dict: Dictionary containing:
            - 'pcoords': High-symmetry point coordinates
            - 'path': Path segments
            - 'path_set': Path point sets
            - 'point_list': List of symmetry points
            - 'sym_points': Symmetry points
            - 'break_from': Break indices
            - 'qpts': Generated q-points array [num_qpts, 3]
            - 'dist_list': Distances between points
            - 'qticks': Labels for q-points
    """
    # Get structure data (cell, positions, atomic numbers) from ASE object
    struct = get_structure(astruct)
    
    # Use Seekpath to calculate high-symmetry path and coordinates
    getpath = seekpath.getpaths.get_path(
        struct, 
        with_time_reversal=True, 
        recipe='hpkot', 
        threshold=threshold, 
        symprec=symprec, 
        angle_tolerance=angle_tolerance
    )
    
    # Convert point coordinates and path points to LaTeX symbols
    pcoords = {symbol_latex(k): v for k, v in getpath['point_coords'].items()}
    path = [(symbol_latex(start), symbol_latex(end)) for start, end in getpath['path']]
    
    # Initialize empty lists to store symmetry paths, point coordinates
    path_set, point_list, sym_points, break_from = [], [], [], []
    p_end0 = None  # Track end point from previous loop iteration

    # Loop through each path segment and construct path sets
    for i, (p_start, p_end) in enumerate(path):     
        # Check if start point of current segment is same as previous end point
        if p_end0 == p_start: 
            path_set[-1].append(pcoords[p_end])  # Extend previous path set
            sym_points.append(p_end)  # Add endpoint to symmetry points
            point_list.append(pcoords[p_end])  # Add endpoint coordinates
        else:
            # New path segment starts, create new path set
            path_set.append([pcoords[p_start], pcoords[p_end]])  
            sym_points.extend([p_start, p_end])  # Add both start and end
            point_list.extend([pcoords[p_start], pcoords[p_end]])
            break_from.append(i)  # Mark break point
        p_end0 = p_end  # Update previous endpoint
    
    # Remove first element from break_from
    break_from.remove(0)

    # Generate q-points between symmetry points
    qpts, dist_list = [], []
    for i in range(len(point_list) - 1):
        start, end = np.array(point_list[i]), np.array(point_list[i + 1])
        dist = np.linalg.norm(end - start)  # Distance between points
        dist_list.append(dist)
        dig = int(dist // res)  # Number of q-points based on resolution

        if i not in break_from:  # Not a break point, distribute q-points evenly
            for j in range(dig):
                kvec = ((dig - j) * start + j * end) / dig
                qpts.append(kvec.tolist())
        else:
            qpts.append(start.tolist())  # Start point if break
    
    qpts.append(end.tolist())  # Add final endpoint
    qpts = np.array(qpts)

    # Generate labels for q-points based on proximity to symmetry points
    qticks = []
    for qpt in qpts:
        high_symmetry = 0
        for label, vec in pcoords.items():
            if np.allclose(qpt, vec, atol=1e-4):  # Check if close to high-symmetry point
                qticks.append(label)
                high_symmetry = 1
                break
        if high_symmetry == 0:
            qticks.append("")  # Blank label if not high-symmetry point

    return {
        'pcoords': pcoords,
        'path': path,
        'path_set': path_set,
        'point_list': point_list,
        'sym_points': sym_points,
        'break_from': break_from,
        'qpts': qpts,
        'dist_list': dist_list,
        'qticks': qticks
    }