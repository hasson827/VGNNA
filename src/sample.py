"""
Sampling and plotting script for kMVN (k-Multi-Virtual Node) Graph Neural Network
Uses functional programming style with `if __name__ == "__main__"` pattern.
"""

import os
import sys
import torch
import numpy as np
import glob
from sklearn.model_selection import train_test_split

from utils.config import setup_config, load_yaml_config, print_config, setup_environment
from utils.load import load_band_structure_data
from utils.data import generate_data_dict
from models.kmvn import GraphNetwork_kMVN
from utils.loss import BandLoss
from utils.plot import generate_dataframe, plot_bands

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_config(config_path):
    """
    Parse configuration from YAML file.
    
    Args:
        config_path (str): Path to YAML config file.
        
    Returns:
        dict: Configuration dictionary.
    """
    config = load_yaml_config(config_path)
    
    # Add device setting
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Update save extension
    if 'save_extension' in config:
        from utils import plot
        plot.save_extension = config['save_extension']
    
    return config


def find_model_file(model_path_pattern):
    """
    Find the latest model file matching the pattern.
    
    Args:
        model_path_pattern (str): Glob pattern for model files.
        
    Returns:
        str: Path to the latest model file.
    """
    # Expand pattern if it contains *
    if '*' in model_path_pattern:
        model_files = glob.glob(model_path_pattern)
        if not model_files:
            raise FileNotFoundError(f"No model files found matching: {model_path_pattern}")
        
        # Sort by modification time and get the latest
        latest_model = max(model_files, key=os.path.getmtime)
        print(f"Found latest model: {latest_model}")
        return latest_model
    else:
        # Direct path provided
        if not os.path.exists(model_path_pattern):
            raise FileNotFoundError(f"Model file not found: {model_path_pattern}")
        print(f"Using model: {model_path_pattern}")
        return model_path_pattern


def load_model(config):
    """
    Load trained model from checkpoint.
    
    Args:
        config (dict): Configuration dictionary.
        
    Returns:
        tuple: (model, checkpoint)
    """
    print("\nLoading model...")
    model_path = find_model_file(config['model_path'])
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Extract model configuration
    if 'conf_dict' in checkpoint:
        model_conf = checkpoint['conf_dict']
    elif 'model_args' in checkpoint:
        # Handle Bandformer-style checkpoints
        model_conf = checkpoint['model_args']
    else:
        raise ValueError("Checkpoint does not contain model configuration")
    
    # Initialize model
    model = GraphNetwork_kMVN(
        mul=model_conf.get('mul', config['mul']),
        irreps_out=model_conf.get('irreps_out', config['irreps_out']),
        lmax=model_conf.get('lmax', config['lmax']),
        nlayers=model_conf.get('nlayers', config['nlayers']),
        number_of_basis=model_conf.get('number_of_basis', config['number_of_basis']),
        radial_layers=model_conf.get('radial_layers', config['radial_layers']),
        radial_neurons=model_conf.get('radial_neurons', config['radial_neurons']),
        node_dim=model_conf.get('node_dim', config['node_dim']),
        node_embed_dim=model_conf.get('node_embed_dim', config['node_embed_dim']),
        input_dim=model_conf.get('input_dim', config['input_dim']),
        input_embed_dim=model_conf.get('input_embed_dim', config['input_embed_dim'])
    )
    
    # Load state dict
    state_dict = checkpoint['state']
    
    # Fix state dict keys (remove _orig_mod prefix if present)
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(config['device'])
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model, checkpoint


def load_data(config):
    """
    Load and prepare data for sampling.
    
    Args:
        config (dict): Configuration dictionary.
        
    Returns:
        torch.utils.data.Subset: Dataset to sample from.
    """
    data_dir = config['data_dir']
    raw_dir = config['raw_dir']
    data_file = config['data_file']
    r_max = config['r_max']
    descriptor = config['descriptor']
    factor = config['factor']
    tr_ratio = config['tr_ratio']
    seed = config['seed']
    split = config['split']
    
    # Load data
    print("\nLoading data...")
    data = load_band_structure_data(data_dir, raw_dir, data_file)
    print(f"Loaded {len(data)} samples")
    
    # Generate data dict
    data_dict = generate_data_dict(
        data=data,
        r_max=r_max,
        descriptor=descriptor,
        factor=factor,
    )
    
    # Load train/test split if exists
    # Check if split files exist
    possible_splits = ['train', 'val', 'test']
    dataset = None
    
    # Try to load from saved split files
    for split_name in possible_splits:
        idx_file = os.path.join(data_dir, f'idx_*_{split_name}.txt')
        if os.path.exists(idx_file):
            # Find the latest index file
            idx_files = glob.glob(os.path.join(data_dir, f'idx_*_{split_name}.txt'))
            latest_idx_file = max(idx_files, key=os.path.getmtime)
            
            # Load indices
            with open(latest_idx_file, 'r') as f:
                indices = [int(line.strip()) for line in f]
            
            # Create dataset
            data_set = torch.utils.data.Subset(list(data_dict.values()), range(len(data_dict)))
            dataset = torch.utils.data.Subset(data_set, indices)
            print(f"Loaded {split_name} split: {len(dataset)} samples")
            break
    
    # If no split file found, create split based on split parameter
    if dataset is None:
        print(f"\nNo saved split found, creating split...")
        num = len(data_dict)
        
        if split == 'test':
            # Use test split (last 10%)
            te_num = int(num * 0.1)
            _, idx_te = train_test_split(range(num), test_size=te_num, random_state=seed)
            data_set = torch.utils.data.Subset(list(data_dict.values()), range(len(data_dict)))
            dataset = torch.utils.data.Subset(data_set, idx_te)
        elif split == 'train':  # train
            tr_num = int(num * 0.9)
            idx_tr, _ = train_test_split(range(num), test_size=num-tr_num, random_state=seed)
            data_set = torch.utils.data.Subset(list(data_dict.values()), range(len(data_dict)))
            dataset = torch.utils.data.Subset(data_set, idx_tr)
        
        print(f"Created {split} split: {len(dataset)} samples")
    
    # Limit number of samples if specified
    if config.get('num_samples') is not None and config['num_samples'] < len(dataset):
        np.random.seed(config['seed'])
        indices = np.random.choice(len(dataset), config['num_samples'], replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Limited to {config['num_samples']} samples")
    
    return dataset


def sample_data(model, dataset, config):
    """
    Sample predictions from dataset.
    
    Args:
        model: Trained model.
        dataset: Dataset to sample from.
        config (dict): Configuration dictionary.
        
    Returns:
        pandas.DataFrame: DataFrame with predictions.
    """
    print("\nSampling predictions...")
    
    # Create data loader
    from torch_geometric.loader import DataLoader
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize loss function for error calculation
    loss_fn = BandLoss()
    
    # Generate dataframe
    df = generate_dataframe(
        model=model,
        dataloader=loader,
        loss_fn=loss_fn,
        device=config['device'],
        factor=config['factor']
    )
    
    print(f"Generated dataframe with {len(df)} samples")
    print(f"Average loss: {df['loss'].mean():.4f}")
    
    return df


def plot_bands_distribution(df, config, run_name):
    """
    Plot phonon bands with loss distribution.
    
    Args:
        df (pandas.DataFrame): DataFrame with predictions.
        config (dict): Configuration dictionary.
        run_name (str): Name for saving plots.
    """
    print("\nPlotting bands...")
    
    # Get plot settings from config
    n = config.get('n', 5)
    m = config.get('m', 2)
    lwidth = config.get('lwidth', 0.5)
    windowsize = tuple(config.get('windowsize', [3, 2]))
    seed = config['seed']
    
    # Create output directory
    out_dir = config['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    # Generate save path
    header = os.path.join(out_dir, f'{run_name}_bands')
    
    # Plot bands
    plot_bands(
        df_in=df,
        header=header,
        title=config['split'],
        n=n,
        m=m,
        lwidth=lwidth,
        windowsize=windowsize,
        seed=seed
    )
    
    print(f"Saved plots to: {header}_*.{config['save_extension']}")


def main():
    """Main sampling function."""
    # Default config path
    default_config_path = os.path.join(os.path.dirname(__file__), '../configs/sample.yaml')
    
    # Setup configuration system
    config = setup_config(globals())
    
    # Use default config if not specified
    if '--config' not in sys.argv:
        config = parse_config(default_config_path)
        config['config'] = default_config_path
    
    # Print configuration
    print_config(config, process = "Sampling")
    
    # Setup environment
    setup_environment(config)
    
    # Load model
    model, checkpoint = load_model(config)
    
    # Load data
    dataset = load_data(config)
    
    # Sample data
    df = sample_data(model, dataset, config)
    
    # Generate run name from checkpoint
    run_name = checkpoint.get('run_name', 'sample')
    
    # Plot results
    plot_bands_distribution(df, config, run_name)
    
    print("\n" + "=" * 60)
    print("Sampling and plotting completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()