"""
Model comparison script for k-MVN with and without attention.

This script compares the performance of baseline k-MVN model and
k-MVN with attention mechanism on the test set.
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
from utils.plot import generate_dataframe, compare_models

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
    Find the latest model file matching pattern.
    
    Args:
        model_path_pattern (str): Glob pattern for model files.
        
    Returns:
        str: Path to the latest model file.
    """
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


def load_model(config, model_path, use_attention=False):
    """
    Load trained model from checkpoint.
    
    Args:
        config (dict): Configuration dictionary.
        model_path (str): Path to model checkpoint.
        use_attention (bool): Whether to use attention mechanism.
        
    Returns:
        GraphNetwork_kMVN: Loaded model.
    """
    print(f"\nLoading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    
    # Initialize model
    model = GraphNetwork_kMVN(
        mul=config['mul'],
        irreps_out=config['irreps_out'],
        lmax=config['lmax'],
        nlayers=config['nlayers'],
        number_of_basis=config['number_of_basis'],
        radial_layers=config['radial_layers'],
        radial_neurons=config['radial_neurons'],
        node_dim=config['node_dim'],
        node_embed_dim=config['node_embed_dim'],
        input_dim=config['input_dim'],
        input_embed_dim=config['input_embed_dim'],
        use_attention=use_attention,
        attn_heads=config['attn_heads'],
        attn_dropout=config['attn_dropout']
    )
    
    # Load state dict
    state_dict = checkpoint['state']
    
    model.load_state_dict(state_dict)
    model.to(config['device'])
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


def load_test_data(config):
    """
    Load test dataset.
    
    Args:
        config (dict): Configuration dictionary.
        
    Returns:
        torch.utils.data.Subset: Test dataset.
    """
    data_dir = config['data_dir']
    raw_dir = config['raw_dir']
    data_file = config['data_file']
    r_max = config['r_max']
    descriptor = config['descriptor']
    factor = config['factor']
    tr_ratio = config['tr_ratio']
    seed = config['seed']
    
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


    num = len(data_dict)
    te_num = int(num * 0.1)
    _, idx_te = train_test_split(range(num), test_size=te_num, random_state=seed)
    data_set = torch.utils.data.Subset(list(data_dict.values()), range(len(data_dict)))
    test_set = torch.utils.data.Subset(data_set, idx_te)
    print(f"Created test split: {len(test_set)} samples")
    
    return test_set


def generate_predictions(model, dataset, config):
    """
    Generate predictions on dataset.
    
    Args:
        model: Trained model.
        dataset: Dataset to sample from.
        config (dict): Configuration dictionary.
        
    Returns:
        pandas.DataFrame: DataFrame with predictions.
    """
    print("\nGenerating predictions...")
    
    # Create data loader
    from torch_geometric.loader import DataLoader
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize loss function
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


def main():
    """Main comparison function."""
    print("=" * 60)
    print("Model Comparison: k-MVN vs k-MVN + Attention")
    print("=" * 60)
    
    # Setup configuration
    config = setup_config(globals())
    
    # Use default config if not specified
    if '--config' not in sys.argv:
        default_config_path = os.path.join(os.path.dirname(__file__), '../configs/sample.yaml')
        config = parse_config(default_config_path)
        config['config'] = default_config_path
    
    # Print configuration
    print_config(config, process="Comparison")
    
    # Setup environment
    setup_environment(config)
    
    # Load test data
    test_set = load_test_data(config)
    
    # Load baseline model (without attention)
    baseline_model_path = config.get('baseline_model_path', '../ckpt/kMVN_best.torch')
    baseline_model = load_model(config, baseline_model_path, use_attention=False)
    
    # Generate predictions for baseline model
    df_baseline = generate_predictions(baseline_model, test_set, config)
    
    # Load attention model (with attention)
    attn_model_path = config.get('attention_model_path', '../ckpt/kMVN_Attn_best.torch')
    attn_model = load_model(config, attn_model_path, use_attention=True)
    
    # Generate predictions for attention model
    df_attn = generate_predictions(attn_model, test_set, config)
    
    # Create output directory
    out_dir = config.get('out_dir', '../out')
    os.makedirs(out_dir, exist_ok=True)
    
    # Generate comparison plot
    print("\nGenerating comparison plot...")
    compare_models(
        df1=df_baseline,
        df2=df_attn,
        header=os.path.join(out_dir, 'comparison'),
        color1='#F8961E',  # Orange for baseline
        color2='#43AA8B',  # Teal for attention
        labels=('k-MVN (baseline)', 'k-MVN + Attention'),
        size=5,
        lw=3,
        r2=True  # Calculate and display R^2 scores
    )
    
    print(f"\nComparison plot saved to: {out_dir}/comparison_model_comparison.{config['save_extension']}")
    
    # Save comparison results to CSV
    comparison_csv = os.path.join(out_dir, 'comparison.csv')
    with open(comparison_csv, 'w') as f:
        f.write("model,average loss,num samples\n")
        f.write(f"k-MVN (baseline),{df_baseline['loss'].mean():.4f},{len(df_baseline)}\n")
        f.write(f"k-MVN + Attention,{df_attn['loss'].mean():.4f},{len(df_attn)}\n")
    print(f"Comparison results saved to: {comparison_csv}")
    
    print("\n" + "=" * 60)
    print("Model comparison completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()