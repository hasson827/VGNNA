"""
Training script for kMVN (k-Multi-Virtual Node) Graph Neural Network
Uses functional programming style with `if __name__ == "__main__"` pattern.
"""

import os
import sys
import torch
import numpy as np
from sklearn.model_selection import train_test_split

from utils.config import setup_config, load_yaml_config, print_config, setup_environment
from utils.load import load_band_structure_data
from utils.data import generate_data_dict
from models.kmvn import GraphNetwork_kMVN
from utils.loss import BandLoss
from utils.trainer import train

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
    
    return config


def load_data(config):
    """
    Load and prepare data.
    
    Args:
        config (dict): Configuration dictionary.
        
    Returns:
        tuple: (tr_set, te_set, tr_nums)
    """
    data_dir = config['data_dir']
    raw_dir = config['raw_dir']
    data_file = config['data_file']
    r_max = config['r_max']
    descriptor = config['descriptor']
    factor = config['factor']
    tr_ratio = config['tr_ratio']
    k_fold = config['k_fold']
    seed = config['seed']
    run_name = config['run_name']

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

    # Train/test split
    print("\nSplitting data...")
    num = len(data_dict)
    tr_nums = [int((num * tr_ratio) // k_fold)] * k_fold
    te_num = num - sum(tr_nums)
    idx_tr, idx_te = train_test_split(range(num), test_size=te_num, random_state=seed)

    # Create datasets
    data_set = torch.utils.data.Subset(list(data_dict.values()), range(len(data_dict)))
    tr_set = torch.utils.data.Subset(data_set, idx_tr)
    te_set = torch.utils.data.Subset(data_set, idx_te)

    print(f"Training samples: {len(tr_set)}")
    print(f"Test samples: {len(te_set)}")

    return tr_set, te_set, tr_nums, run_name


def initialize_model(config):
    """
    Initialize kMVN model.
    
    Args:
        config (dict): Configuration dictionary.
        
    Returns:
        model: Initialized GraphNetwork_kMVN model.
    """
    print("\nInitializing model...")
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
        input_embed_dim=config['input_embed_dim']
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model


def initialize_optimizer_and_scheduler(model, config):
    """
    Initialize optimizer and learning rate scheduler.
    
    Args:
        model: The model to optimize.
        config (dict): Configuration dictionary.
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        opt, 
        gamma=config['schedule_gamma']
    )
    return opt, scheduler


def run_training(config, model, opt, scheduler, tr_set, te_set, tr_nums, run_name):
    """
    Run training process.
    
    Args:
        config (dict): Configuration dictionary.
        model: The model to train.
        opt: The optimizer.
        scheduler: The learning rate scheduler.
        tr_set: Training dataset.
        te_set: Test dataset.
        tr_nums: List of training split sizes.
        run_name: Name for saving checkpoints.
    """
    # Initialize loss function
    loss_fn = BandLoss()

    # Print training info
    print("\nStarting training...")
    print(f"Run name: {run_name}")
    print(f"Output directory: {config['out_dir']}")
    print(f"Device: {config['device']}")
    print("=" * 60)

    # Run training
    train(
        model=model,
        opt=opt,
        tr_set=tr_set,
        tr_nums=tr_nums,
        te_set=te_set,
        loss_fn=loss_fn,
        run_name=run_name,
        max_iter=config['max_iter'],
        scheduler=scheduler,
        device=config['device'],
        batch_size=config['batch_size'],
        k_fold=config['k_fold'],
        factor=config['factor'],
        conf_dict=config,
        out_dir=config['out_dir']
    )

    print("\nTraining completed!")
    print(f"Model saved as: {run_name}.torch")
    print(f"Results saved to: {config['out_dir']}")


def main():
    """Main training function."""
    # Default config path
    default_config_path = os.path.join(os.path.dirname(__file__), '../configs/train.yaml')
    
    # Setup configuration system
    config = setup_config(globals())
    
    # Use default config if not specified
    if '--config' not in sys.argv:
        config = parse_config(default_config_path)
        config['config'] = default_config_path
    
    # Print configuration
    print_config(config, process="Training")
    
    # Setup environment
    setup_environment(config)
    
    # Load data
    tr_set, te_set, tr_nums, run_name = load_data(config)
    
    # Initialize model
    model = initialize_model(config)
    
    # Initialize optimizer and scheduler
    opt, scheduler = initialize_optimizer_and_scheduler(model, config)
    
    # Run training
    run_training(config, model, opt, scheduler, tr_set, te_set, tr_nums, run_name)


if __name__ == "__main__":
    main()