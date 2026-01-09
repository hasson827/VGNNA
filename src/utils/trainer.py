"""
Training utilities for kMVN
"""

import os
import torch
import time
import math
from models.kmvn import get_spectra
from torch_geometric.loader import DataLoader

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def evaluate(model, dataloader, loss_fn, device):
    """
    Evaluate model on a dataloader.
    
    Args:
        model: The kMVN model.
        dataloader: PyTorch Geometric DataLoader.
        loss_fn: Loss function.
        device: Device to run evaluation on.
        
    Returns:
        float: Average loss over all batches.
    """
    model.eval()
    loss_cumulative = 0.
    
    with torch.no_grad():
        for d in dataloader:
            d.to(device)
            
            Hs, shifts = model(d)
            output = get_spectra(Hs, shifts, d.qpts)
            loss = loss_fn(output, d.y).cpu()
            loss_cumulative += loss.detach().item()
    
    return loss_cumulative / len(dataloader)


def loglinspace(rate, step, end=None):
    """Generate checkpoints using log-linear spacing."""
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step * (1 - math.exp(-t * rate / step)))


def train(model,
          opt,
          tr_set,
          tr_nums,
          te_set,
          loss_fn,
          run_name,
          max_iter,
          scheduler,
          device,
          batch_size,
          k_fold,
          factor=1000,
          conf_dict=None,
          out_dir=None):
    """
    Train kMVN model with cross-validation.
    
    Args:
        model: kMVN model to train.
        opt: Optimizer.
        tr_set: Training dataset.
        tr_nums: List of training split sizes for k-fold CV.
        te_set: Test dataset.
        loss_fn: Loss function.
        run_name: Name for saving checkpoints.
        max_iter: Maximum number of iterations.
        scheduler: Learning rate scheduler.
        device: Device to train on.
        batch_size: Batch size (should be 1 for kMVN).
        k_fold: Number of folds for k-fold cross-validation.
        factor: Scaling factor for band values.
        conf_dict: Configuration dictionary for logging.
        out_dir: Directory to save outputs.
    """
    model.to(device)
    checkpoint_generator = loglinspace(0.3, 5)
    checkpoint = next(checkpoint_generator)
    start_time = time.time()

    record_lines = []
    best_valid_loss = float('inf')  # Initialize best validation loss to infinity

    # Try to load existing model
    try:
        print(f'Use model.load_state_dict to load existing model: {run_name}.torch')
        model.load_state_dict(torch.load(f'{run_name}.torch')['state'])
        results = torch.load(f'{run_name}.torch')
        history = results['history']
        s0 = history[-1]['step'] + 1
        
        # Find best validation loss from history
        for entry in history:
            if entry['valid']['loss'] < best_valid_loss:
                best_valid_loss = entry['valid']['loss']
    except:
        print('No existing model found, starting from scratch')
        results = {}
        history = []
        s0 = 0

    # Create k-fold splits
    tr_sets = torch.utils.data.random_split(tr_set, tr_nums)
    te_loader = DataLoader(te_set, batch_size=batch_size)
    
    for step in range(max_iter):
        k = step % k_fold
        tr_loader = DataLoader(
            torch.utils.data.ConcatDataset(tr_sets[:k] + tr_sets[k+1:]), 
            batch_size=batch_size, 
            shuffle=True
        )
        va_loader = DataLoader(tr_sets[k], batch_size=batch_size)
        
        model.train()
        N = len(tr_loader)
        
        for i, d in enumerate(tr_loader):
            start = time.time()
            d.to(device)
            
            Hs, shifts = model(d)
            output = get_spectra(Hs, shifts, d.qpts)

            
            loss = loss_fn(output, d.y).cpu()
            opt.zero_grad()
            loss.backward()
            opt.step()

            print(f'num {i+1:4d}/{N}, loss = {loss}, train time = {time.time() - start}', end='\r')

        end_time = time.time()
        wall = end_time - start_time
        print(wall)
        
        if step == checkpoint:
            checkpoint = next(checkpoint_generator)
            assert checkpoint > step

            # Evaluate on validation and training sets
            valid_avg_loss = evaluate(model, va_loader, loss_fn, device)
            train_avg_loss = evaluate(model, tr_loader, loss_fn, device)

            history.append({
                'step': s0 + step,
                'wall': wall,
                'batch': {'loss': loss.item()},
                'valid': {'loss': valid_avg_loss},
                'train': {'loss': train_avg_loss},
            })

            results = {
                'history': history,
                'state': model.state_dict()
            }
            
            if conf_dict is not None:
                results['conf_dict'] = conf_dict

            print(f"Iteration {step+1:4d}   " +
                  f"train loss = {train_avg_loss:8.20f}   " +
                  f"valid loss = {valid_avg_loss:8.20f}   " +
                  f"elapsed time = {time.strftime('%H:%M:%S', time.gmtime(wall))}")

            # Save checkpoint
            os.makedirs(out_dir, exist_ok=True)
            save_name = os.path.join(out_dir, run_name)
            with open(f'{save_name}.torch', 'wb') as f:
                torch.save(results, f)

            # Save best model if validation loss improved
            if valid_avg_loss < best_valid_loss:
                best_valid_loss = valid_avg_loss
                print(f"  -> New best model saved! Valid loss = {best_valid_loss:.20f}")
                with open(f'{save_name}_best.torch', 'wb') as f:
                    torch.save(results, f)

            record_line = '%d\t%.20f\t%.20f' % (step, train_avg_loss, valid_avg_loss)
            record_lines.append(record_line)

            if scheduler is not None:
                scheduler.step()

    # Save final results
    save_name = os.path.join(out_dir, run_name)
    text_file = open(f"{save_name}.txt", "w")
    for line in record_lines:
        text_file.write(line + "\n")
    text_file.close()


def load_model(model_class, model_file, device):
    """
    Load a pre-trained model, its weights, and hyperparameters.
    
    Args:
        model_class: The class of the model to be instantiated.
        model_file: Path to the saved model file.
        device: The device to load the model on.
        
    Returns:
        model: The model with loaded weights.
        conf_dict: A dictionary of hyperparameters (if available).
        history: A list of training history (if available).
        int: The starting step (for resuming training).
    """
    if os.path.exists(model_file):
        print(f"Loading model from: {model_file}")
        checkpoint = torch.load(model_file)
        
        # Extract hyperparameters and initialize model
        conf_dict = checkpoint.get('conf_dict', checkpoint)
        model = model_class(**conf_dict)
        model.load_state_dict(checkpoint['state'])
        model.to(device)

        # Extract history and step number
        history = checkpoint.get('history', [])
        s0 = history[-1]['step'] + 1 if history else 0

        return model, conf_dict, history, s0
    else:
        raise FileNotFoundError(f"No model found at {model_file}")