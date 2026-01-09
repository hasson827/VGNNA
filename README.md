# VGNNA: Virtual Node Graph Neural Network with Attention for Full Phonon Prediction

## Environment Setup

```bash
# Create virtual environment
conda create -n phonon python=3.9 -y
conda activate phonon

# Install dependencies
pip install uv
uv pip install -r requirements.txt -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
uv pip uninstall numpy
uv pip install "numpy<2.0.0"
```

Replace `cu118` with your CUDA version (e.g., `cu117`, `cpu`).

## Configuration

Edit `configs/train.yaml` to customize training parameters.

Edit `configs/sample.yaml` to customize sampling parameters.

## Training

The Training process takes about 11.5h on a single 4090 GPU for 200 Iterations.

### Basic Usage

```bash
cd VGNNA/src
python train.py
```

### Override Parameters via Command Line

Any parameter in the YAML config can be overridden:

```bash
python train.py --lr 0.01 --max_iter 100
```

## Sampling

### Basic Usage

```bash
cd VGNNA/src
python sample.py
```

### Override Parameters via Command Line

Any parameter in the YAML config can be overridden:

```bash
python sample.py --out_dir ../viz --save_extension png
```

## Reference

This code is based on:

- [Virtual Node Graph Neural Network for Full Phonon Prediction](https://arxiv.org/abs/2301.02197) | [Github Code](https://github.com/RyotaroOKabe/phonon_prediction)
- [Graph Transformer Networks for Accurate Band Structure Prediction: An End-to-End Approach](https://arxiv.org/abs/2411.16483) | [Github Code](https://github.com/qmatyanlab/Bandformer)

## License

MIT License - See LICENSE file for details
