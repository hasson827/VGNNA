# VGNNA: Virtual Node Graph Neural Network with Attention for Full Phonon Prediction

This repository provides the implementation of the Virtual Node Graph Neural Network with Attention (VGNNA) for full phonon prediction in materials science. VGNNA is designed to address the challenges in phonon prediction using graph neural networks.

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

- Virtual Node Graph Neural Network for Full Phonon Prediction [Paper](https://arxiv.org/abs/2301.02197) | [Github Code](https://github.com/RyotaroOKabe/phonon_prediction)
- Graph Transformer Networks for Accurate Band Structure Prediction: An End-to-End Approach [Paper](https://arxiv.org/abs/2411.16483) | [Github Code](https://github.com/qmatyanlab/Bandformer)
- **Dataset**: Petretto, Guido; Dwaraknath, Shyam; Miranda, Henrique P. C.; Winston, Donald; Giantomassi, Matteo; Rignanese, Gian-Marco; et al. (2018). High-throughput Density-Functional Perturbation Theory phonons for inorganic materials. figshare. Collection. [Dataset](https://doi.org/10.6084/m9.figshare.5649298)
- **Architecture:** Zhantao Chen, Nina Andrejevic, _et al._ "Virtual Node Graph Neural Network for Full Phonon Prediction." Adv. Sci. 8, 2004214 (2021). [Website](https://onlinelibrary.wiley.com/doi/10.1002/advs.202004214)
- **E(3)NN:** Mario Geiger, Tess Smidt, Alby M., Benjamin Kurt Miller, _et al._ Euclidean neural networks: e3nn (2020) v0.4.2. [Website](https://doi.org/10.5281/zenodo.5292912)
- **seekpath:** Y. Hinuma, G. Pizzi, Y. Kumagai, F. Oba, I. Tanaka, Band structure diagram paths based on crystallography, Comp. Mat. Sci. 128, 140 (2017) [Website](https://seekpath.readthedocs.io/en/latest/index.html)
- **Transformer Implementation**: We referred to [PyTorch's official implementation](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html) and [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html).
- **Open Source Repositories**:
  - [Xtal2DoS](https://github.com/JunwenBai/Xtal2DoS)
  - [e3nn](https://github.com/e3nn/e3nn)
  - [nanoGPT](https://github.com/karpathy/nanoGPT)

## License

MIT License - See LICENSE file for details
