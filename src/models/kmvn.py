"""
kMVN (k-Multi-Virtual Node) Graph Neural Network for Phonon Prediction
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from e3nn.o3 import Irrep, Irreps, spherical_harmonics, TensorProduct, FullyConnectedTensorProduct
from e3nn.nn import Gate, FullyConnectedNet
from e3nn.math import soft_one_hot_linspace


class SelfAttentionLayer(nn.Module):
    """
    Self-attention layer for graph nodes.
    Operates on scalar features to preserve equivariance.
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        # Q, K, V projections
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x, edge_index, edge_len):
        """
        Args:
            x: Node features [num_nodes, d_model]
            edge_index: Edge indices [2, num_edges]
            edge_len: Edge lengths [num_edges]
        Returns:
            Enhanced node features
        """
        # Self-attention with residual
        x = x + self._self_attention_block(x, edge_index, edge_len)
        
        # Feed-forward with residual
        x = x + self._ffn_block(x)
        
        return x
    
    def _self_attention_block(self, x, edge_index, edge_len):
        x_norm = self.norm1(x)
        
        # Compute Q, K, V
        qkv = self.qkv(x_norm).chunk(3, dim=-1)
        q, k, v = [t.view(-1, self.num_heads, self.d_head) for t in qkv]
        
        # Attention computation with graph structure (edge-level)
        src, dst = edge_index
        attn_scores = (q[src] * k[dst]).sum(-1) / (self.d_head ** 0.5)
        
        # Edge-level attention weights (no scatter here)
        attn_weights = F.softmax(attn_scores, dim=0)
        attn_weights = self.dropout(attn_weights)
        
        # Aggregate messages from edges to nodes
        messages = v[dst] * attn_weights.unsqueeze(-1)
        out = scatter(messages, dst, dim=0, dim_size=x.size(0), reduce='sum')
        
        out = out.view(-1, self.d_model)
        return self.out_proj(out)
    
    def _ffn_block(self, x):
        x_norm = self.norm2(x)
        return self.dropout(self.ffn(x_norm))


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    """Check if a tensor product path exists."""
    irreps_in1 = Irreps(irreps_in1).simplify()
    irreps_in2 = Irreps(irreps_in2).simplify()
    ir_out = Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


class CustomCompose(nn.Module):
    """Custom module to sequentially apply two modules, storing intermediate outputs."""
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        self.first_out = x.clone()
        x = self.second(x)
        self.second_out = x.clone()
        return x


class GraphConvolution(nn.Module):
    """Graph convolution layer that processes node and edge features."""
    def __init__(self,
                 irreps_in,
                 irreps_node_attr,
                 irreps_edge_attr,
                 irreps_out,
                 number_of_basis,
                 radial_layers,
                 radial_neurons):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_node_attr = Irreps(irreps_node_attr)
        self.irreps_edge_attr = Irreps(irreps_edge_attr)
        self.irreps_out = Irreps(irreps_out)

        self.linear_input = FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_in)
        self.linear_mask = FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_out)
        
        irreps_mid = []
        instructions = []
        for i, (mul, irrep_in) in enumerate(self.irreps_in):
            for j, (_, irrep_edge_attr) in enumerate(self.irreps_edge_attr):
                for irrep_mid in irrep_in * irrep_edge_attr:
                    if irrep_mid in self.irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, irrep_mid))
                        instructions.append((i, j, k, 'uvu', True))
        irreps_mid = Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [(i_1, i_2, p[i_out], mode, train) for (i_1, i_2, i_out, mode, train) in instructions]

        self.tensor_edge = TensorProduct(self.irreps_in,
                                         self.irreps_edge_attr,
                                         irreps_mid,
                                         instructions,
                                         internal_weights=False,
                                         shared_weights=False)
        
        self.edge2weight = FullyConnectedNet([number_of_basis] + radial_layers * [radial_neurons] + [self.tensor_edge.weight_numel], F.silu)
        self.linear_output = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, self.irreps_out)

    def forward(self,
                node_input,
                node_attr,
                node_deg,
                edge_src,
                edge_dst,
                edge_attr,
                edge_length_embedded,
                numb, n):

        node_input_features = self.linear_input(node_input, node_attr)
        node_features = torch.div(node_input_features, torch.pow(node_deg, 0.5))

        node_mask = self.linear_mask(node_input, node_attr)

        edge_weight = self.edge2weight(edge_length_embedded)
        edge_features = self.tensor_edge(node_features[edge_src], edge_attr, edge_weight)

        node_features = scatter(edge_features, edge_dst, dim = 0, dim_size = node_features.shape[0])
        node_features = torch.div(node_features, torch.pow(node_deg, 0.5))

        node_output_features = self.linear_output(node_features, node_attr)

        node_output = node_output_features

        c_s, c_x = math.sin(math.pi / 8), math.cos(math.pi / 8)
        mask = self.linear_mask.output_mask
        c_x = (1 - mask) + c_x * mask
        return c_s * node_mask + c_x * node_output


class GraphHamiltonianConvolution(GraphConvolution):
    """Graph Hamiltonian convolution layer with matrix multiplication for complex output."""
    def __init__(self, 
                 irreps_in, 
                 irreps_node_attr, 
                 irreps_edge_attr, 
                 irreps_out, 
                 number_of_basis, 
                 radial_layers, 
                 radial_neurons):
        super().__init__(irreps_in, 
                         irreps_node_attr, 
                         irreps_edge_attr, 
                         irreps_out, 
                         number_of_basis, 
                         radial_layers, 
                         radial_neurons)
        tr = 3 ** -0.5
        tw = 2 ** -0.5
        self.irrep2tens = torch.tensor([[    tr,  0,   0,   0,      tr,  0,  0,   0,     tr],
                                        [     0,  0,   0,   0,       0, tw,  0, -tw,      0],
                                        [     0,  0, -tw,   0,       0,  0, tw,   0,      0],
                                        [     0, tw,   0, -tw,       0,  0,  0,   0,      0],
                                        [     0,  0,  tw,   0,       0,  0, tw,   0,      0],
                                        [     0, tw,   0,  tw,       0,  0,  0,   0,      0],
                                        [-tw*tr,  0,   0,   0, 2*tw*tr,  0,  0,   0, -tw*tr],
                                        [     0,  0,   0,   0,       0, tw,  0,  tw,      0],
                                        [   -tw,  0,   0,   0,       0,  0,  0,   0,     tw]], dtype = torch.complex128)

    @staticmethod
    def glue(blocks, numb, n):
        """Glue tensor blocks into final matrix."""
        return torch.cat(torch.cat(list(blocks), dim = 1).t().chunk(n*numb), dim = 1).t().reshape((n, 3*numb, 3*numb))

    def forward(self,
                node_input,
                node_attr,
                node_deg,
                edge_src,
                edge_dst,
                edge_attr,
                edge_length_embedded,
                numb, n):
        node_output = super().forward(node_input,
                                        node_attr,
                                        node_deg,
                                        edge_src,
                                        edge_dst,
                                        edge_attr,
                                        edge_length_embedded,
                                        numb, n)
        output = node_output[numb:].type(torch.complex128)
        output = torch.add(output[:, [0, 2, 3, 4, 8, 9, 10, 11, 12]], output[:, [1, 5, 6, 7, 13, 14, 15, 16, 17]], alpha = 1j)
        output = torch.matmul(output, self.irrep2tens.to(device = node_input.device))
        output = output.reshape((-1, 3, 3))
        Hs = self.glue(output, numb.item(), n)
        return Hs


class GraphNetwork_kMVN(nn.Module):
    """kMVN (k-Multi-Virtual Node) Graph Neural Network"""
    
    def __init__(self, mul, irreps_out, lmax, nlayers, number_of_basis, radial_layers, radial_neurons, 
                 node_dim, node_embed_dim, input_dim, input_embed_dim, use_attention=False,
                 attn_heads=4, attn_dropout=0.1):
        super().__init__()
        self.mul = mul
        self.irreps_in = Irreps(str(input_embed_dim) + 'x0e')
        self.irreps_node_attr = Irreps(str(node_embed_dim) + 'x0e')
        self.irreps_edge_attr = Irreps.spherical_harmonics(lmax)
        self.irreps_hidden = Irreps([(self.mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])
        self.irreps_out = Irreps(irreps_out)
        self.number_of_basis = number_of_basis

        self.act = {1: F.silu, -1: torch.tanh}
        self.act_gates = {1: F.sigmoid, -1: torch.tanh}
        
        # Embedding layers
        self.emx = nn.Linear(input_dim, input_embed_dim, dtype=torch.float64)
        self.emz = nn.Linear(node_dim, node_embed_dim, dtype=torch.float64)
        
        self.layers = self._build_layers(nlayers, number_of_basis, radial_layers, radial_neurons, use_attention, attn_heads, attn_dropout)
        
        # Add GraphHamiltonianConvolution layer
        self.layers.append(GraphHamiltonianConvolution(
            self.irreps_in_fin,   
            self.irreps_node_attr,
            self.irreps_edge_attr,
            self.irreps_out,
            number_of_basis,
            radial_layers,
            radial_neurons
        ))

    def _build_layers(self, nlayers, number_of_basis, radial_layers, radial_neurons, use_attention=False, attn_heads=4, attn_dropout=0.1):
        """Build layers for network with gates and convolutions."""
        layers = nn.ModuleList()
        irreps_in = self.irreps_in
        for i in range(nlayers):
            irreps_scalars = Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l == 0 and tp_path_exists(irreps_in, self.irreps_edge_attr, ir)])
            irreps_gated = Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l > 0 and tp_path_exists(irreps_in, self.irreps_edge_attr, ir)])
            ir = '0e' if tp_path_exists(irreps_in, self.irreps_edge_attr, '0e') else '0o'
            irreps_gates = Irreps([(self.mul, ir) for self.mul, _ in irreps_gated])

            gate = Gate(irreps_scalars, [self.act[ir.p] for _, ir in irreps_scalars],
                        irreps_gates, [self.act_gates[ir.p] for _, ir in irreps_gates],
                        irreps_gated)
            conv = GraphConvolution(irreps_in, self.irreps_node_attr, self.irreps_edge_attr, gate.irreps_in, number_of_basis, radial_layers, radial_neurons)

            irreps_in = gate.irreps_out
            
            # Add convolution+gate layer
            layers.append(CustomCompose(conv, gate))
            
            # Add attention layer (if enabled and not last convolution layer)
            if use_attention and i < nlayers - 1:
                attn_d_model = irreps_in.dim
                attn_layer = SelfAttentionLayer(attn_d_model, attn_heads, attn_dropout)
                layers.append(attn_layer)
        
        self.irreps_in_fin = irreps_in    
        return layers

    def forward(self, data):
        """Forward pass."""
        edge_src, edge_dst = data['edge_index']
        edge_vec = data['edge_vec']
        edge_len = data['edge_len']
        edge_length_embedded = soft_one_hot_linspace(edge_len, 0.0, data['r_max'].item(), self.number_of_basis, basis='gaussian', cutoff=False)
        edge_sh = spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization='component')
        edge_attr = edge_sh

        numb = data['numb']
        x = torch.relu(self.emx(torch.relu(data['x'])))
        z = torch.relu(self.emz(torch.relu(data['z'])))
        node_deg = data['node_deg']
        
        ucs = data['ucs'][0]
        n = len(ucs.shift_reverse)

        # Process layers: handle both E(3)NN conv layers and attention layers
        for layer in self.layers:
            if isinstance(layer, SelfAttentionLayer):
                # Attention layer: pass x, edge_index, edge_len
                x = layer(x, data['edge_index'], edge_len)
            else:
                # E(3)NN convolution layer: pass all required arguments
                x = layer(x, z, node_deg, edge_src, edge_dst, edge_attr, edge_length_embedded, numb, n)
        
        return x, torch.tensor(ucs.shift_reverse, dtype=torch.complex128).to(device=x.device)


def get_spectra(Hs, shifts, qpts):
    """Calculate spectra using Hamiltonians and q-points."""
    H = torch.sum(torch.mul(Hs.unsqueeze(1), torch.exp(2j*math.pi*torch.matmul(shifts, qpts.type(torch.complex128).t())).unsqueeze(-1).unsqueeze(-1)), dim = 0)
    eigvals = torch.linalg.eigvals(H)
    abx = torch.abs(eigvals)
    try:
        epsilon = torch.min(abx[abx > 0])/100
    except:
        epsilon = 1E-8
    eigvals = torch.sqrt(eigvals + epsilon)
    return torch.sort(torch.real(eigvals))[0]