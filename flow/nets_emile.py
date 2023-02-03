# source: Emile, who's source is the pytorch implementation:
# source: https://github1s.com/ehoogeboom/e3_diffusion_for_molecules/blob/HEAD/egnn/models.py

from typing import Callable, Optional
from functools import partial
import dataclasses

import jax
from jax import vmap
import jax.numpy as jnp
import haiku as hk
import numpy as np

def get_activation(activation):
    return getattr(jax.nn, activation)


def scatter(input, dim, index, src, reduce=None):
    # Works like PyTorch's scatter. See https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html

    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )

    if reduce is None:
        _scatter = jax.lax.scatter
    elif reduce == "add":
        _scatter = jax.lax.scatter_add
    elif reduce == "multiply":
        _scatter = jax.lax.scatter_mul

    _scatter = partial(_scatter, dimension_numbers=dnums)
    vmap_inner = partial(vmap, in_axes=(0, 0, 0), out_axes=0)
    vmap_outer = partial(vmap, in_axes=(1, 1, 1), out_axes=1)

    for idx in range(len(input.shape)):
        if idx == dim:
            pass
        elif idx < dim:
            _scatter = vmap_inner(_scatter)
        else:
            _scatter = vmap_outer(_scatter)

    return _scatter(input, jnp.expand_dims(index, axis=-1), src)


@dataclasses.dataclass
class E_GCL(hk.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    output_nf: int
    hidden_nf: int
    act_fn: Callable = get_activation("silu")
    residual: bool = True
    attention: bool = False
    normalize: bool = False
    coords_agg: str = "mean"
    tanh: bool = False
    norm_constant: float = 1e-8

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = jnp.concatenate([source, target, radial], axis=1)
        else:
            out = jnp.concatenate([source, target, radial, edge_attr], axis=1)
        edge_mlp = hk.Sequential(
            [
                hk.Linear(self.hidden_nf),
                self.act_fn,
                hk.Linear(self.hidden_nf),
                self.act_fn,
            ]
        )
        out = edge_mlp(out)

        if self.attention:
            att_mlp = hk.Sequential([hk.Linear(1), jax.nn.sigmoid])
            att_val = att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.shape[0])
        if node_attr is not None:
            agg = jnp.concatenate([x, agg, node_attr], axis=1)
        else:
            agg = jnp.concatenate([x, agg], axis=1)

        node_mlp = hk.Sequential(
            [hk.Linear(self.hidden_nf), self.act_fn, hk.Linear(self.output_nf)]
        )
        out = node_mlp(agg)

        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index

        w_init = hk.initializers.VarianceScaling(0.001, "fan_avg", "uniform")
        # torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        layer = hk.Linear(1, with_bias=False, w_init=w_init)

        coord_mlp = []
        coord_mlp.append(hk.Linear(self.hidden_nf))
        coord_mlp.append(self.act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(jnp.tanh)
        coord_mlp = hk.Sequential(coord_mlp)
        trans = coord_diff * coord_mlp(edge_feat)

        if self.coords_agg == "sum":
            agg = unsorted_segment_sum(trans, row, num_segments=coord.shape[0])
        elif self.coords_agg == "mean":
            agg = unsorted_segment_mean(trans, row, num_segments=coord.shape[0])
        else:
            raise Exception("Wrong coords_agg parameter" % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = jnp.expand_dims(jnp.sum(coord_diff**2, 1), 1)

        if self.normalize:
            norm = jax.lax.stop_gradient(jnp.sqrt(radial)) + self.norm_constant
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def __call__(self, h, edge_index, y, edge_attr=None, node_attr=None):
        row, col = edge_index

        radial, diff = self.coord2radial(edge_index, y)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        y = self.coord_model(y, edge_index, diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, y, edge_attr


@dataclasses.dataclass
class EGNN(hk.Module):
    """
    :param hidden_nf: Number of hidden features
    :param out_node_nf: Number of features for 'h' at the output
    :param act_fn: Non-linearity
    :param n_layers: Number of layer for the EGNN
    :param residual: Use residual connections, we recommend not changing this one
    :param attention: Whether using attention or not
    :param normalize: Normalizes the coordinates messages such that:
                instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                We noticed it may help in the stability or generalization in some future works.
                We didn't use it in our paper.
    :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                    phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                    We didn't use it in our paper.
    """
    hidden_nf: int
    # out_node_nf: int
    act_fn: str = "silu"
    n_layers: int = 4
    residual: bool = True
    attention: bool = False
    normalize: bool = False
    tanh: bool = False
    coords_agg: str = "mean"
    norm_constant: int = 0

    def __call__(self, h, y, edges=None, edge_attr=None, node_attr=None):
        y_leading_shape = y.shape[:-1]
        if len(y.shape) == 3:
            if edges is None:
                edges, _ = get_edges_batch(n_nodes=y.shape[1], batch_size=y.shape[0])

        else:
            edges = get_edges(n_nodes=y.shape[0])
            edges = [jnp.array(edges[0]), jnp.array(edges[1])]


        h = jnp.reshape(h, (np.prod(y_leading_shape), h.shape[-1]))
        y = jnp.reshape(y, (np.prod(y_leading_shape), y.shape[-1]))

        edge_attr = jnp.sum((y[edges[0]] - y[edges[1]]) ** 2, axis=1, keepdims=True)

        embedding_in = hk.Linear(self.hidden_nf)
        embedding_out = hk.Linear(h.shape[-1])
        act_fn = get_activation(self.act_fn)
        h = embedding_in(h)
        for i in range(0, self.n_layers):
            layer = E_GCL(
                output_nf=self.hidden_nf,
                hidden_nf=self.hidden_nf,
                act_fn=act_fn,
                residual=self.residual,
                attention=self.attention,
                normalize=self.normalize,
                tanh=self.tanh,
                coords_agg=self.coords_agg,
                norm_constant=self.norm_constant,
            )
            h, y, _ = layer(h, edges, y, edge_attr=edge_attr, node_attr=node_attr)
        h = embedding_out(h)


        h = jnp.reshape(h, (*y_leading_shape, h.shape[-1]))
        y = jnp.reshape(y, (*y_leading_shape, y.shape[-1]))

        return h, y


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.shape[1])
    result = jnp.zeros(result_shape)  # Init empty result tensor.
    # segment_ids = segment_ids[..., None].expand(-1, data.shape[1])
    segment_ids = segment_ids[..., None]
    segment_ids = jnp.broadcast_to(
        segment_ids, (*segment_ids.shape[:-1], data.shape[-1])
    )
    # result.scatter_add_(0, segment_ids, data)
    result = scatter(result, 0, segment_ids, data, reduce="add")
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.shape[1])
    # segment_ids = segment_ids[..., None].expand(-1, data.shape[1])
    segment_ids = segment_ids[..., None]
    segment_ids = jnp.broadcast_to(
        segment_ids, (*segment_ids.shape[:-1], data.shape[-1])
    )
    result = jnp.zeros(result_shape)  # Init empty result tensor.
    count = jnp.zeros(result_shape)
    # result.scatter_add_(0, segment_ids, data)
    # count.scatter_add_(0, segment_ids, jnp.ones_like(data))
    result = scatter(result, 0, segment_ids, data, reduce="add")
    count = scatter(count, 0, segment_ids, jnp.ones_like(data), reduce="add")
    return result / jnp.clip(count, a_min=1)


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = jnp.ones((len(edges[0]) * batch_size, 1))
    # edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    edges = [jnp.array(edges[0]), jnp.array(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [jnp.concatenate(rows), jnp.concatenate(cols)]
    return edges, edge_attr
