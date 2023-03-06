import functools
import math
from typing import Callable, Optional, Union, Tuple

import haiku as hk
import jax
import numpy as np
import jax.numpy as jnp
import e3nn_jax as e3nn
from mace_jax import tools
from mace_jax.modules.models import safe_norm
from mace_jax.modules import (
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    MessagePassingConvolution
)
from nets.e3gnn_linear_haiku import Linear
from nets.e3gnn_blocks import HaikuMLP


def irreps_array_repeat(array: e3nn.IrrepsArray, n_repeat: int, axis: int):
    irreps = array.irreps
    new_array = jnp.repeat(array.array, n_repeat, axis=axis)
    return e3nn.IrrepsArray(irreps, new_array)


def get_edge_norm_vectors_and_lengths(positions, senders, receivers, eps):
    """shifts defaults to off and normalisation to True"""
    shifts = jnp.zeros_like(positions) * jnp.nan
    vectors = tools.get_edge_relative_vectors(
        positions=positions,
        senders=senders,
        receivers=receivers,
        shifts=shifts,
        cell=None,
        n_edge=senders.shape[0],
    )
    lengths = safe_norm(vectors, axis=-1)

    norm_vectors = vectors / jnp.clip(lengths, a_min=eps)[..., None]
    return norm_vectors, lengths


class FlexMACE(hk.Module):
    """Flexible Mace block for generative modelling."""
    def __init__(
        self,
        output_irreps: e3nn.Irreps,  # Irreps of the output
        mace_layer_output_irreps: e3nn.Irreps, # Irreps of the MaceLayer output, default  x0e+1x1o
        hidden_irreps: e3nn.Irreps,  # 256x0e or 128x0e + 128x1o
         # Layer Readout MLP params -- single hidden later e3nn gate based 
        readout_mlp_irreps: e3nn.Irreps,  # Hidden irreps of the MLP in last readout, default 16x0e
        num_features: int,  # Number of features per node, default gcd of hidden_irreps multiplicities
        avg_num_neighbors: float,
        
        max_ell: int = 5,  # Max spherical harmonic degree, default 5 for generative modelling 
        num_layers: int = 2,  # Number of interactions (layers), default 2
        correlation: int = 3,  # Correlation order at each layer (~ node_features^correlation), default 3
        
        num_species: int = 1,
        epsilon: float = 1e-6,
        

        activation: Callable = jax.nn.silu,  # activation function
        # Radial MLP params
        interaction_mlp_depth: int = 3,
        interaction_mlp_width: int = 256,
       
        # Residual MLP  params
        residual_mlp_width: int = 128,
        residual_mlp_depth: int = 1,
        
        
        # symmetric_tensor_product_basis: bool = False,  # this is used for the product contraction, which I am not looking into
        # off_diagonal: bool = False,
       
    ):
        super().__init__()
        self.output_irreps = output_irreps
        self.mace_layer_output_irreps  = mace_layer_output_irreps
        self.hidden_irreps = e3nn.Irreps(hidden_irreps)
        self.readout_mlp_irreps = e3nn.Irreps(readout_mlp_irreps)
        self.num_features = num_features

        # irreps of spherical harmonics expansion of vectors
        self.sh_irreps = e3nn.Irreps.spherical_harmonics(max_ell)[1:]  # discard 0e

        self.num_layers = num_layers
        self.correlation = correlation
        self.avg_num_neighbors = avg_num_neighbors
        self.epsilon = epsilon
        self.num_species = num_species
        
        self.symmetric_tensor_product_basis = False
        self.off_diagonal = False

        self.activation = activation
        self.residual_mlp_width = residual_mlp_width
        self.residual_mlp_depth = residual_mlp_depth
        self.interaction_mlp_depth = interaction_mlp_depth
        self.interaction_mlp_width = interaction_mlp_width

        # Embeddings
        # embedding for node features 
        #Note: this embeds into 0e, since self.hidden_irreps is filtered internally
        # given int a, the irreps product self.num_features * self.hidden_irreps does 1x0e -> ax0e
        self.node_embedding = LinearNodeEmbeddingBlock(
            self.num_species, self.num_features * self.hidden_irreps
        )
        # embedding for shared features
        self.hidden_scalar_dim = self.num_features * self.hidden_irreps.filter("0e").regroup().dim
        self.shared_feat_embedding = hk.Linear(self.hidden_scalar_dim)

    def __call__(
        self,
        # vectors: jnp.ndarray,  # [n_edges, 3]
        positions: jnp.ndarray, # [n_nodes, 3]
        node_specie: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        shared_features: jnp.ndarray, # [dim_shared_features] features shared among all nodes, like time, or total number of atoms in system
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
    ) -> e3nn.IrrepsArray:
        assert positions.ndim == 2 and positions.shape[1] == 3
        assert node_specie.ndim == 1
        assert senders.ndim == 1 and receivers.ndim == 1

        positions = positions[:, None, :]
        positions = e3nn.IrrepsArray("1x1o", positions - jnp.mean(positions, axis=0, keepdims=True))
        positions_in = positions
        
        # Note features are Embeddings of discrete node "species" (i.e. hidrogen, carbon,z)
        node_feats = self.node_embedding(node_specie).astype(
            positions.dtype
        ) /  self.num_species #  importance normalisation for diffusion -
        + self.shared_feat_embedding(shared_features) # add shared info to embedding [n_nodes, feature * irreps]
        # node_feats = profile("embedding: node_feats", node_feats)

        # node_feats are rotation invariant scalars (0e) initially but become full irreps after MACE layers

        residual_node_feats = node_feats.filter('0e')
        
        # Interactions
        # outputs = []

        for i in range(self.num_layers):
            norm_vectors, lengths = jax.vmap(get_edge_norm_vectors_and_lengths, in_axes=(1, None, None, None),
                                             out_axes=1)(
                                                    positions.array,
                                                    senders,
                                                    receivers,
                                                    self.epsilon)  # [n_edges, 3]  [n_edges,]
            lengths = jnp.reshape(lengths, (lengths.shape[0], np.prod(lengths.shape[1:])))

            # Mace Layer
            first = i == 0
            if first:
                interaction_irreps = self.hidden_irreps.filter("0e")
                lengths_0 = lengths
            else:
                interaction_irreps = self.hidden_irreps


            many_body_scalars, many_body_vectors, node_feats = FlexMACELayer(  
                num_features=self.num_features,
                interaction_irreps=interaction_irreps,
                hidden_irreps=self.hidden_irreps,
                avg_num_neighbors=self.avg_num_neighbors,
                activation=self.activation,
                num_species=self.num_species,
                epsilon=self.epsilon,
                correlation=self.correlation,
                output_irreps=self.mace_layer_output_irreps,
                readout_mlp_irreps=self.readout_mlp_irreps,
                symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
                off_diagonal=self.off_diagonal,
                name=f"layer_{i}",
                sh_irreps=self.sh_irreps,
                interaction_mlp_depth=self.interaction_mlp_depth,
                interaction_mlp_width=self.interaction_mlp_width,
            )(
                vectors=norm_vectors,
                lengths=lengths_0,
                node_feats=node_feats,
                node_specie=node_specie,  # not sure if we are going to need this
                scalar_edge_feats=lengths,
                senders=senders,
                receivers=receivers,
            )
            if first:
                positions = irreps_array_repeat(positions, self.mace_layer_output_irreps.filter("1o").num_irreps, axis=1)
            # residual paths

            # update vectors and lengths note that this presserves equivariance
            #this can be summed because L1 harmonics are just vectors 
            positions = positions + many_body_vectors.mul_to_axis(axis=1)
            # update node features 
            mlp_inputs = e3nn.concatenate([residual_node_feats, many_body_scalars], axis=-1)
            mlp_layer_sizes = (self.residual_mlp_depth-1) * [self.residual_mlp_width] + [self.hidden_scalar_dim]
            residual_node_feats = residual_node_feats + HaikuMLP(
             output_sizes=mlp_layer_sizes,
             activation=self.activation, activate_final=False)(mlp_inputs)  # [self.num_features * self.hidden_irreps]

        # Note that in this configuration only scalars will be outputed
        readout_in = e3nn.concatenate([residual_node_feats, (positions - positions_in).axis_to_mul(axis=1)])
        node_outputs = Linear(self.output_irreps,
                name="output_linear", biases=True)(readout_in)
        return node_outputs  # [n_nodes, output_irreps]




class FlexMACELayer(hk.Module):
    def __init__(
        self,
        num_features: int,
        interaction_irreps: e3nn.Irreps,  # irreps to expand into for interaction block (without num feat multiplicity)
        hidden_irreps: e3nn.Irreps,
        activation: Callable,
        num_species: int,
        epsilon: Optional[float],
        name: Optional[str],
        # InteractionBlock:
        avg_num_neighbors: float,
        # EquivariantProductBasisBlock:
        correlation: int,
        symmetric_tensor_product_basis: bool,
        off_diagonal: bool,
        # ReadoutBlock:
        output_irreps: e3nn.Irreps,
        readout_mlp_irreps: e3nn.Irreps,
        # adding new inputs 
        sh_irreps: e3nn.Irreps,
        interaction_mlp_width: int,
        interaction_mlp_depth: int,

    ) -> None:
        super().__init__(name=name)

        self.num_features = num_features
        
        self.sh_irreps = sh_irreps
        self.interaction_irreps = interaction_irreps 
        self.hidden_irreps = hidden_irreps
        self.output_irreps = output_irreps
        self.readout_mlp_irreps = readout_mlp_irreps
        
        self.avg_num_neighbors = avg_num_neighbors
        self.activation = activation
        self.num_species = num_species
        self.epsilon = epsilon
        self.correlation = correlation

        self.symmetric_tensor_product_basis = symmetric_tensor_product_basis
        self.off_diagonal = off_diagonal
        # new inputs
        self.interaction_mlp_width = interaction_mlp_width
        self.interaction_mlp_depth = interaction_mlp_depth
        

    def __call__(
        self,
        # # these are new
        vectors: jnp.ndarray, # norm_vectors [n_edges, 3]
        lengths: jnp.ndarray, # lengths_0  [n_edges, ]
        # # these were there before
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        node_specie: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        scalar_edge_feats: jnp.ndarray,  # lengths [n_edges, ]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
    ):
        edge_feats = e3nn.spherical_harmonics(
                self.sh_irreps,
                vectors,
                normalize=False,
                normalization="component",
                )  # [n_edges, sh_irreps]
        edge_feats = edge_feats.axis_to_mul(axis=-2)

            
        node_feats = FlexInteractionBlock(
            target_irreps=self.num_features * self.hidden_irreps,
            hidden_irreps=self.num_features * self.interaction_irreps,
            avg_num_neighbors=self.avg_num_neighbors,
            mlp_width = self.interaction_mlp_width,
            mlp_depth = self.interaction_mlp_depth,
            activation=self.activation,
        )(
            node_feats=node_feats,
            edge_feats=edge_feats,
            scalar_edge_feats=scalar_edge_feats,
            lengths = lengths,
            receivers=receivers,
            senders=senders,
        ) # [n_nodes, target_irreps]

        # node_feats = profile(f"{self.name}: node_feats after interaction", node_feats)


        node_feats = FlexEquivariantProductBasisBlock(
            target_irreps=self.num_features * self.hidden_irreps,
            correlation=self.correlation,
            num_species=self.num_species,
            symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
            off_diagonal=self.off_diagonal,
        )(node_feats=node_feats, node_specie=node_specie)  # node specie is not used in PT DiffusionMace but I will leave it here

        # node_feats = profile(f"{self.name}: node_feats after tensor power", node_feats)

        
        # TODO: this readout block could combine info from both
        #  incoming node feats and producted node feats
        # TODO: Could pop scalar MLP before this readout 
        # TODO: make sure that this output is 0e + 1o
        
        node_outputs = FlexNonLinearReadoutBlock(
            hidden_irreps=self.num_features * self.readout_mlp_irreps,
            output_irreps=self.output_irreps,
            activation=self.activation,
            gate=self.activation,
        )(
            node_feats
        )  # [n_nodes, output_irreps]

        # node_outputs = profile(f"{self.name}: node_outputs", node_outputs)
        
        return node_outputs.filter("0e"), node_outputs.filter("1o"), node_feats
        # return many_body_scalars, many_body_vectors, node_feats
        

class FlexNonLinearReadoutBlock(hk.Module):
    """This is a funky single hidden layer MLP --
    Hidden layer can be wider than the input irreps but not higher order
      as this operation can create irreps."""
    #TODO: add flexibility by adding a scalar MLP before the funky interaction
    def __init__(
        self,
        hidden_irreps: e3nn.Irreps,
        output_irreps: e3nn.Irreps,
        activation: Optional[Callable] = None,
        gate: Optional[Callable] = None,
    ):
        super().__init__()
        self.hidden_irreps = hidden_irreps
        self.output_irreps = output_irreps
        self.activation = activation
        self.gate = gate

    def __call__(self, x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        # x = [n_nodes, irreps]
        num_vectors = (
            self.hidden_irreps.num_irreps
            - self.hidden_irreps.filter(["0e", "0o"]).num_irreps
        )  # Multiplicity of (l > 0) irreps for which we need extra scalars to act as gates
        
        # input linear
        x = Linear(
            (self.hidden_irreps + e3nn.Irreps(f"{num_vectors}x0e")).simplify(),
            biases=True)(x)
        
        # given k l>0 irreps, passes first n-k scalars through act and concatenates 
        # to l>0 irreps multiplied with next k scalars passed through gate_act
        # activations must be Callable[[float], float]]
        x = e3nn.gate(x, even_act=self.activation, even_gate_act=self.gate)
        
        # output lineanode_feats
        return Linear(self.output_irreps, biases=True)(x)  # [n_nodes, output_irreps]
    
        
class FlexInteractionBlock(hk.Module):
    def __init__(
        self,
        *,
        target_irreps: e3nn.Irreps,  #
        hidden_irreps: e3nn.Irreps,
        avg_num_neighbors: float,
        mlp_width = int,
        mlp_depth: int,
        activation: Callable,
    ) -> None:
        super().__init__()
        self.hidden_irreps = hidden_irreps
        self.target_irreps = target_irreps
        self.avg_num_neighbors = avg_num_neighbors
        self.mlp_width = mlp_width
        self.mlp_depth = mlp_depth
        self.activation = activation

    def __call__(
        self,
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        edge_feats: e3nn.IrrepsArray,  # [n_edges, irreps]
        scalar_edge_feats: jnp.ndarray,  # [n_edges, ] # TODO: make into irreps array
        lengths: jnp.ndarray,   # [n_edges, ]
        senders: jnp.ndarray,  # [n_edges, ]
        receivers: jnp.ndarray,  # [n_edges, ]
    ) -> Tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]:
        assert node_feats.ndim == 2

        # input linear (cant increase order but can increase channels)
        node_feats = Linear(self.hidden_irreps, name="linear_up", biases=True)(node_feats)
        
        # This outputs target irreps since these may be larger than hidden irreps in first layer
        node_feats = FlexMessagePassingConvolution(
            avg_num_neighbors=self.avg_num_neighbors, target_irreps=self.target_irreps,
            mlp_width=self.mlp_width,
            mlp_depth=self.mlp_depth, activation=self.activation
        )(node_feats, edge_feats, scalar_edge_feats, lengths, senders, receivers)

        # output linear
        node_feats = Linear(self.target_irreps, name="linear_down", biases=True)(
            node_feats
        )

        assert node_feats.ndim == 2
        # returning node feats is correct
        return node_feats  # [n_nodes, target_irreps]
    
    
    
class FlexMessagePassingConvolution(hk.Module):
    def __init__(
        self,
        avg_num_neighbors: float,
        target_irreps: e3nn.Irreps,  # irreps of module output
        mlp_width: int = 128,
        mlp_depth: int = 3,
        activation: Callable = jax.nn.silu,
    ):
        super().__init__()
        self.avg_num_neighbors = avg_num_neighbors
        # TODO: why is this like this?
        # self.target_irreps = e3nn.Irreps(target_irreps)
        self.target_irreps = target_irreps
        
        self.mlp_width = mlp_width
        self.mlp_depth = mlp_depth
        self.activation = activation

    def __call__(
        self,
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        edge_feats: e3nn.IrrepsArray,  # [n_edges, irreps]
        ## new 
        scalar_edge_feats: jnp.ndarray,  # some sort of scalars? [n_edges, feats?]  #  TODO: size of this
        lengths: jnp.ndarray, # [n_edges, ]
        ## -- end new --
        senders: jnp.ndarray,  # [n_edges, ]
        receivers: jnp.ndarray,  # [n_edges, ]
    ) -> e3nn.IrrepsArray:
        assert node_feats.ndim == 2
        assert edge_feats.ndim == 2

        messages = node_feats[senders]
        # note that tensor product does not increase multiplicity. target_irreps just specifies order
        messages = e3nn.tensor_product(
                    messages,
                    edge_feats.filter(drop="0e"),  # TODO: check if we need to drop these scalars
                    filter_ir_out=self.target_irreps,
                )  # [n_edges, irreps]

        node_scalars = node_feats.filter('0e')
        mlp_input_features = e3nn.concatenate(
            [node_scalars[senders], node_scalars[receivers],
             scalar_edge_feats, lengths], axis=1).regroup()  # we should get one per edge
        
        # MLP is not equivariant, applies to scalars only 
        mix = HaikuMLP(
            output_sizes=(self.mlp_depth - 1) * [self.mlp_width] + [messages.irreps.num_irreps], # number of vectors that transform independently, counting channels as different vectors
            activation=self.activation,
            activate_final=False,
        )(
            mlp_input_features.filter(keep="0e")   # vector should only contain 0e anyway
        )  # [n_edges, num_irreps]
 
        # irreps only elementwise product with arrays of size num_irreps

        messages = messages * mix  # [n_edges, irreps]
        
        # scatter operation
        zeros = e3nn.IrrepsArray.zeros(
            messages.irreps, node_feats.shape[:1], messages.dtype
        )
        
        # skip layer
        node_feats = zeros.at[receivers].add(messages)  # [n_nodes, irreps]

        return node_feats / self.avg_num_neighbors

from mace_jax.modules import SymmetricContraction


class FlexEquivariantProductBasisBlock(hk.Module):
    def __init__(
        self,
        target_irreps: e3nn.Irreps,
        correlation: int,
        num_species: int,
        symmetric_tensor_product_basis: bool = True,
        off_diagonal: bool = False,
    ) -> None:
        super().__init__()
        self.target_irreps = e3nn.Irreps(target_irreps)
        self.symmetric_contractions = SymmetricContraction(
            keep_irrep_out={ir for _, ir in self.target_irreps},
            correlation=correlation,
            num_species=num_species,
            gradient_normalization="element",  # NOTE: This is to copy mace-torch
            symmetric_tensor_product_basis=symmetric_tensor_product_basis,
            off_diagonal=off_diagonal,
        )

    def __call__(
        self,
        node_feats: e3nn.IrrepsArray,  # [n_nodes, feature * irreps]
        node_specie: jnp.ndarray,  # [n_nodes, ] int
    ) -> e3nn.IrrepsArray:
        node_feats = node_feats.mul_to_axis().remove_nones()
        node_feats = self.symmetric_contractions(node_feats, node_specie)
        node_feats = node_feats.axis_to_mul()
        return Linear(self.target_irreps, biases=True)(node_feats)