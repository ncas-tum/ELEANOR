import os
from typing import Sequence
from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrand
import orbax.checkpoint as ocp
from chex import Array, PRNGKey
from etils import epath
from snnax.snn.architecture import GraphStructure, ForwardFnOutput
from snnax.snn.layers.stateful import StatefulLayer, RequiresStateLayer


class EquinoxCheckpointHandler(ocp.CheckpointHandler):
    def save(
        self,
        directory: epath.Path,
        args: "EquinoxStateSave",
    ):
        full_path = os.path.join(directory, "model.eqx")
        eqx.tree_serialise_leaves(full_path, args.item, is_leaf=eqx.is_array_like)

    def restore(
        self,
        directory: epath.Path,
        args: "EquinoxStateRestore",
    ) -> eqx.Module:
        loaded = eqx.tree_deserialise_leaves(
            os.path.join(directory, "model.eqx"), args.item, is_leaf=eqx.is_array_like
        )
        return loaded


@ocp.args.register_with_handler(EquinoxCheckpointHandler, for_save=True)
@dataclass
class EquinoxStateSave(ocp.args.CheckpointArgs):
    item: eqx.Module


@ocp.args.register_with_handler(EquinoxCheckpointHandler, for_restore=True)
@dataclass
class EquinoxStateRestore(ocp.args.CheckpointArgs):
    item: eqx.Module


def forward_fn(
    layers: Sequence[eqx.Module],
    struct: GraphStructure,
    key: PRNGKey,
    states: Sequence[Array],
    data: Sequence[Array],
) -> ForwardFnOutput:
    """
    Computes the forward pass (via jax.lax.scan) through the layers in a
    straight-through manner, i.e. every layer takes the input from the last
    layer at the same time step. The layers are traversed in the order specified
    by the connectivity graph.

    Arguments:
        `layers`: Specifies layers in our model.
        `struct`: Specifies graph structure
        `states`: States as returned by init_state
        `data`: Input Sequence data of the model.
        `key`: Random key for the forward pass.
    """
    keys = jrand.split(key, len(layers))
    new_states, new_outs = [], []
    batch = data if isinstance(data, Sequence) else [data]
    data = data if isinstance(data, Sequence) else [data]

    for ilayer, (key, state, layer) in enumerate(zip(keys, states, layers)):
        # Grab output from nodes for which the connectivity graph
        # specifies a connection
        inputs, inputs_v = [], []

        # If the node is also a input layer, also append external input
        if ilayer in struct.input_layer_ids:
            inputs.append(batch)
            inputs_v.append(batch)

        # TODO suboptimal solution below, won't generalize to "deeper"" states

        for layer_id in struct.input_connectivity[ilayer]:
            if type(states[layer_id][-1]) is list:
                inputs.append(states[layer_id][-1][-1])
                inputs_v.append(states[layer_id][-1][0])
            else:
                inputs.append(states[layer_id][-1])
                inputs_v.append(states[layer_id][-1])

        # If the layer also gets external input append it as well
        external_inputs = [data[id] for id in struct.input_layer_ids[ilayer]]
        inputs += external_inputs
        inputs_v += external_inputs

        if len(inputs) == 1:
            inputs = jnp.concatenate(inputs, axis=0)
        inputs_v = jnp.concatenate(inputs_v, axis=0)

        # Check if layer is a StatefulLayer
        if isinstance(layer, StatefulLayer):
            new_state, new_out = layer(state, inputs, key=key)
            new_states.append(new_state)
            # if ilayer == len(layers) - 1:
            new_outs.append(new_out)
        elif isinstance(layer, RequiresStateLayer):
            new_out = layer(inputs_v, key=key)
            new_states.append([new_out])
            # if ilayer == len(layers) - 1:
            new_outs.append(new_out)
        else:
            new_out = layer(inputs, key=key)
            new_states.append([new_out])
            # if ilayer == len(layers) - 1:
            new_outs.append(new_out)

    return new_states, new_outs
