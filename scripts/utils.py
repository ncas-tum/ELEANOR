import os
from dataclasses import dataclass

import equinox as eqx
import orbax.checkpoint as ocp
from jax import custom_jvp
from etils import epath
from aqt.jax.v2.aqt_quantizer import Quantizer


def make_fake_quant(quantizer: Quantizer, calibration_axes=None):
    @custom_jvp
    def fake_quant(x):
        x_q, _ = quantizer.quant(x, calibration_axes=calibration_axes)
        return x_q.dequant()

    @fake_quant.defjvp
    def fake_quant_jvp(primals, tangents):
        (x,) = primals
        (x_dot,) = tangents
        x_q, grad_fn = quantizer.quant(x, calibration_axes=calibration_axes)
        primal_out = x_q.dequant()
        tangent_out = grad_fn(x_dot)[0]
        return primal_out, tangent_out

    return fake_quant


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
