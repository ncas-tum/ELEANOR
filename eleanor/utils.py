import os
from dataclasses import dataclass

import equinox as eqx
import orbax.checkpoint as ocp
from etils import epath


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
