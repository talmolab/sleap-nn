"""Miscellaneous utility functions for Inference modules."""

from omegaconf import OmegaConf
import sleap_io as sio


def get_skeleton_from_config(skeleton_config: OmegaConf):
    """Create Sleap-io Skeleton objects from config.

    Args:
        skeleton_config: OmegaConf object containing the skeleton config.

    Returns:
        Returns a list of `sio.Skeleton` objects created from the skeleton config
        stored in the `training_config.yaml`.

    """
    skeletons = []
    for name in skeleton_config.keys():

        # create `sio.Node` object.
        nodes = [
            sio.model.skeleton.Node(n["name"]) for n in skeleton_config[name].nodes
        ]

        # create `sio.Edge` object.
        edges = [
            sio.model.skeleton.Edge(
                sio.model.skeleton.Node(e["source"]["name"]),
                sio.model.skeleton.Node(e["destination"]["name"]),
            )
            for e in skeleton_config[name].edges
        ]

        # create `sio.Symmetry` object.
        if skeleton_config[name].symmetries:
            list_args = [
                set(
                    [
                        sio.model.skeleton.Node(s[0]["name"]),
                        sio.model.skeleton.Node(s[1]["name"]),
                    ]
                )
                for s in skeleton_config[name].symmetries
            ]
            symmetries = [sio.model.skeleton.Symmetry(x) for x in list_args]
        else:
            symmetries = []

        skeletons.append(sio.model.skeleton.Skeleton(nodes, edges, symmetries, name))

    return skeletons
