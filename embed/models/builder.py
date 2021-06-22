import warnings

from embed.cv.utils import Registry, build_from_cfg
from torch import nn

ENCODERS = Registry('encoder')
POSITION_DECODING_GENERATORS = Registry('position_decoding_generator')
POSITION_DECODING_HEADS = Registry('position_decoding_head')
POSITION_EMBEDDING_HEADS = Registry('position_embedding_head')


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_encoder(cfg):
    """Build encoder."""
    return build(cfg, ENCODERS)


def build_position_decoding_generator(cfg):
    """Build position decoding generator."""
    return build(cfg, POSITION_DECODING_GENERATORS)


def build_position_decoding_head(cfg):
    """Build position decoding head."""
    return build(cfg, POSITION_DECODING_HEADS)


def build_position_embedding_head(cfg):
    """Build position embedding head."""
    return build(cfg, POSITION_EMBEDDING_HEADS)
