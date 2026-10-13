"""Spacetime tube and entity-bank export utilities for ReferGaussian."""

__all__ = ["export_entitybank"]


def __getattr__(name):
    if name == "export_entitybank":
        from .export import export_entitybank
        return export_entitybank
    raise AttributeError(name)
