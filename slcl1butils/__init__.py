from . import coloc, compute, legacy_ocean, plotting

__all__ = ["utils", "compute", "plotting", "scripts", "coloc", "legacy_ocean"]
try:
    from importlib import metadata
except ImportError:  # for Python<3.8
    import importlib_metadata as metadata
__version__ = metadata.version("slcl1butils")
