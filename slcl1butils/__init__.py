
__all__ = ['utils','compute','plotting','scripts','coloc','compute.macs']
try:
    from importlib import metadata
except ImportError: # for Python<3.8
    import importlib_metadata as metadata
__version__ = metadata.version('slcl1butils')
