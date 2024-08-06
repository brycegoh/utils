from .nodes import *

NODE_CLASS_MAPPINGS = {
    'BlackoutMLSD': BlackoutMLSD,
    'PasteMask': PasteMask,
    'PrintImageSize': PrintImageSize,
}

__all__ = ['NODE_CLASS_MAPPINGS']

