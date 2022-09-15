try:
    import mirtk
    __mirtk_installed__ = True
except ModuleNotFoundError:
    __mirtk_installed__ = False

try:
    import torch
    __torch_installed__ = True
except ModuleNotFoundError:
    __torch_installed__ = False

try:
    import vtk
    __vtk_installed__ = True
except ModuleNotFoundError:
    __vtk_installed__ = False


from .image import *
__all__ = image.__all__

if __mirtk_installed__ and __vtk_installed__:
    from . import register
    from . import refine
    __all__ += ["register", "refine"]

if __torch_installed__ and __mirtk_installed__:
    from . import segment
    __all__ += ["segment"]

if __vtk_installed__:
    from . import landmark
    from . import mesh
    from . import visual
    __all__ += ["landmark", "mesh", "visual"]
