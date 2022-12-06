try:
    import mirtk
    __mirtk_installed__ = True
except ModuleNotFoundError:
    __mirtk_installed__ = False

try:
    import torch
    __torch_installed__ = True
except:
    __torch_installed__ = False

try:
    import vtk
    __vtk_installed__ = True
except ModuleNotFoundError:
    __vtk_installed__ = False


try:
    import trimesh
    __trimesh_installed__ = True
except ModuleNotFoundError:
    __trimesh_installed__ = False


from .image import *  # nibabel scipy
__all__ = image.__all__

if __mirtk_installed__ and __vtk_installed__:
    from . import register  # mirtk vtk
    from . import refine  # SimpleITK, vtk, mirtk
    from . import motion  # mirtk
    __all__ += ["register", "refine", "motion"]

if __torch_installed__:
    from . import nn  # torch
    __all__ += ["nn"]
    if __mirtk_installed__:
            from . import segment
            __all__ += ["segment"]

if __vtk_installed__:
    from . import landmark  # vtk nibabel
    from . import mesh  # vtk
    __all__ += ["landmark", "mesh", "visual"]
    try:
        import imageio, trimesh, meshcut, vedo
        from . import visual  # imageio trimesh vtk meshcut vedo
        __all__ += ["visual"]
    except ModuleNotFoundError:
        pass
