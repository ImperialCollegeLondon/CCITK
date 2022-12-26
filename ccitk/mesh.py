__all__ = [
    "read_vkt_mesh",
    "surface_mesh_to_volumetric_mesh",
    "decimate_mesh",
    "extract_mesh_from_segmentation",
]

import vtk
import numpy as np
from pathlib import Path
from vtk.util.numpy_support import vtk_to_numpy
from typing import Tuple, List


def read_vkt_mesh(vtk_path: Path, affine: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read VTk mesh, and apply affine transformation if provided
    Args:
        vtk_path: mesh vtk path
        affine (optional): affine transformation matrix (4 x 4)

    Returns:
        A tuple, (vertices, triangles)
        vertices has shape (N1, 3), and triangles has shape (N2, 3), where each value is an index of the vertices array.
    """
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(vtk_path))
    reader.Update()
    mesh = reader.GetOutput()

    vertices = vtk_to_numpy(mesh.GetPoints().GetData())
    vertices = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
    if affine is not None:
        vertices = np.matmul(affine, np.transpose(vertices))
        vertices = np.transpose(vertices)
    vertices = vertices[:, :3]

    triangles = vtk_to_numpy(mesh.GetPolys().GetConnectivityArray())
    triangles = np.reshape(triangles, (-1, 3))
    return vertices, triangles


def surface_mesh_to_volumetric_mesh(vertices: np.ndarray, triangles: np.ndarray, plot: bool = False) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform surface mesh to volumetric mesh. pyvista and tetgen library required.
    Args:
        vertices: an array of 3D points, (N1, 3)
        triangles: list of indices of vertices array, (N2, 3)
        plot (optional): whether to plot the result

    Returns:
        A tuple, (nodes, tetras).
        nodes is an array of points (N1, 3), and tetras is an array of indices (N3, 4)

    """
    import pyvista as pv
    import tetgen
    triangles = np.concatenate([np.ones((triangles.shape[0], 1)) * 3, triangles], axis=1)
    triangles = np.int32(triangles)
    mesh = pv.PolyData(vertices, triangles)

    tet = tetgen.TetGen(mesh)

    # nodes, tetras = tet.tetrahedralize(order=1, mindihedral=10, minratio=1.5)
    nodes, tetras = tet.tetrahedralize()

    if plot:
        grid = tet.grid
        grid.plot(show_edges=True)

    return nodes, tetras


def decimate_mesh(mesh_path: Path, output_path: Path, downsample_rate: float, preserve_topology: bool = True,
                  register: bool = True) -> Path:
    """Decimate (remove a large portion of) mesh

    Args:
        mesh_path: path to mesh vtk
        output_path: output mesh vtk path
        downsample_rate: decimate percentage, 98.8 means 98.8% removed.
        preserve_topology (optional): whether to preserve the topology of the mesh in decimation
        register (optional): whether to register after decimation to decrease errors.

    Returns:
        Path to the output mesh vtk
    """
    import mirtk
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(str(mesh_path))
    reader.Update()

    polydata = reader.GetOutput()
    print("Mesh number of points: ", polydata.GetPoints().GetNumberOfPoints())
    print('Decimating')
    mirtk.decimate_surface(
        str(mesh_path),
        str(output_path),
        reduceby=downsample_rate,
        preservetopology="on" if preserve_topology else "off",
    )
    if register:
        print('Registering')
        mirtk.register(
            str(output_path),
            str(mesh_path),
            "-par", "Point set distance correspondence", "CP",
            model="FFD",
            dofout=str(output_path.parent.joinpath("downsample.dof.gz")),
        )
        print('Transforming points')
        mirtk.transform_points(
            str(output_path),
            str(output_path),
            dofin=str(output_path.parent.joinpath("downsample.dof.gz")),
        )
        output_path.parent.joinpath("downsample.dof.gz").unlink()
    return output_path


def extract_mesh_from_segmentation(
        segmentation: Path, output_path: Path, labels: List[int] = None, phase: str = None, iso_value: int = 120,
        blur: int = 2, overwrite: bool = False
):
    """Convert mesh to 3D segmentation using Marching Cubes

    Args:
        segmentation: path to segmentation
        output_path: output mesh path
        labels (optional): list of label indices to use to extract segmentations.
        phase (optional): If labels is not None, then phase can be part of the output name if provided.
        iso_value (optional):
        blur (optional):
        overwrite (optional): whether to overwrite of output exists

    Returns:
        output segmentation path

    """
    import mirtk
    if not output_path.exists() or overwrite:
        if labels is not None:
            if len(labels) > 0:
                temp_dir = output_path.parent.joinpath("temp")
                temp_dir.mkdir(exist_ok=True, parents=True)
                suffix = "_".join([str(l) for l in labels])
                output = temp_dir.joinpath(f"{segmentation.stem}_{phase}_{suffix}.nii.gz") \
                    if phase is not None else temp_dir.joinpath(f"{segmentation.stem}_{suffix}.nii.gz")
                mirtk.calculate_element_wise(
                    str(segmentation),
                    "-label", *labels, set=255, pad=0,
                    output=str(output),
                )
                mirtk.extract_surface(
                    str(output),
                    str(output_path),
                    isovalue=iso_value, blur=blur,
                )
                # shutil.rmtree(str(temp_dir))
                return output_path
        mirtk.extract_surface(
            str(segmentation),
            str(output_path),
            isovalue=iso_value, blur=blur,
        )
    return output_path
