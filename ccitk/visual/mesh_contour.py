__all__ = [
 "visualize_mesh_contour_motion",
]
import os
import vtk
import trimesh

import meshcut
import numpy as np
from pathlib import Path
from typing import List, Dict
from vtk.util.numpy_support import vtk_to_numpy

import imageio
import itertools
import numpy.linalg as la
from ccitk.image import read_nii_image
from matplotlib import pyplot as plt


def points3d(verts, point_size=3, **kwargs):
    import mayavi.mlab as mlab
    if 'mode' not in kwargs:
        kwargs['mode'] = 'point'
    p = mlab.points3d(verts[:, 0], verts[:, 1], verts[:, 2], **kwargs)
    p.actor.property.point_size = point_size


def trimesh3d(verts, faces, **kwargs):
    import mayavi.mlab as mlab
    mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], faces,
                         **kwargs)


def orthogonal_vector(v):
    """Return an arbitrary vector that is orthogonal to v"""
    if v[1] != 0 or v[2] != 0:
        c = (1, 0, 0)
    else:
        c = (0, 1, 0)
    return np.cross(v, c)


def show_plane(orig, n, scale=1.0, **kwargs):
    """
    Show the plane with the given origin and normal. scale give its size
    """
    b1 = orthogonal_vector(n)
    b1 /= la.norm(b1)
    b2 = np.cross(b1, n)
    b2 /= la.norm(b2)
    verts = [orig + scale*(-b1 - b2),
             orig + scale*(b1 - b2),
             orig + scale*(b1 + b2),
             orig + scale*(-b1 + b2)]
    faces = [(0, 1, 2), (0, 2, 3)]
    trimesh3d(np.array(verts), faces, **kwargs)


def show(mesh, plane, expected_n_contours):
    import mayavi.mlab as mlab
    P = meshcut.cross_section_mesh(mesh, plane)
    colors = [
        (0, 1, 1),
        (1, 0, 1),
        (0, 0, 1)
    ]
    print("num contours : ", len(P), ' expected : ', expected_n_contours)

    if True:
        trimesh3d(mesh.verts, mesh.tris, color=(1, 1, 1),
                        opacity=0.5)
        show_plane(plane.orig, plane.n, scale=1, color=(1, 0, 0),
                         opacity=0.5)

        for p, color in zip(P, itertools.cycle(colors)):
            p = np.array(p)
            mlab.plot3d(p[:, 0], p[:, 1], p[:, 2], tube_radius=None,
                        line_width=3.0, color=color)
    return P


def visualize_mesh_contour_motion(
        motion_mesh_dir: Path, image_phases_dir: Path, output_dir: Path, slice_numbers: List[int],
        mlab_plot: bool = False, file_prefix: List[str] = None, dofin: Dict = None,
):
    """For each motion mesh in subject space, slice the mesh at certain z values
    to find the contours of the 2D slice plane and the mesh boundary surface
    and construct 2D motions of these contours for each slice.

    Args:
        motion_mesh_dir: motion mesh directory
        image_phases_dir: subject enlarged gray phases directory
        output_dir: output directory
        slice_numbers: z values of the images from which to show the contour motion
        mlab_plot (optional): whether to use mlab to plot (for debug)
        file_prefix (optional): file prefix for input mesh directory
        dofin (optional): if the motion mesh is in atlas space rather than subject space,
                          provide a transformation that maps mesh back to subject space

    Returns:

    """
    if file_prefix is None:
        file_prefix = ["LV_endo", "LV_epi", "RV"]
    output_dir.mkdir(exist_ok=True, parents=True)
    if dofin is not None:
        import mirtk
    slice_filenames = {}
    for slice_num in slice_numbers:
        slice_filenames[slice_num] = []

    motion_vtks = {}
    for fp in file_prefix:
        motion_vtks[fp] = list(os.listdir(str(motion_mesh_dir.joinpath(fp))))
    image_phases = [image_phases_dir.joinpath(file) for file in os.listdir(str(image_phases_dir))]
    for idx, vtk_name in enumerate(motion_vtks[file_prefix[0]]):
        print(vtk_name)
        img_nii_path = image_phases[idx]
        image, affine = read_nii_image(img_nii_path)

        affine = np.linalg.inv(affine)
        meshes = {}
        for prefix, color in zip(file_prefix, ["r.", "g.", "b."]):
            vtk_path = motion_mesh_dir.joinpath(prefix, motion_vtks[prefix][idx])

            temp_vtk_path = output_dir.joinpath("temp", f"vtk_{prefix}_{idx}.vtk")
            temp_vtk_path.parent.mkdir(parents=True, exist_ok=True)
            if dofin is not None:
                rv_dof = dofin["rv"]
                lv_dof = dofin["lv"]
                if prefix == "RV":
                    mirtk.transform_points(
                        str(vtk_path),
                        str(temp_vtk_path),
                        invert=None,
                        dofin=str(rv_dof)
                    )
                else:
                    mirtk.transform_points(
                        str(vtk_path),
                        str(temp_vtk_path),
                        invert=None,
                        dofin=str(lv_dof)
                    )

            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(str(vtk_path))
            reader.Update()
            mesh = reader.GetOutput()

            vertices = vtk_to_numpy(mesh.GetPoints().GetData())
            vertices = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
            vertices = np.matmul(affine, np.transpose(vertices))
            vertices = np.transpose(vertices)
            vertices = vertices[:, :3]

            triangles = vtk_to_numpy(mesh.GetPolys().GetConnectivityArray())
            triangles = np.reshape(triangles, (-1, 3))
            print(prefix, np.min(vertices[:, 2]), np.max(vertices[:, 2]))
            meshes[prefix] = (vertices, triangles)

        for slice_num in slice_numbers:
            img_slice = image[:, :, slice_num]
            plt.imshow(img_slice)

            for prefix, color in zip(file_prefix, ["r.", "g.", "b."]):
                vertices, triangles = meshes[prefix]
                tri_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
                plane_orig = np.array(tri_mesh.centroid)
                plane_orig[2] = slice_num
                plane_norm = (0, 0, 1)

                slice = tri_mesh.section(plane_origin=plane_orig, plane_normal=plane_norm)
                contour = np.array(slice.vertices)[:, :2]

                if mlab_plot:
                    import mayavi.mlab as mlab
                    mesh = meshcut.TriangleMesh(vertices, triangles)
                    plane = meshcut.Plane(plane_orig, plane_norm)
                    show(mesh, plane, 1)
                    mlab.show()
                plt.plot(contour[:, 1], contour[:, 0], color, markersize=1)
            filename = output_dir.joinpath("img", f"img_{slice_num:02d}_{idx:02d}.png")
            filename.parent.mkdir(parents=True, exist_ok=True)
            slice_filenames[slice_num].append(filename)
            plt.savefig(str(filename))
            plt.close()

    for slice_num in slice_numbers:
        with imageio.get_writer(output_dir.joinpath(f"slice_{slice_num}.gif"), mode="I") as writer:
            for file in slice_filenames[slice_num]:
                image = imageio.imread_v2(str(file))
                writer.append_data(image)
