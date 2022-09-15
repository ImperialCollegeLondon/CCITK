import os
from pathlib import Path
import numpy as np
import trimesh
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import meshcut

import itertools
import numpy.linalg as la
import nibabel as nib
from ccitk.image import read_nii_image
from matplotlib import pyplot as plt
import imageio
# import mirtk
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--motion-dir", dest="motion_dir", type=str)
    parser.add_argument("--img-frame-dir", dest="img_frame_dir", type=str)
    parser.add_argument("--lv-rigid-dof", dest="lv_rigid_dof_path", type=str)
    parser.add_argument("--rv-rigid-dof", dest="rv_rigid_dof_path", type=str)
    parser.add_argument("--slice-nums", dest="slice_nums", nargs='+')
    parser.add_argument("--output-dir", dest="output_dir", type=str)
    return parser.parse_args()


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


def main_2():
    # vtk_path = Path("Z:\\UKBB_40616\\4D_Segmented_2.0_to_do_motion\\1035470\\vtks\\LVendo_ED.vtk")
    # motion_vtk_dir = Path("Z:\\UKBB_40616\\UKBB_motion_analysis\\results\\UKBB_01_2022\\1035470\\VTK\\LV_endo")
    # seg_nii_path = Path("Z:\\UKBB_40616\\4D_Segmented_2.0_to_do_motion\\1035470\\seg_lvsa_SR_ED.nii.gz")
    # img_nii_dir = Path("Z:\\UKBB_40616\\UKBB_motion_analysis\\results\\UKBB_01_2022\\1035470\\img_frame")
    # rigid_dof_path = Path(
    #     "Z:\\UKBB_40616\\UKBB_motion_analysis\\results\\UKBB_01_2022\\1035470\\dofs\\ENDO_srreg.dof.gz")
    #

    motion_vtk_dir = Path("/mnt/storage/home/suruli/suruli/sheffield/results_contrast_enlarge/phd001/motion/VTK")
    img_nii_dir = Path("/mnt/storage/home/suruli/suruli/sheffield/results_contrast_enlarge/phd001/enlarged")
    lv_rigid_dof_path = Path("/mnt/storage/home/suruli/suruli/sheffield/results/phd001/results_contrast_enlarge/lv/endo/temp/dof/srreg.dof.gz")
    rv_rigid_dof_path = Path("/mnt/storage/home/suruli/suruli/sheffield/results/phd001/results_contrast_enlarge/rv/temp/dof/srreg.dof.gz")

    # #
    output_dir = Path(__file__).parent.joinpath("output_motion_sheffield")
    slice_nums = [40, 50, 60]
    visual = False
    # args = parse_args()
    # motion_vtk_dir = Path(args.motion_dir)
    # img_nii_dir = Path(args.img_frame_dir)
    # lv_rigid_dof_path = Path(args.lv_rigid_dof_path)
    # rv_rigid_dof_path = Path(args.rv_rigid_dof_path)
    # output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # slice_nums = args.slice_nums
    # slice_nums = [40]

    slice_filenames = {}
    for slice_num in slice_nums:
        slice_filenames[slice_num] = []

    for idx, vtk_name in enumerate(os.listdir(str(motion_vtk_dir.joinpath("LV_endo")))):
        print(vtk_name)
        img_nii_path = img_nii_dir.joinpath("lvsa_{:02d}.nii.gz".format(idx))
        image, affine = read_nii_image(img_nii_path)

        affine = np.linalg.inv(affine)
        meshes = {}
        for prefix, file_prefix, color in zip(["LV_endo", "LV_epi", "RV"], ["LV_endo", "LV_epi", "RV"], ["r.", "g.", "b."]):
            vtk_path = motion_vtk_dir.joinpath(prefix, f"fr{idx:02d}.vtk")
            # vtk_path = Path("Z:\\UKBB_40616\\4D_Segmented_2.0_to_do_motion\\1035470\\vtks\\LVendo_ED.vtk")

            # temp_vtk_path = output_dir.joinpath("vtk", f"vtk_{prefix}_{idx}.vtk")
            # temp_vtk_path.parent.mkdir(parents=True, exist_ok=True)
            # if prefix == "RV":
            #     mirtk.transform_points(
            #         str(vtk_path),
            #         str(temp_vtk_path),
            #         invert=None,
            #         dofin=str(rv_rigid_dof_path)
            #     )
            # else:
            #     mirtk.transform_points(
            #         str(vtk_path),
            #         str(temp_vtk_path),
            #         invert=None,
            #         dofin=str(lv_rigid_dof_path)
            #     )

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
            print(file_prefix, np.min(vertices[:, 2]), np.max(vertices[:, 2]))
            meshes[file_prefix] = (vertices, triangles)

        for slice_num in slice_nums:
            img_slice = image[:, :, slice_num]
            plt.imshow(img_slice)

            for prefix, color in zip(["LV_endo", "LV_epi", "RV"], ["r.", "g.", "b."]):
                vertices, triangles = meshes[prefix]
                tri_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
                plane_orig = np.array(tri_mesh.centroid)
                plane_orig[2] = slice_num
                plane_norm = (0, 0, 1)

                slice = tri_mesh.section(plane_origin=plane_orig, plane_normal=plane_norm)
                contour = np.array(slice.vertices)[:, :2]

                if visual:
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

    for slice_num in slice_nums:
        with imageio.get_writer(output_dir.joinpath(f"slice_{slice_num}.gif"), mode="I") as writer:
            for file in slice_filenames[slice_num]:
                image = imageio.imread_v2(str(file))
                writer.append_data(image)


def main():
    # vtk_path = Path("Z:\\UKBB_40616\\4D_Segmented_2.0_to_do_motion\\1035470\\vtks\\LVendo_ED.vtk")
    # motion_vtk_dir = Path("Z:\\UKBB_40616\\UKBB_motion_analysis\\results\\UKBB_01_2022\\1035470\\VTK\\LV_endo")
    # seg_nii_path = Path("Z:\\UKBB_40616\\4D_Segmented_2.0_to_do_motion\\1035470\\seg_lvsa_SR_ED.nii.gz")
    # img_nii_dir = Path("Z:\\UKBB_40616\\UKBB_motion_analysis\\results\\UKBB_01_2022\\1035470\\img_frame")
    # rigid_dof_path = Path(
    #     "Z:\\UKBB_40616\\UKBB_motion_analysis\\results\\UKBB_01_2022\\1035470\\dofs\\ENDO_srreg.dof.gz")
    #
    motion_vtk_dir = Path("/mnt/storage/home/suruli/cardiac/UKBB_40616/UKBB_motion_analysis/results/UKBB_01_2022/1035470/VTK")
    img_nii_dir = Path("/mnt/storage/home/suruli/cardiac/UKBB_40616/UKBB_motion_analysis/results/UKBB_01_2022/1035470/img_frame")
    lv_rigid_dof_path = Path("/mnt/storage/home/suruli/cardiac/UKBB_40616/UKBB_motion_analysis/results/UKBB_01_2022/1035470/dofs/ENDO_srreg.dof.gz")
    rv_rigid_dof_path = Path("/mnt/storage/home/suruli/cardiac/UKBB_40616/UKBB_motion_analysis/results/UKBB_01_2022/1035470/dofs/RV_srreg.dof.gz")

    # #
    output_dir = Path(__file__).parent.joinpath("output_motion")
    slice_nums = [30, 40, 50]
    visual = False
    # args = parse_args()
    # motion_vtk_dir = Path(args.motion_dir)
    # img_nii_dir = Path(args.img_frame_dir)
    # lv_rigid_dof_path = Path(args.lv_rigid_dof_path)
    # rv_rigid_dof_path = Path(args.rv_rigid_dof_path)
    # output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # slice_nums = args.slice_nums
    # slice_nums = [40]

    slice_filenames = {}
    for slice_num in slice_nums:
        slice_filenames[slice_num] = []

    for idx, vtk_name in enumerate(os.listdir(str(motion_vtk_dir.joinpath("LV_endo")))):
        img_nii_path = img_nii_dir.joinpath("lvsa_img_fr{:02d}.nii.gz".format(idx))
        image, affine = read_nii_image(img_nii_path)

        affine = np.linalg.inv(affine)
        meshes = {}
        for prefix, file_prefix, color in zip(["LV_endo", "LV_epi", "RV"], ["LVendo", "LVepi", "RV"], ["r.", "g.", "b."]):
            vtk_path = motion_vtk_dir.joinpath(prefix, file_prefix + f"_fr{idx:02d}.vtk")
            # vtk_path = Path("Z:\\UKBB_40616\\4D_Segmented_2.0_to_do_motion\\1035470\\vtks\\LVendo_ED.vtk")

            temp_vtk_path = output_dir.joinpath("vtk", f"vtk_{prefix}_{idx}.vtk")
            temp_vtk_path.parent.mkdir(parents=True, exist_ok=True)
            if prefix == "RV":
                mirtk.transform_points(
                    str(vtk_path),
                    str(temp_vtk_path),
                    invert=None,
                    dofin=str(rv_rigid_dof_path)
                )
            else:
                mirtk.transform_points(
                    str(vtk_path),
                    str(temp_vtk_path),
                    invert=None,
                    dofin=str(lv_rigid_dof_path)
                )

            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(str(temp_vtk_path))
            reader.Update()
            mesh = reader.GetOutput()

            vertices = vtk_to_numpy(mesh.GetPoints().GetData())
            vertices = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
            vertices = np.matmul(affine, np.transpose(vertices))
            vertices = np.transpose(vertices)
            vertices = vertices[:, :3]

            triangles = vtk_to_numpy(mesh.GetPolys().GetConnectivityArray())
            triangles = np.reshape(triangles, (-1, 3))
            print(np.min(vertices[:, 2]), np.max(vertices[:, 2]))
            meshes[prefix] = (vertices, triangles)

        for slice_num in slice_nums:
            img_slice = image[:, :, slice_num]
            plt.imshow(img_slice)

            for prefix, color in zip(["LV_endo", "LV_epi", "RV"], ["r.", "g.", "b."]):
                vertices, triangles = meshes[prefix]
                tri_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
                plane_orig = np.array(tri_mesh.centroid)
                plane_orig[2] = slice_num
                plane_norm = (0, 0, 1)

                slice = tri_mesh.section(plane_origin=plane_orig, plane_normal=plane_norm)
                contour = np.array(slice.vertices)[:, :2]

                if visual:
                    import mayavi.mlab as mlab
                    mesh = meshcut.TriangleMesh(vertices, triangles)
                    plane = meshcut.Plane(plane_orig, plane_norm)
                    show(mesh, plane, 1)
                    mlab.show()
                plt.plot(contour[:, 1], contour[:, 0], color, markersize=3)
            filename = output_dir.joinpath("img", f"img_{slice_num:02d}_{idx:02d}.png")
            filename.parent.mkdir(parents=True, exist_ok=True)
            slice_filenames[slice_num].append(filename)
            plt.savefig(str(filename))
            plt.close()

    for slice_num in slice_nums:
        with imageio.get_writer(output_dir.joinpath(f"slice_{slice_num}.gif"), mode="I") as writer:
            for file in slice_filenames[slice_num]:
                image = imageio.imread_v2(str(file))
                writer.append_data(image)


if __name__ == '__main__':
    # main()
    main_2()
