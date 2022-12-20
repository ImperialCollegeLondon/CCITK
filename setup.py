from setuptools import setup, find_namespace_packages


setup(
    name="ccitk",
    use_scm_version=True,
    author="Surui Li",
    author_email="surui.li@imperial.ac.uk",
    description="Computational Cardiac Imaging Toolkit",
    package_dir={"": "."},
    packages=find_namespace_packages(),
    setup_requires=["setuptools >= 40.0.0"],
    package_data={"": ["*.conf", "*.txt"]},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "ccitk-ukbb-download = ccitk.ukbb.download.cli:main",
            "ccitk-ukbb-convert = ccitk.ukbb.convert.cli:main",
            "ccitk-ukbb-segment = ccitk.ukbb.segment.cli:main",
            "ccitk-ukbb-analyze = ccitk.ukbb.analyze.cli:main",

            "ccitk-cmr-segment = ccitk.cmr_segment.cli:main",
            "ccitk-slurm-submit = ccitk.slurm.cli:main"
        ]
    },
    # install_requires=[
    #     "numpy",
    #     "nibabel",
    #     "vtk",
    #     "SimpleITK",
    #     "pydicom",
    #
    # ],
)
