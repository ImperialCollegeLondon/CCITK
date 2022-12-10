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
            "ccitk-ukbb-batch-download = ccitk.ukbb.download.batch:main",
            "ccitk-ukbb-batch-convert = ccitk.ukbb.convert.batch:main",
            "ccitk-ukbb-segment = ccitk.ukbb.segment.cli:main",
            "ccitk-ukbb-batch-segment = ccitk.ukbb.segment.batch:main",
            "ccitk-ukbb-analyze = ccitk.ukbb.analyze.cli:main",
            "ccitk-ukbb-batch-analyze = ccitk.ukbb.analyze.batch:main",
            "ccitk-cmr-segment = ccitk.cmr_segment.cli:main",
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
