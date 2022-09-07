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
            "ccitk-ukbb-download = ccitk.core.ukbb.download:main",
            "ccitk-ukbb-convert = ccitk.core.ukbb.convert:main",
            "ccitk-ukbb-segment = ccitk.core.ukbb.segment:main",
            "ccitk-ukbb-analyze = ccitk.core.ukbb.analyze:main",
        ]
    },
    install_requires=[
        "numpy",
        "nibabel",
        "vtk",
        "SimpleITK",
        "pydicom",

    ],
)
