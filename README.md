# Introduction

Computational Cardiac Imaging Toolkit (CCITK) is a collection of python functions and applications for
cardiac image processing, developed by BioMedIA group and Computational Cardiac Imaging group. 

Besides utility packages, it contains following applications:
 * [`cmr_segment`](ccitk/cmr_segment/README.md) is a pipeline for processing cardiac MRI cine 
images, including segmentation, mesh extraction, landmark extraction, registration and motion tracking.
* [`ukbb`](ccitk/ukbb/README.md) contains applications to download UKBB DICOM images, convert them to NIFTY images, segment short-axis slices and 
long-axis slices, and analyze images and segmentations to extract metrics. 
* [`slurm`](ccitk/slurm/README.md) is an application to submit batch jobs 
SLURM managed BioMedia resources. 

# Dependencies
### packages
Core:
- `image`: nibabel, scipy

Optional (will be imported if dependencies exist):
- `mesh`: vtk

- `visual`: imageio, trimesh, vtk, meshcut, vedo

- `landmark`: vtk nibabel

- `segment`: torch

- `nn`: torch

- `refine`: SimpleITK, vtk, mirtk

- `motion`: mirtk

- `register`: mirtk, vtk

### Applications
Please reference the README in each application. 


# Installation

```
python setup.py install
```

Alternatively,
```
python setup.py develop
```

This create a link to python files in the directory, rather than copying them to the site packages directory. This
is convenient to develop code, as you don't need to do setup install every time you change your python code.

However, for command line commands, if the name of the command change, you still need to run the setup develop again. 

# Python packages

### create hyperlinks to sphinx docs

# Applications

[`ccitk-cmr-segment`](ccitk/cmr_segment/README.md)

[`ccitk-slurm-submit`](ccitk/slurm/README.md)

[`ccitk-ukbb-download`](ccitk/ukbb/README.md)

[`ccitk-ukbb-convert`](ccitk/ukbb/README.md)

[`ccitk-ukbb-segment`](ccitk/ukbb/README.md)

[`ccitk-ukbb-analyze`](ccitk/ukbb/README.md)


# Sphinx docs
```
pip install docutils==0.19
pip install sphinex==5.3.0
pip install myst-parser==0.18.1
pip install sphinx_rtd_theme

cd docs/
make html

make clean html
make html
```
