## Introduction

Computational Cardiac Imaging Toolkit (CCITK) is a collection of python functions and command line tools for
cardiac image processing, developed by BioMedIA group and Computational Cardiac Imaging group. 

Besides utility functions, it contains two applications. `cmr_segment` is a pipeline for processing cardiac MRI cine 
images, including segmentation, mesh extraction, landmark extraction, registration and motion tracking.
`ukbb` contains applications to download UKBB DICOM images, convert them to NIFTY images, segment short-axis slices and 
long-axis slices, and analyze images and segmentations to extract metrics. 


## Installation

```
python setup.py install
```

## Python modules

### create hyperlinks to sphinx docs
image

mesh

landmark

segment

refine

motion

register


## Command line tools

### [`ccitk-cmr-segment`](ccitk/cmr_segment/README.md)



ccitk-ukbb-download

ccitk-ukbb-convert

ccitk-ukbb-segment

ccitk-ukbb-analyze
