### GAC
Code to implement Generalized Assorted Cameras (see [my webpage](http://jrholloway.com/Projects/GAC) for more information.

Implementing GAC requires many steps, which I have split up into three parts:
1) Calibration
2) Depth Estimation and Warping
3) Image fusion

C++ code is provided for each of these tasks for RGBY and RGBY+NIR imaging. The hyperspectral fusion/recovery code will be provided as MATLAB code once it is cleaned up.

The codes in this repository rely on OpenCV (2.4.x or 3.0.x) and the [graph cuts implementation from Veksler and Delong](http://vision.csd.uwo.ca/code/). For the graph cuts code, look under "Multi-label optimization" for the gco-v3.0 zipped file. The codes are presented for OpenCV 3.0.x, though only a few lines need to be commented/uncommented for backwards compatiblity with 2.4.x. (These lines are marked in the files, mostly at the top EXCEPT for stereo calibrate at the bottom of the Calibration code.)

Windows users: Building OpenCV is not required to use this code. Everything should run if you install OpenCV from the executable.

The three steps for the GAC implementation are intended to be independet programs to help keep data input to a minimum for a single program call, as well as to minimize redundant computation if you decide to tweak a parameter. Source code for each of the steps is kept in a separate folder in the repository.

### Data
Data from the paper is available in the data folder. Data for the hyperspectral dataset is not currently available, though it will be shortly. Though processing for this data will only be provided in MATLAB code.
Calibration files have been computed using the codes in this repository. This means that external camera positions have not been refined using bundle adjustment. 