In this code, we calibrate the cameras used in the array. There is a small difference between this calibration and the one mentioned in the paper, sparse bundle adjustment is not used to jointly refine the calibration parameters. This is to avoid additional dependencies in using the code, though you may of course implement bundle adjustment to improve accuracy. Without BA we observe backprojection error of <1 pixel in all configurations of our array.

Aspects of the code:
This code computes the intrinsic and extrinsic calibration parameters for the cameras in the array. For simplicity the code assumes that the entire checkerboard (plus a small border) is visible in each image. If the checkerboard cannot be found in one of the input images, that view is skipped entirely. At least 5 valid views are required for camera calibration and 10-15 are preferred. Also when capturing images it is preferable to try to fill the field of view of the array (the intersection of the FOV of each camera) as much as possible.

Intrinsic parameters are found for each of the array elements, and then extrinisic parameters between the reference camera view and each outlying view are computed in a pairwise manner. As mentioned above, joint refinement using bundle adjustment is not included in this code. In general, it is not required for the reference camera to coincide with an array element, though for the sake of simplicity this code assumes that one of the cameras is the reference viewpoint. NOTE: If you are planning to use the imageFusion code to create RGBY or RGB+NIR images, you will see the best results if you choose the Y camera to be the reference.

Potential pitfalls:
If you find that in the course of calibrating the cameras the distortion is incorrectly modelled, it is probably because the checkerboard is not getting close enough to the edge of the image for that particular camera. If, due to camera arrangement, you cannot reach the image edges it will be neccessesary to perform intrinsic camera calibration for each camera independently. Then use the accurate distortion and K parameters find extrinsic parameters. THIS CANNOT BE ACHEIVED WITH THIS CODE. You will need to write new calibration code to account for this, though it should be straightforward.

Usage:
There are a few input parameters for the cameraCalibration code which may be set at runtime using switches after calling the function using the command line.
The parameters are:
imFN - the image filename which should have TWO incrementing counts, both starting from 0. The first is for the camera index, the second is the image index. It may also include the relative or full path to the images e.g.
	camera%02_image%02.png -> 
	camera00_image00.png, camera00_image01.png, ..., camera00_image(M-1).png
	camera01_image00.png, camera01_image01.png, ..., camera01_image(M-1).png
	.
	.
	.
	camera(N-1)_image00.png, camera(N-1)_image01.png, ..., camera(N-1)_image(M-1).png

calibFN - the location to save the calibration file, it may include a relative or full path, e.g.
	calib.yml
	data/calibrationFile.yml

N - The number of cameras in the array

calibFN - The name of the calibration function found using cameraCalibration, it must end in "xml" or "yml" (default is "calib.yml")

refCam - The camera used for reference, it should be the same as the reference camera in cameraCalibration

cornersPerRow - the number of interior corners in each row. The provided images have 10 interior corners per row (default 10)

cornersPerCol - the number of interior corners in each column. The provided images have 8 interior corners per column (default 8)

squareSize - The size of the square in units of your choosing (typically mm). The provided images have squares that are 30.0 mm (default 0) AT LEAST ONE SIZE MUST BE SPECIFIED

squareSizeX - If the checkerboard isn't square, you can specify the horizontal size. (default 0)

squareSizeY - If the checkerboard isn't square, you can specify the vertical size. (default 0)

verbose - print to the screen or not, values > 0 print to the screen, values <= 0 do not. (default 1)

Example usage using RGBY scene 1:
For the sake of simplicity I will assume that the calibration images have been placed in a folder in the same directory as the executable function.
Not all switches must be activated, and they can be in any order

cameraCalibration -imFN=scene1_calibration/camera%02_%02.png -calibFN=scene1_calibration/calib.yml -refCam=0 -N=4 -cornersPerRow=10 -cornersPerCol=8 -squareSize=30.0

Bare minimum when using provided calibration images for RGBY:
cameraCalibration -imFN=folder/camera%02_%02.png -calibFN=folder/calib.yml -squareSize=30.0

Bare minimum when using provided calibration images for RGBY+NIR:
cameraCalibration -imFN=folder/camera%02_%02.png -calibFN=folder/calib.yml -squareSize=30.0 -N=5