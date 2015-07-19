In order to use this code, you will need the graph cuts code from Veksler and Delong (http://vision.csd.uwo.ca/code/). For the graph cuts code, look under "Multi-label optimization" for the gco-v3.0 zipped file.

Aspects of the code:
This code performs a series of steps in order to compute the depth map of the given scene. In order to accomplish this task, the user must provide a few input parameters. Default values are set in the code, but they can be overwritten using switches during the function call.

In order to compute the depth, the code sweeps a plane through the scene. At each depth d, a cost is computed for image patches having a depth of d. We have abstracted the physical depths from the code and use a "virtual camera" which is identical to the reference camera except that it has been translated away from the array. The further away the camera is, the higher the depth resolution that can be acheived, up to a limit. As the virtual camera moves farther away fromt the reference, subsequent steps in the plane sweep will be closer together; however, this improved resolution will eventually result in subpixel shifts when sampling the outlying image views. Thus, there will be a point of diminishing returns where further depth resolution cannot be achieved. Also, increasing the depth resolution will increase the program's run time. A good rule of thumb is to place the virtual camera outside of the camera array at roughly twice the baseline of the array (measured as the distance from the reference camera to the outlying camera that is the farthest away). Determining the sweep range of the system is a bit hit and miss. In the next update to the code I will include a utility to help visualize the sweep range so proper values can be used in the code.

Flexible patch size - All depth estimation codes are susceptible to false depth estimates in flat patches. To help avoid this, we implement a variable patch size and use graph cuts regularization to improve robustness. In order to find the correct patch size, we search through the reference image to find patches which have a sufficient number of "strong" gradients, starting at 8x8 patches and increasing to 16x16 and 32x32 patches. Larger patch sizes may cause memory issues, so be cautious if you tinker with the source code to increase the patch size. (N.B. The current implementation requires the patches to be sequential powers of 2.)

After using the variable patch sizes to compute the cross-channel cost, we further refine the estimate of the depth using graph cuts to enforce depth similarity for adjacent patches who exhibit similar intensities in the reference image.

Finally, using the refinded depth map, the input images are warped to the reference view and the images are saved. Image fusion occurs in a separate code to enable fine parameter tweaks without having to rerun the entire depth estimation step. The seperate function also abstracts away the fusion step which may differ for various applications.

Usage:
There are a few input parameters for the depthEstimation code which may be set at runtime using switches after calling the function using the command line.
The parameters are:

**imFN** - the image filename which should have an incrementing count starting from 0. It may also include the relative or full path to the images e.g.
  * camera_%d.png -> camera_0.png, camera_1.png, ..., camera_(N-1).png
  * cam_%02d.jpg -> cam_00.jpg, cam_01.jpg, ..., cam_(N-1).jpg
  * data/cam%d.png -> data/cam0.png, data/cam1.png, ..., data/cam(N-1).png

**outFN** - output filename for the warped images with an incrementing count starting from 0. As with imFN it may include a relative or full path e.g.
  * warped_%d.png -> warped_0.png, warped_1.png, ..., warped_(N-1).png
  * warp_%02d.jpg -> warp_00.jpg, warp_01.jpg, ..., warp_(N-1).jpg
  * data/warp%d.png -> data/warp0.png, data/warp1.png, ..., data/warped(N-1).png

**N** - The number of cameras in the array

**calibFN** - The name of the calibration function found using cameraCalibration (default is "calib.yml")

**refCam** - The camera used for reference, it should be the same as the reference camera in cameraCalibration

**vCamPos** - The position of the virutal camera relative to the the reference camera. It is placed horizontally away from the reference by vCamPos units. The units are determined by the calibration file parameters (typically millimeters). Typical values of vCamPos are twice the array baseline. For the RGBY and RGB+NIR cases the baseline for the array was ~30mm so a vCamPos of 50-60mm is good

**maxDisp** - Maximum displacement for the plane sweep, determining these values is a bit hit and miss. In the next update I will provide a utility to make this easier.

**minDisp** - Minimum displacement for the plane sweep, determining these values is a bit hit and miss. In the next update I will provide a utility to make this easier.

Example usage using RGBY scene 1:
For the sake of simplicity I will assume that the input images and calibration file have been placed in a folder in the same directory as the executable function

depthEstimation -imFN=scene1_images/camera_%02.png -outFN=scene1_images/warped_%02.png -calibFN=scene1_images/calib.yml -refCam=0 -vCamPos=50 -minDisp=50 -maxDisp=90