#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\calib3d\calib3d.hpp>
#include<stdio.h>
#include<omp.h>
#include<GCoptimization.h> // 3rd party software with GPL license, GCO v3 toolbox from http://vision.csd.uwo.ca/code/

using namespace cv;
using namespace std;

const char* keys =
{
	"{imFN|images/Camera_%0.2d.png|input images file name template}" // input file name template for images, may include relative or full path
	"{outFN|images/Warped_%0.2d.png|output file name}"
	"{N|4|number of cameras in the array}"
	"{calibFN|calib.yml|calibration file name}"
	"{minDisp|50|minimum Disparity}"
	"{maxDisp|80|maximum disparity}"
	"{refCam|0|Reference camera}"
	"{vCamPos|-50|Position of the virtual camera (should be negative)}"
	// input format for Opencv 2.4.x
	//"{1|imFN|images/Camera_%0.2d.png|input images file name template}" // input file name template for images, may include relative or full path
	//"{2|outFN|images/Warped_%0.2d.png|output file name}"
	//"{3|N|4|number of cameras in the array}"
	//"{4|calibFN|calib.yml|calibration file name}"
	//"{5|minDisp|40|minimum Disparity}"
	//"{6|maxDisp|100|maximum disparity}"
	//"{7|refCam|0|Reference camera}"
	//"{8|vCamPos|-50|Position of the virtual camera (should be negative)}"
};

bool parseCalib(string calibFN, vector<Mat> &K, vector<Mat> &d, vector<Mat> &R, vector<Mat> &T, vector<Mat> &P);
Mat im2colstep(Mat im, Size patchSize, Size stepSize);
Mat col2imstep(Mat in, Size imSize, Size patchSize, Size stepSize, bool doAvg = true);
void blurGradient(Mat im, Mat &gX, Mat &gY);
Mat findPatchSize(Mat im, Mat_<unsigned char> sizes, Size pShift);
Mat findCost(int tDisp, int minDisp, int refCam, int nPatches, vector<Mat> gX_vec, vector<Mat> gY_vec, vector<Mat> im_vec, Mat_<unsigned char> pSizes, Size pShift, vector<Mat> Rvec, vector<Mat> T, vector<Mat> K, Mat Q, vector<Mat> maskSize);
void runGC(Mat_<int> cost, Mat_<float> refIm, Mat_<int> &gcDepth, Size patchSz, Size patchShift, double alpha = 0.1, double beta = 0.005);
vector<Mat> warpIms(vector<Mat> ims, Mat_<int> dispMap, int minDisp, int tDisp, int refCam, vector<Mat> Rvec, vector<Mat> T, vector<Mat> K, Mat Q);

int main(int argc, char ** argv)
{
	// parse the inputs
	CommandLineParser parser(argc, argv, keys);
	string imFN = parser.get<string>("imFN");
	string outFN = parser.get<string>("outFN");
	string calibFN = parser.get<string>("calibFN");
	int nCams = parser.get<int>("N");
	int minDisp = parser.get<int>("minDisp");
	int maxDisp = parser.get<int>("maxDisp");
	int refCam = parser.get<int>("refCam");
	double vPos = parser.get<double>("vCamPos");

	if (vPos > 0) // position of the virtual camera should be negative
		vPos *= -1;

	// read the input images as grayscale images
	vector<Mat> im_vec;
	for (int i = 0; i < nCams; i++)
	{
		char imfn[200];
		sprintf(imfn, imFN.c_str(), i);
		Mat im = imread(imfn, IMREAD_GRAYSCALE);
		if (!im.data)
		{
			printf("Failed to read image %s\n", imfn);
			return -1;
		}
		// convert to single
		im.convertTo(im, CV_32FC1);
		im /= 255;

		im_vec.push_back(im);
	}

	Size imSz = im_vec[0].size();
	int tDisp = maxDisp - minDisp + 1; // number of disparities

	// smallest patch size and the shift between neighboring patches 
	// making pShift smaller will greatly increase memory usage!
	Size pSize(8, 8);
	Size pShift(4, 4);

	// parse the calibration file
	vector<Mat> K(nCams), R(nCams), T(nCams), d(nCams), P(nCams);
	bool readCalib = parseCalib(calibFN, K, d, R, T, P);
	if (!readCalib) // error message passed inside of parse calib
		return -1;

	// undistort the input images
	for (int i = 0; i < nCams; i++)
	{
		Mat tmp = im_vec[i].clone();
		undistort(tmp, im_vec[i], K[i], d[i]);
	}

	//// merge the input images into a single N channel Mat
	//Mat ims;
	//merge(im_vec, ims);

	// find the inverse mapping matrix Q
	Mat Q;
	{
		Mat_<double> ref_R = Mat_<double>::eye(3, 3), ref_T = Mat_<double>::zeros(3, 1);
		ref_T(0) = vPos;
		Mat d1 = Mat::zeros(1, 5, CV_64F);
		Mat j1, j2, j3, j4; // junk matrices
		stereoRectify(K[refCam], d1, K[refCam], d1, imSz, ref_R, ref_T, j1, j2, j3, j4, Q);
	}

	// determine the appropriate patch size for the image
	Mat_<unsigned char> pSizes = (Mat_<unsigned char>(3, 1) << 8, 16, 32);
	int numPSize = pSizes.total();

	Mat matchSize = findPatchSize(im_vec[refCam], pSizes, pShift);

	int numSizes = pSizes.total();

	// create the masks for the patch sizes
	vector<Mat> maskSize(numSizes);
	for (int i = 0; i < numSizes; i++)
		maskSize[i] = matchSize == pSizes(i);

	// set up the cost
	Mat depthmap_scale_factor;
	int nPatches;
	{
		Mat tmp = Mat::ones(imSz, CV_32FC1);
		tmp = im2colstep(tmp, pSize, pShift);
		nPatches = tmp.rows;
		depthmap_scale_factor = col2imstep(tmp, imSz, pSize, pShift, false);
	} // tmp goes out of scope

	// prep the data
	vector<Mat> gX_vec, gY_vec;
	vector<float> epsilonX(nCams), epsilonY(nCams);
	for (int i = 0; i < nCams; i++)
	{
		// find the gradients of the input images
		Mat tmpGx, tmpGy, tmpIm;
		blurGradient(im_vec[i], tmpGx, tmpGy);
		tmpGx = abs(tmpGx);
		tmpGy = abs(tmpGy);

		// find the average gradient strength for each input image
		Scalar meanX, meanY;
		meanX = cv::mean(tmpGx);
		meanY = cv::mean(tmpGy);

		epsilonX[i] = meanX[0] / 2;
		epsilonY[i] = meanY[0] / 2;

		// add the average gradient back to the directional gradients
		tmpGx += meanX[0] / 2;
		tmpGy += meanY[0] / 2;

		gX_vec.push_back(tmpGx);
		gY_vec.push_back(tmpGy);
	}

	// get the reference image gradients
	Mat ref_im_X, ref_im_Y;
	ref_im_X = gX_vec[refCam].clone();
	ref_im_Y = gY_vec[refCam].clone();

	// convert the rotation matrices to rotation vectors
	vector<Mat> Rvec(nCams);
	for (int i = 0; i < nCams; i++)
		Rodrigues(R[i], Rvec[i]);

	// Compute the cost term (data fidelity) using a plane sweep approach through the scene
	Mat_<float> costGxGy = findCost(tDisp, minDisp, refCam, nPatches, gX_vec, gY_vec, im_vec, pSizes, pShift, Rvec, T, K, Q, maskSize);

	// compute the naive depth map
	Mat_<float> depth(nPatches, 1);
	for (int i = 0; i < nPatches; i++)
	{
		double minVal, maxVal;
		Point minLoc, maxLoc;
		minMaxLoc(costGxGy.row(i), &minVal, &maxVal, &minLoc, &maxLoc);
		depth(i) = minLoc.x;
	}

	copyMakeBorder(depth, depth, 0, 0, 0, pSize.area() - 1, BORDER_REPLICATE);
	Mat depthIm = col2imstep(depth, imSz, pSize, pShift);

	string naiveDepth = "Naive depth map";
	namedWindow(naiveDepth, WINDOW_NORMAL);
	resizeWindow(naiveDepth, depthIm.cols / 2, depthIm.rows / 2);
	moveWindow(naiveDepth, 10, 10);
	imshow(naiveDepth, depthIm / (tDisp - 1));
	waitKey(1000);
	destroyAllWindows();

	// save the naive depth
	imwrite("depthMapNaive.png", depthIm / (tDisp - 1) * 255);

	printf("Running graph cuts\n");
	Mat_<int> gcCost, gcDepth;
	Mat_<float> gcRefIm = im_vec[refCam].clone(); // avoid the median filter
	costGxGy *= 100000; // scale the costs to avoid quantization error when converting to int
	costGxGy.convertTo(gcCost, CV_32SC1);
	runGC(gcCost, im_vec[refCam], gcDepth, pSize, pShift);

	gcDepth.convertTo(depth, CV_32FC1); 
	copyMakeBorder(depth, depth, 0, 0, 0, pSize.area() - 1, BORDER_REPLICATE);
	depthIm = col2imstep(depth, imSz, pSize, pShift); // overwrite the previous depthIm

	// show the depth map
	string gcDepthWin = "Graph cuts depth";
	namedWindow(gcDepthWin, WINDOW_NORMAL);
	resizeWindow(gcDepthWin, depthIm.cols / 2, depthIm.rows / 2);
	moveWindow(gcDepthWin, 10, 10);
	imshow(gcDepthWin, depthIm / (tDisp - 1));
	waitKey(1000);
	destroyAllWindows();

	// save the graph cuts image
	imwrite("depthMapGC.png", depthIm / (tDisp - 1) * 255);


	// use the depth map to warp iamges to the reference view
	vector<Mat> warpedImages;
	warpedImages = warpIms(im_vec, depthIm, minDisp, tDisp, refCam, Rvec, T, K, Q);


	// save the warped images
	for (int i = 0; i < nCams; i++)
	{
		char fileName[200];
		sprintf(fileName, outFN.c_str(), i);
		imwrite(fileName, warpedImages[i] * 255);
	}

	return 0;
}
