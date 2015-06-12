#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <stdio.h>

using namespace std;
using namespace cv;

const char* keys =
{
	// input format for OpenCV 3.0+
	"{imFN|data/camera%0.2d_image%0.2d.png|input images file name template}" // input file name template for images, may include relative or full path
	"{nCams|4|number of cameras in the array}"
	"{nIms|10|number of images captured by each camera}"
	"{calibFN|calib.yml|calibration file name}" // may include a full or relative path
	"{refCam|0|Reference camera}"
	"{cornersPerRow|10|Number of corners along each row of the checkerboard}"
	"{cornersPerCol|8|Number of corners along each column of the checkerboard}"
	"{squareSize|0|size of the square (typically in mm, but can be any unit)}"
	"{squareSizeX|0|length of the square in the X direction}"
	"{squareSizeY|0|length of the square in the Y direction}"
	"{verbose|1|verbose switch to print text to the screen}"
	// input format for OpenCV 2.4.x
	//"{1|imFN|camera%0.2d_image%0.2d.png|input images file name template}" // input file name template for images, may include partial or full path
	//"{2|nCams|4|number of cameras in the array}"
	//"{3|nIms|10|number of images captured by each camera}"
	//"{4|calibFN|calib.yml|calibration file name}"
	//"{5|refCam|0|Reference camera}"
	//"{6|cornersPerRow|10|Number of corners along each row of the checkerboard}"
	//"{7|cornersPerCol|8|Number of corners along each column of the checkerboard}"
	//"{8|squareSize|0|size of the square (typically in mm, but can be any unit)}"
	//"{9|squareSizeX|0|length of the square in the X direction}"
	//"{10|squareSizeY|0|length of the square in the Y direction}"
	//"{11|verbose|1|verbose switch to print text to the screen}"
};

void readme();

int main(int argc, char ** argv)
{
	if (argc < 2)
	{
		readme();
		return -1;
	}

	CommandLineParser parser(argc, argv, keys);
	string imFN = parser.get<string>("imFN");
	string calibFN = parser.get<string>("calibFN");
	int nCams = parser.get<int>("nCams");
	int nIms = parser.get<int>("nIms");
	int refCam = parser.get<int>("refCam");
	int meshRow = parser.get<int>("cornersPerRow");
	int meshCol = parser.get<int>("cornersPerCol");
	bool verbose = (parser.get<int>("verbose") > 0);
	Size mesh(meshRow, meshCol);
	//Size squareSize;

	// determine the square size
	float sqSzX = parser.get<float>("squareSizeX");
	float sqSzY = parser.get<float>("squareSizeY");
	{
		float sqSz = parser.get<float>("squareSize");

		if (sqSz == 0.0 && sqSzY == 0.0 && sqSzX == 0.0)
		{
			printf("You must specify the size of the square\n");
			return -1;
		}


		if (sqSzY == 0.0 && sqSzX == 0.0)
		{
			sqSzX = sqSz;
			sqSzY = sqSz;
		}
		else if (sqSz != 0.0)
		{
			if (sqSzX == 0.0)
				sqSzY = sqSz;

			if (sqSzY == 0.0)
				sqSzY = sqSz;
		}
		//squareSize = Size(sqSzX, sqSzY);
	}

	// populate the 3D chessboard corners using the square size
	vector<Point3f> pattern;
	for (int j = 0; j < mesh.height; j++)
		for (int i = 0; i < mesh.width; i++)
			pattern.push_back(Point3f(i*sqSzX, j*sqSzY, 0));

	vector<vector<Point3f>> obPts;
	
	// read in the images
	vector<vector<vector<Point2f>>> points(nCams);
	int goodViews = 0;
	Size imSz;

	// find the checkerboard corners for each of the images
	if (verbose)
		printf("Finding chessboard corners\n");
	for (int i = 0; i < nIms; i++)
	{
		if (verbose)
			printf("Image %02d\n", i);

		vector<vector<Point2f>> corners(nCams);
		bool completed = false;
		for (int j = 0; j < nCams; j++)
		{
			Mat im;
			bool foundCorners;
			char imFileName[200];
			sprintf(imFileName, imFN.c_str(), j, i);

			im = imread(imFileName, IMREAD_COLOR);
			if (!im.data)
			{
				printf("Failed to read in image %s. Skipping this view\n", imFileName);
				break;
			}

			if (i == 0 && j == 0) // this could break if the first image can't be loaded
				imSz = im.size();

			foundCorners = findChessboardCorners(im, mesh, corners.at(j), CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);

			// if the corners aren't found in one of the cameras, abandon the view
			if (!foundCorners)
				break;
		
			if (j == nCams - 1)
				completed = true;
		}

		if (completed) // found corners in all of the views
		{
			goodViews++;
			for (int j = 0; j < nCams; j++)
				points[j].push_back(corners[j]);
			obPts.push_back(pattern);
		}
	}

	// make sure there are enough good views
	if (goodViews < 5)
	{
		printf("There should be at least 5 views where all of the cameras can see the checkerboard. Quitting.\n");
		return -1;
	}

	if (verbose)
		printf("Using %d views to calibrate cameras\n\n", goodViews);

	// for each camera first perform internal calibration
	vector<Mat> K(nCams), R(nCams), T(nCams), d(nCams);
	R[refCam] = Mat::eye(3, 3, CV_64F);
	T[refCam] = Mat::zeros(3, 1, CV_64F);
	Mat_<double> errSingle = Mat_<double>::ones(nCams, 1) + 100;
	Mat_<double> errStereo = Mat_<double>::ones(nCams, 1) + 100;
	
	{
		vector<Mat> r, t; // let r and t run out of scope
		for (int j = 0; j < nCams; j++)
			errSingle(j) = calibrateCamera(obPts, points[j], imSz, K[j], d[j], r, t);
	}

	if (verbose) // show the error
	{
		for (int j = 0; j < nCams; j++)
			printf("Internal calibration error for camera %d is %0.4f\n", j, errSingle(j));
		printf("\n"); // space out the error text
	}


	for (int j = 0; j < nCams; j++)
	{
		Mat E, F; // won't be using these, let them run out of scope
		if (j == refCam)
		{
			errStereo(j) = 0;
			continue;
		}
		errStereo(j) = stereoCalibrate(obPts, points[refCam], points[j], K[refCam], d[refCam], K[j], d[j], imSz, R[j], T[j], E, F, CV_CALIB_USE_INTRINSIC_GUESS + CV_CALIB_FIX_INTRINSIC);
		
		/*~~~~~ comment the above line and uncomment the following line if using OpenCV 2.4.x ~~~~~~~~~~*/
		//errStereo(J) = stereoCalibrate(obPts, points[refCam], imPts[i], K[refCam], d[refCam], K[i], d[i], imSz, R[i], T[i], E, F, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 1e-6), CV_CALIB_USE_INTRINSIC_GUESS + CV_CALIB_FIX_INTRINSIC);
	}

	if (verbose) // show the error
		for (int j = 0; j < nCams; j++)
			printf("External calibration error for camera %d is %0.4f\n", j, errStereo(j));

	FileStorage fs(calibFN, FileStorage::WRITE);
	fs << "K" << K;
	fs << "d" << d;
	fs << "R" << R;
	fs << "T" << T;
	fs << "errSingle" << errSingle;
	fs << "errStereo" << errStereo;
	fs.release();

	return 0;
}

void readme()
{
	printf("This code calibrates cameras in a camera array\n");
	printf("For simplicity, this code assumes that the checkerboard is visible in each camera in each image\n");
	printf("If the checkerboard is not detected for a camera in a given view, all images from that view are discarded\n\n");
	printf("Example usage is:\n");
	printf("cameraCalibration -imFN=calib%%02d_image%%02d.png -outFN=calib.yml -N=4 -nIms=10 -cornersPerRow=10 -cornersPerColumn=8 -squareSize=30.0 -verbose=1\n\n");
	printf("outFN must end in \"xml\" or \"yml\".\nN is the number of cameras in the array.\nnIms is the number of checkerboard images\n");
	printf("cornersPerRow/cornersPerCol are the number of checkers in the pattern\n");
	printf("squareSize is the size of the checkers.\n  (If the checkers are not square use the -squareSizeX and -squareSizeY switches)\n");
	printf("verbose controls whether text is output to the screen, default is 1 (true)\n");
	printf("imFN must be a file name template which counts up from 0 both for the camera index and image index\n");
	printf("e.g. if \"-imFN=camera%%02_image%%02.png\" the program would load:\n");
	printf("camera00_image00.png, camera00_image01.png, ...\n");
	printf("camera01_image00.png, camera01_image01.png, ...\n");
	printf("camera02_image00.png, camera02_image01.png, ...\n");
	printf("...\n");
	printf("camera(N-1)_image00.png, camera(N-1)_image01.png, ...\n\n");
	printf("Typically 10-15 images are sufficient to calibrate cameras.\nThis code requires at least 5 valid views.\n");
	printf("For the best results, try to fill the FOV of the array (intersection of array elements' FOV)");
}