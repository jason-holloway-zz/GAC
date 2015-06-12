#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\\imgproc\imgproc.hpp>
#include <stdio.h>

using namespace cv;
using namespace std;

const char* keys =
{
	"{imFN|images/Warped_%0.2d.png|input images file name template}" // input file name template for images, may include relative or full path
	"{outFN|images/fusedImage.png|output file name}"
	"{N|4|number of cameras in the array}"
	"{alpha|.5|weighting for NIR fusion}"
	// input format for Opencv 2.4.x
	//"{1|imFN|images/Warped_%0.2d.png|input images file name template}" // input file name template for images, may include relative or full path
	//"{2|outFN|images/fusedImage.png|output file name}"
	//"{3|N|4|number of cameras in the array}"
	//"{4|alpha|.5|weighting for NIR fusion}"
};

int main(int argc, char ** argv)
{
	CommandLineParser parser(argc, argv, keys);
	string imFN = parser.get<string>("imFN");
	string outFN = parser.get<string>("outFN");
	int nCams = parser.get<int>("N");
	float alpha = parser.get<float>("alpha");

	// if the number of cameras is 4, then we do RGBY
	// if the number of cameras is 5, then we also do NIR fusion
	// if the number of cameras is anything else, quit
	if (nCams < 4 || nCams > 5)
	{
		printf("The number of input images must be either 4 or 5\n");
		return -1;
	}

	// the image order is fixed:
	// 0 - Y
	// 1 - R
	// 2 - G
	// 3 - B
	// 4 - NIR (optional)

	// read the input images as grayscale images
	vector<Mat_<float>> ims;
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

		ims.push_back(im);
	}

	
	// put pointers to the middle three images in a separate vector
	vector<Mat_<float>> rgb;
	for (int i = 1; i < 4; i++)
		rgb.push_back(ims[i]);

	// merge into a single color image
	Mat_<Vec3f> colorIm;
	merge(rgb, colorIm);

	// convert the color image to YCrCb
	Mat_<Vec3f> ycrcb;
	cvtColor(colorIm, ycrcb, CV_RGB2YCrCb);

	// extract the second and third channels from the YCrCb image after blurring slightly
	GaussianBlur(ycrcb, ycrcb, Size(3, 3), 0.5, 0.0);
	vector<Mat_<float>> fusedChan(3);

	extractChannel(ycrcb, fusedChan[1], 1);
	extractChannel(ycrcb, fusedChan[2], 2);

	if (nCams == 4)
		fusedChan[0] = ims[0].clone();
	else
	{
		// use bilateral filtering to remove high frequency details from the input image and replace with the NIR image
		Mat_<float> imBase, imDetail;
		bilateralFilter(ims[0], imBase, -1, .1, 16);
		imDetail = ims[0] - imBase;

		Mat_<float> nirBase, nirDetail;
		bilateralFilter(ims[4], nirBase, -1, .1, 16);
		nirDetail = ims[4] - nirBase;

		fusedChan[0] = imBase + alpha * nirDetail + (1 - alpha)*imDetail;
		
		// threshold the output to the range [0,1]
		threshold(fusedChan[0], fusedChan[0], 0, 1, THRESH_TOZERO);
		threshold(fusedChan[0], fusedChan[0], 1, 1, THRESH_TRUNC);
	}

	// merge into the output image
	Mat_<Vec3f> fusedIm;
	merge(fusedChan, fusedIm);
	
	// convert to BGR
	cvtColor(fusedIm, fusedIm, CV_YCrCb2BGR);

	imwrite(outFN, fusedIm * 255);

	return 0;
}