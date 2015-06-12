#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\calib3d\calib3d.hpp>
#include<omp.h>
#include<stdio.h>
#include<GCoptimization.h>

using namespace cv;
using namespace std;

bool parseCalib(string calibFN, vector<Mat> &K, vector<Mat> &d, vector<Mat> &R, vector<Mat> &T, vector<Mat> &P)
{
	int nCams = K.size();
	FileStorage fs(calibFN, FileStorage::READ);

	fs["K"] >> K;
	fs["d"] >> d;
	fs["R"] >> R;
	fs["T"] >> T;

	fs.release();

	// find the projection matrices
	for (int i = 0; i < nCams; i++)
	{
		Mat tmp;
		hconcat(R[i], T[i], tmp);
		P[i] = K[i] * tmp;
	}

	return true;
}

/*
im2colstep and col2imstep have been adapted from mex files written by Ron Rubinstein at Technion for his KSVD-Box software
*/
Mat im2colstep(Mat im, Size patchSize, Size stepSize)
{
	if (im.depth() != CV_32F)
	{
		printf("Input must be a float (32F)\n");
		Exception e;
		error(e);
		return im;
	}

	int totalPatches = ((im.cols - patchSize.width) / stepSize.width + 1)*((im.rows - patchSize.height) / stepSize.height + 1);
	int patchLen = patchSize.area()*im.channels();

	Mat out(totalPatches, patchLen, CV_32FC1); // input and outputs are floats
	int p0 = im.channels(), p1 = patchSize.width, p2 = patchSize.height;
	int s0 = im.channels(), s1 = stepSize.width, s2 = stepSize.height;
	int i0 = im.channels(), i1 = im.cols, i2 = im.rows;
	float * imData = (float *)im.data;
	float * outData = (float *)out.data;

	// extract patches
	int blocknum = 0;
	for (int k = 0; k <= i2 - p2; k += s2)
	{
		for (int j = 0; j <= i1 - p1; j += s1)
		{
			for (int i = 0; i <= i0 - p0; i += s0)
			{
				// copy a block
				for (int m = 0; m < p2; m++)
				{
					for (int n = 0; n < p1; n++)
					{
						memcpy(outData + blocknum*p0*p1*p2 + m*p0*p1 + n*p0, imData + (k + m)*i0*i1 + (j + n)*i0 + i, i0*sizeof(float));
						//printf("%04d %04d\n", blocknum*p0*p1*p2 + m*p0*p1 + n*p0, (k + m)*i0*i1 + (j + n)*i0 + i);
					}
				}

				blocknum++;
			}
		}
	}

	return out;
}

Mat col2imstep(Mat in, Size imSize, Size patchSize, Size stepSize, bool doAvg)
{
	int chan = in.cols / patchSize.area();
	int i0 = chan, i1 = imSize.width, i2 = imSize.height;
	int p0 = chan, p1 = patchSize.width, p2 = patchSize.height;
	int s0 = chan, s1 = stepSize.width, s2 = stepSize.height;

	// Output matrix has to be non-integer to avoid saturation, will convert back later if necessary
	Mat out = Mat::zeros(imSize, CV_32FC(chan));

	// if we are computing the average
	Mat scale;
	Mat onesPatch;
	if (doAvg)
	{
		scale = Mat::zeros(imSize, out.type());
		onesPatch = Mat::ones(1, 1, CV_32FC1);
	}

	float * iData = (float *)in.data;
	float * oData = (float *)out.data;
	float * sData = (float *)scale.data;
	float * nData = (float *)onesPatch.data;

	int blocknum = 0;

	for (int k = 0; k <= i2 - p2; k += s2)
	{
		for (int j = 0; j <= i1 - p1; j += s1)
		{
			for (int i = 0; i <= i0 - p0; i += s0)
			{
				// add back a single patch
				for (int m = 0; m < p2; m++)
				{
					for (int n = 0; n < p1; n++)
					{
						for (int t = 0; t < p0; t++)
						{
							(oData + (k + m)*i0*i1 + (j + n)*i0 + i)[t] += (iData + blocknum*p0*p1*p2 + m*p0*p1 + n*p0)[t];
							if (doAvg)
								(sData + (k + m)*i0*i1 + (j + n)*i0 + i)[t] += (nData)[0];
						}
					}
				}
				blocknum++;
			}
		}
	}

	if (doAvg)
		out /= scale;

	return out;
}

void blurGradient(Mat im, Mat &gX, Mat &gY)
{
	// blur the input image prior to taking the derivatives
	Mat imBlur;
	GaussianBlur(im, imBlur, Size(5, 5), 1.0, 1.0, BORDER_REFLECT);

	// set up the derivative kernels
	Mat kX = (Mat_<float>(1, 3) << -0.5, 0, 0.5);
	Mat kY = (Mat_<float>(3, 1) << -0.5, 0, 0.5);
	filter2D(imBlur, gX, -1, kX);
	filter2D(imBlur, gY, -1, kY);
}

Mat findPatchSize(Mat im, Mat_<unsigned char> sizes, Size pShift)
{
	// vectorize sizes (just in case)
	sizes = sizes.reshape(1, sizes.size().area());

	// find the gradients
	Mat_<float> imCopy = im.clone();
	Mat_<float> gX, gY;
	blurGradient(im, gX, gY);
	vector<int> patchCount(sizes.total());

	for (int i = 0; i < sizes.total(); i++)
		patchCount[i] = 0;

	// take the absolute value of the gradients
	gX = abs(gX);
	gY = abs(gY);

	// determine the value of epsilon that will be used
	// this choice is critical, must be large enough to avoid dividing by zero but not too large to kill the gradients
	// we use mean grad magnitude/2
	// this is done in the middle of the next step to avoid making another copy of the image

	// find the threshold to determine the window size to use for a particular patch
	float epsX, epsY, threshX, threshY;
	int indX, indY;
	Mat_<float> distX, distY;
	Mat_<float> sortX, sortY;
	sortX = gX.clone();
	sortY = gY.clone();

	// sort the gradients
	sortX = sortX.reshape(1, sortX.total());
	sortY = sortY.reshape(1, sortY.total());

	cv::sort(sortX, sortX, SORT_EVERY_COLUMN + SORT_ASCENDING); // use cv namespace to avoid ambiguity
	cv::sort(sortY, sortY, SORT_EVERY_COLUMN + SORT_ASCENDING);

	// use the reshaped gradient magnitudes to find the mean
	Mat_<float> meanX, meanY;
	reduce(sortX, meanX, 0, REDUCE_AVG);
	reduce(sortY, meanY, 0, REDUCE_AVG);
	epsX = meanX(0) / 2;
	epsY = meanY(0) / 2;

	// compute the cumulative sum of the images
	distX = sortX.clone();
	distY = sortY.clone();
	for (int i = 1; i < sortX.total(); i++)
	{
		distX(i) += distX(i - 1);
		distY(i) += distY(i - 1);
	}

	// divide by the total sum to get fractional values
	Scalar sumX = sum(sortX), sumY = sum(sortY);

	distX /= sumX[0];
	distY /= sumY[0];

	// find the location where the distribution of gradients goes above 0.25
	for (int i = 0; i < distX.total(); i++)
	{
		if (distX(i) > 0.25)
		{
			indX = i;
			break;
		}
	}
	for (int i = 0; i < distY.total(); i++)
	{
		if (distY(i) > 0.25)
		{
			indY = i;
			break;
		}
	}

	// add the epislon value to the gradients
	gX += epsX;
	gY += epsY;

	// set the threshold
	threshX = sortX(indX) + epsX;
	threshY = sortY(indY) + epsY;

	/*	heart of this function
	Find the patch size for each pixel */

	// initialize the output patchSizeInd to be the maximum patch size
	Mat_<unsigned char> patchSizeInd;
	{
		Mat tmp = im2colstep(gX, Size(sizes(0), sizes(0)), pShift); // let tmp run out of scope
		patchSizeInd = Mat_<unsigned char>::zeros(tmp.rows, sizes.total()) + sizes(sizes.total() - 1);
	}

	// if there is only one patch size, set the cost accordingly
	if (sizes.total() == 1)
		return patchSizeInd;

	// extract the appropriate value of the extracted and sorted patch
	// heuristic that at least N pixels are greater than the threshold 
	// N = 10% smallest patch size (e.g. 8*8/10);
	int N = sizes(0) * sizes(0) / 10; // integer rounding truncates the remainder
	N--;

	// for each patch determine whether the gradients are strong enough at each patch size
	// keep the smallest patch with valid patch sizes
	Mat gxTest, gyTest;
	for (int i = 0; i < sizes.total() - 1; i++)
	{
		if (i == 0)
		{
			gxTest = gX.clone();
			gyTest = gY.clone();
		}
		else // pad the arrays to make the centers align
		{
			int pad = sizes(i) / 4; // array sizes are powers of 2
			copyMakeBorder(gxTest, gxTest, pad, pad, pad, pad, BORDER_REFLECT);
			copyMakeBorder(gyTest, gyTest, pad, pad, pad, pad, BORDER_REFLECT);
		}

		// extract patches, sort them, determine if they are above the threshold
		int pW = sizes(i), pH;
		pH = pW;

		// get the patches for the given scale	
		Mat patchX, patchY;
		patchX = im2colstep(gxTest, Size(pW, pH), pShift);
		patchY = im2colstep(gyTest, Size(pW, pH), pShift);

		// sort the patches
		cv::sort(patchX, patchX, SORT_EVERY_ROW + SORT_DESCENDING);
		cv::sort(patchY, patchY, SORT_EVERY_ROW + SORT_DESCENDING);

		// extract the Nth column as determined earlier
		Mat xN, yN;

		patchX.col(N).copyTo(xN);
		patchY.col(N).copyTo(yN);

		// determine if the gradients are strong enough
		xN = xN > threshX;
		yN = yN > threshY;

		xN.convertTo(xN, CV_8U);
		yN.convertTo(yN, CV_8U);


		Mat_<unsigned char> mask;
		bitwise_and(xN, yN, mask);

		Scalar numPatches = sum(mask / 250); // rounding will make this 1
		patchCount[i] = (int)numPatches[0];
		int curPatches=0;
		if (i>0)
			curPatches = patchCount[i-1];

		printf("Found %d valid patches with size %d\n", patchCount[i]-curPatches, sizes(i));

		patchSizeInd.col(i).setTo(sizes(i), mask);
	}
	patchCount[sizes.total() - 1] = patchSizeInd.rows - patchCount[sizes.total() - 2];
	printf("Found %d valid patches with size %d\n", patchCount[sizes.total() - 1], sizes(sizes.total() - 1));

	// keep the patches which are smallest but pass the robustness test
	reduce(patchSizeInd, patchSizeInd, 1, REDUCE_MIN);

	return patchSizeInd;
}

Mat findCost(int tDisp, int minDisp, int refCam, int nPatches, vector<Mat> gX_vec, vector<Mat> gY_vec, vector<Mat> im_vec, Mat_<unsigned char> pSizes, Size pShift, vector<Mat> Rvec, vector<Mat> T, vector<Mat> K, Mat Q, vector<Mat> maskSize)
{
	Size imSz = im_vec[0].size();
	int nCams = im_vec.size();
	int numPSize = pSizes.total();

	Mat_<float> costGxGy(nPatches, tDisp);

	// heart of the algorithm
	printf("Iter.:\tDepth:\tTime:\n");
	for (int i = 0; i < tDisp; i++)
	{
		printf("%03d\t%03d\t", i, minDisp + i);
		double startTime = omp_get_wtime();

		// reproject the images to a common depth plane
		Mat vDisp = Mat::ones(imSz, CV_32FC1)*(minDisp + i);
		Mat WC;
		reprojectImageTo3D(vDisp, WC, Q);
		WC = WC.reshape(3, 1);

		vector<Mat> px_cube_x(nCams), px_cube_y(nCams), warpedIm(nCams);
		for (int j = 0; j < nCams; j++)
		{
			if (j == refCam)
			{
				px_cube_x[j] = gX_vec[j].clone();
				px_cube_y[j] = gY_vec[j].clone();
				continue;
			}
			Mat tmpX, tmpY, tmpPts, map1, map2, d0 = Mat::zeros(1, 5, CV_32FC1);
			vector<Mat> pts;
			projectPoints(WC, Rvec[j], T[j], K[j], d0, tmpPts);

			// interpolate the projected images
			// use the remap function to perform the pixel interpolations
			// we convert the maps to fixed point representation (slight loss in accuracy, but 2x speedup)
			// for extrapolation, use a constant border value of zero (could also use transparent borders)
			tmpPts = tmpPts.reshape(2, imSz.height);
			split(tmpPts, pts);
			convertMaps(pts[0], pts[1], map1, map2, CV_16SC2);
			remap(gX_vec[j], px_cube_x[j], map1, map2, INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
			remap(gY_vec[j], px_cube_y[j], map1, map2, INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

			// the remaped image needs to be reshaped
			px_cube_x[j] = px_cube_x[j].reshape(1, imSz.height);
			px_cube_y[j] = px_cube_y[j].reshape(1, imSz.height);
		}

		// Flexible patch size cost
		Mat_<float> xyCost;
		Mat_<float> patches;
		vector<Mat_<float>> xCost(numPSize), yCost(numPSize);

		for (int j = 0; j < nCams; j++)
		{
			Mat_<float> imX, imY;
			for (int k = 0; k < numPSize; k++)
			{
				Mat_<float> mean;
				if (k == 0)
				{
					imX = px_cube_x[j].clone();
					imY = px_cube_y[j].clone();
				}
				else
				{
					int pad = pSizes(k) / 4;
					copyMakeBorder(imX, imX, pad, pad, pad, pad, BORDER_REFLECT);
					copyMakeBorder(imY, imY, pad, pad, pad, pad, BORDER_REFLECT);
				}

				Size patchSz(pSizes(k), pSizes(k));

				// run the x patches
				patches = im2colstep(imX, patchSz, pShift);

				reduce(patches, mean, 1, REDUCE_AVG);
				for (int m = 0; m < patches.rows; m++)
					patches.row(m) /= mean(m);

				if (j == 0)
					xCost[k] = patches.clone();
				else
					xCost[k] = xCost[k].mul(patches);

				// run the y patches
				patches = im2colstep(imY, patchSz, pShift);

				reduce(patches, mean, 1, REDUCE_AVG);
				for (int m = 0; m < patches.rows; m++)
					patches.row(m) /= mean(m);

				if (j == 0)
					yCost[k] = patches.clone();
				else
					yCost[k] = yCost[k].mul(patches);
			}
		}

		// after computing the patch costs for each scale find the total cost
		for (int k = 0; k < numPSize; k++)
		{
			reduce(xCost[k] + yCost[k], xyCost, 1, REDUCE_SUM);
			xyCost /= (pSizes(k) * pSizes(k) * 2);
			xyCost.copyTo(costGxGy.col(i), maskSize[k]);
		}
		printf("%0.4f seconds\n", omp_get_wtime() - startTime);
	}

	// take the nth root of the costs
	float nth = 1 / float(nCams);
	for (int i = 0; i < costGxGy.total(); i++)
		costGxGy(i) = pow(costGxGy(i), nth);

	// find the mean of each row and normalize each
	Mat_<float> mean;
	reduce(costGxGy, mean, 1, REDUCE_AVG);
	for (int i = 0; i < costGxGy.rows; i++)
		costGxGy.row(i) /= mean(i);

	Mat	mask = Mat(costGxGy != costGxGy);
	costGxGy.setTo(0, mask);

	// square the costs
	costGxGy = costGxGy.mul(costGxGy);

	// finally, negate the cost so that the graph cuts optimization can find a minimum
	costGxGy *= -1;

	return costGxGy;
}

// warp input images using a disparity map
vector<Mat> warpIms(vector<Mat> ims, Mat_<int> dispMap, int minDisp, int tDisp, int refCam, vector<Mat> Rvec, vector<Mat> T, vector<Mat> K, Mat Q)
{
	// simply reproject the images using the same plane sweep technique that was used to compute the cost
	// for pixels corresponding to a particular depth, keep the values
	Size imSz = ims[0].size();
	int nCams = ims.size();

	vector<Mat> warpedIms(nCams);
	for (int j = 0; j < nCams; j++)
		warpedIms[j] = Mat_<float>(imSz);

	Mat WC;
	reprojectImageTo3D(dispMap + minDisp, WC, Q);
	WC = WC.reshape(3, 1);

	warpedIms[refCam] = ims[refCam].clone(); // the reference camera doesn't need to be warped
		
	for (int j = 0; j < nCams; j++)
	{
		Mat tmpIm;
		if (j == refCam)
			continue; // the reference camera doesn't need to be warped

		Mat tmpX, tmpY, tmpPts, map1, map2, d0 = Mat::zeros(1, 5, CV_32FC1);
		vector<Mat> pts;
		projectPoints(WC, Rvec[j], T[j], K[j], d0, tmpPts);

		// interpolate the projected images
		// use the remap function to perform the pixel interpolations
		// we convert the maps to fixed point representation (slight loss in accuracy, but 2x speedup)
		// for extrapolation, use a constant border value of zero (could also use transparent borders)
		tmpPts = tmpPts.reshape(2, imSz.height);
		split(tmpPts, pts);
		convertMaps(pts[0], pts[1], map1, map2, CV_16SC2);
		remap(ims[j], tmpIm, map1, map2, INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

		// the remaped image needs to be reshaped
		tmpIm = tmpIm.reshape(ims[j].channels(), imSz.height);
			

		// for valid pixels, copy to the warped image
		tmpIm.copyTo(warpedIms[j]);
	}

	return warpedIms;
}

double sigmoidCost(double x, double alpha, double beta)
{
	x = abs(x);
	return (1 / (1 + exp((x - alpha) / beta)));
}

/* THIRD PARTY CODE
This uses the GCO v3 toolbox from http://vision.csd.uwo.ca/code/

Cost is the computed data cost cast as an integer (could be float, integers are used for speed)
refIm is the reference image - used to compute neighboring weights
gcDepth is the output graph cuts depth map
neighbors are computed as a weighted signmoid
alpha is the mean of the sigmoid, beta is the width. The default values should work for most cases.
*/
void runGC(Mat_<int> cost, Mat_<float> refIm, Mat_<int> &gcDepth, Size patchSz, Size patchShift, double alpha = 0.1, double beta = 0.005)
{

	int num_pixels, num_labels;
	num_pixels = cost.rows;
	num_labels = cost.cols;

	int *result = new int[num_pixels];   // stores result of optimization

	Mat_<float> refCopy = refIm.clone();

	printf("\n");
	// next set up the array for smooth costs, this is a simple L1 cost, but any convex cost could be used
	int *smooth = new int[num_labels*num_labels];
	for (int l1 = 0; l1 < num_labels; l1++)
		for (int l2 = 0; l2 < num_labels; l2++)
			smooth[l1 + l2*num_labels] = abs(l1 - l2);

	gcDepth = Mat::zeros(num_pixels, 1, CV_16UC1);
	try{
		int width = refIm.cols;
		int height = refIm.rows;
		GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(num_pixels, num_labels);

		// add data costs individually
		for (int i = 0; i<num_pixels; i++)
			for (int j = 0; j<num_labels; j++)
				gc->setDataCost(i, j, cost(i, j));

		gc->setSmoothCost(smooth);

		/* Set up neighbors for graph cuts
		Median filter the reference image with a 5x5 window. Pixels intensites that are far apart are likely at different depths.
		OpenCV has a limitation on the size of the window, it can only be 3x3 or 5x5 for 32F, otherwise you must convert to 8U
		*/
		{
			// check to see that the inputs are of the correct type
			// must be grayscale
			if (refCopy.channels()>1)
				cvtColor(refCopy, refCopy, CV_BGR2GRAY);

			int kSize = 5;

			// median filter the reference image
			medianBlur(refCopy, refCopy, kSize);

			// set the sigmoid thresholds on neighborliness 
			// alpha and beta are provided in the function call
			double weightScalingFactor = 100;

			// set up the weighted neighbors, we're going to be lazy and use the im2colstep function to provide the mappings
			int width = (refCopy.cols - patchSz.width) / patchShift.width + 1;
			int height = (refCopy.rows - patchSz.height) / patchShift.height + 1;

			// number the blocks
			int nBlocks = num_pixels;
			Mat blocks(1, num_pixels, CV_32FC1);
			for (int i = 0; i < nBlocks; i++)
				blocks.at<float>(i) = i;
			blocks = blocks.reshape(1, height);

			// subsample the image
			Mat_<float> im(height, width);

			int curY = 0;
			for (int y = 0; y < refCopy.rows - patchSz.height + 1; y += patchShift.height)
			{
				int curX = 0;
				for (int x = 0; x < refCopy.cols - patchSz.width + 1; x += patchShift.width)
				{
					im(curY, curX) = refCopy(y, x);
					curX++;
				}
				curY++;
			}

			// pad the subsampled image and the blocks
			copyMakeBorder(im, im, 1, 1, 1, 1, BORDER_CONSTANT, -1);
			copyMakeBorder(blocks, blocks, 1, 1, 1, 1, BORDER_CONSTANT, -1);

			// use im2colstep to get the neighbors
			Mat_<float> weights = im2colstep(im, Size(3, 3), Size(1, 1));
			Mat blockNum = im2colstep(blocks, Size(3, 3), Size(1, 1));
			blockNum.convertTo(blockNum, CV_32SC1);

			// go through the block weights and add the weighted neighbors to the optimization
			for (int i = 0; i < weights.rows; i++)
			{
				float W = weights(i, 4); // patch value

				for (int j = 0; j < weights.cols; j++)
				{
					// check to make sure the position is valid (will be negative at the borders)
					if (weights(i, j) < 0)
						continue;
					if (j == 4) // skip the center as a patch cannot neighbor itself
						continue;

					// compute the weight
					double diff = double(weights(i, j) - W);
					double curWeight = sigmoidCost(diff, alpha, beta);
					int scaledWeight = int(weightScalingFactor * curWeight);
					gc->setNeighbors(i, blockNum.at<int>(i, j), scaledWeight);
				}
			}
		}

		gc->setVerbosity(1);
		printf("Before optimization energy is %lld\n", gc->compute_energy());
		double startTime = omp_get_wtime();
		gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		printf("Time for expansion is %4f seconds\n", omp_get_wtime() - startTime);
		printf("After optimization energy is %lld\n", gc->compute_energy());

		for (int i = 0; i < num_pixels; i++)
		{
			result[i] = gc->whatLabel(i);
			gcDepth(i) = gc->whatLabel(i);
		}

		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

	delete[] result;
	delete[] smooth;
}