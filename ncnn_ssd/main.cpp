#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <opencv2\opencv.hpp>
#include "net.h"
// #include "common.h"
#include "PriorBox.h"
#include "SSDDetector.h"
#include "benchmark.h"
using namespace ncnn_det;
int main()
{

	// ncnn::Mat matNcnn;
	/*ssd_conf conf;
	rect rec = { 1, 3,4,5 };
	std::cout << conf.vecAspectRatios[0][0] << rec.left;

	ncnn_det::PriorBox priorBox;
	rect * deltaBox;
	int nDeltaSize = 0;
	deltaBox = priorBox.computePriorBox(nDeltaSize);
	rect * box = deltaBox;
	box = box + (nDeltaSize - 20);*/
	/*for (int i = 0; i < 20; i++, box++)
	{
		printf(" %f %f %f %f \n", box->left, box->top, box->right, box->bottom);
	}
	printf("size %d\n", nDeltaSize);*/

	cv::Mat img = cv::imread("../data/example.jpg");
	cv::resize(img, img, cv::Size(300, 300));
	char parmFile[] = "C:/Users/jsc/Desktop/src_git/ncnn_ssd/models/ssd/ssd_vgg300.param";
	char binFile[] = "C:/Users/jsc/Desktop/src_git/ncnn_ssd/models/ssd/ssd_vgg300.bin";
	SSDDetector ssdDet;
	ssdDet.loadModel(parmFile, binFile);

	const float fMean[3] = { 104.0f, 117.0f, 123.0f };
	
	double start_time = ncnn::get_current_time();
	ssdDet.detector(img.data, img.cols, img.rows, fMean, NULL, 0);
	double end_time = ncnn::get_current_time();
	//printf("ssd vgg forward spend time:%.3f ms\n", (end_time - start_time));
	/* cv::imshow("src", img);
	 cv::waitKey(-1);*/
	
	system("pause");
}