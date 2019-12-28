#include <iostream>
#include <opencv2\opencv.hpp>
#include "net.h"
#include "common.h"
#include "PriorBox.h"
using namespace ncnn_det;
int main() 
{
	cv::Mat img;
	ncnn::Mat matNcnn;
	ssd_conf conf;
	rect rec = { 1, 3,4,5 };
	std::cout << conf.vecAspectRatios[0][0]<< rec.left;

	ncnn_det::PriorBox priorBox;
	rect * deltaBox;
	int nDeltaSize = 0;
	deltaBox = priorBox.computePriorBox(nDeltaSize);
	rect * box = deltaBox;
	box = box + (nDeltaSize - 20);
	for (int i = 0; i < 20; i++, box++)
	{
		printf(" %f %f %f %f \n", box->left, box->top, box->right, box->bottom);
	}
	printf("size %d\n", nDeltaSize);


	system("pause");
}