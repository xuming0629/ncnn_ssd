#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <opencv2\opencv.hpp>


// #include "common.h"
#include "PriorBox.h"
#include "SSDDetector.h"
#include "platform.h"
#include "net.h"
#include "gpu.h"
#include "benchmark.h"
using namespace ncnn_det;
void demo()
{
	cv::Mat img = cv::imread("../data/example.jpg");
	cv::resize(img, img, cv::Size(300, 300));

	//ncnn::Mat in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows);
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows, 300, 300);
	printf("input size %d %d %d\n", in.c, in.w, in.h);
	const float fMean[3] = { 104.0f, 117.0f, 123.0f };
	in.substract_mean_normalize(fMean, 0);
	

	ncnn::Mat matCls;
	ncnn::Mat matReg;
	ncnn::Net net;

	net.opt.use_vulkan_compute = true;

	char parmFile[] = "../models/ssd/ssd_vgg300_1229.param";
	char binFile[] = "../models/ssd/ssd_vgg300_1229.bin";
	net.load_param(parmFile);
	net.load_model(binFile);
	ncnn::Extractor ext = net.create_extractor();
	ext.set_light_mode(false);
	ext.input("input", in);
	ext.extract("reg", matReg);
	ext.extract("cls", matCls);

	printf("cls shape: c:%d  height: %d  width: %d \n", matCls.c, matCls.h, matCls.w);
	printf("reg shape: c:%d  height: %d  width: %d \n", matReg.c, matReg.h, matReg.w);

	ncnn::Mat matOutReg = matReg.reshape(matReg.c * matReg.h * matReg.w);

	std::vector<float> vecReg;
	vecReg.resize(matOutReg.c * matOutReg.h * matOutReg.w);
	int count = 0;
	for (int i = 0; i < vecReg.size(); i++)
	{
		vecReg[i] = matOutReg[i];
		if (vecReg[i] < 10)
		{
			printf("%f\n", vecReg[i]);
			count += 1;
		}
		
	}
	printf("%d %d\n", count, matOutReg.c * matOutReg.h * matOutReg.w);
	

}
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
	printf("%d %d \n", img.cols, img.rows);
	cv::resize(img, img, cv::Size(300, 300));
	char parmFile[] = "../models/ssd/ssd_vgg300.param";
	char binFile[] = "../models/ssd/ssd_vgg300.bin";
	SSDDetector ssdDet;
	ssdDet.loadModel(parmFile, binFile);

	const float fMean[3] = { 104.0f, 117.0f, 123.0f };
	
	double start_time = ncnn::get_current_time();
	ssdDet.detector(img.data, img.cols, img.rows, fMean, NULL, 0);
	double end_time = ncnn::get_current_time();
	//printf("ssd vgg forward spend time:%.3f ms\n", (end_time - start_time));
	///* cv::imshow("src", img);
	// cv::waitKey(-1);*/
	//
	// demo();
	system("pause");
}