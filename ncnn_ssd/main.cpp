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
	cv::Mat img = cv::imread("../data/grand.jpg");
	cv::resize(img, img, cv::Size(300, 300));

	/*cv::imshow("img", img);
	cv::waitKey(-1);*/
	//ncnn::Mat in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows);
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows, 300, 300);

	
	//printf("%f", *(float*)in.data);
	printf("input size %d %d %d\n", in.c, in.w, in.h);
	const float fMean[3] = { 104.0f, 117.0f, 123.0f };
	in.substract_mean_normalize(fMean, 0);

	/*ncnn::Mat input = in.reshape(in.c * in.w * in.h);
	for (int i = 0; i < 10; i++)
	{
		printf("%f\n", input[i]);
	}*/

	

	ncnn::Mat matCls;
	ncnn::Mat matReg;
	ncnn::Net net;

	// net.opt.use_vulkan_compute = true;

	char parmFile[] = "../models/ssd/ssd_vgg300_1231.param";
	char binFile[] = "../models/ssd/ssd_vgg300_1231.bin";
	net.load_param(parmFile);
	net.load_model(binFile);
	ncnn::Extractor ext = net.create_extractor();
	ext.set_light_mode(true);
	ext.input("input", in);
	ext.extract("cls", matCls);
	ext.extract("reg", matReg);

	printf("cls shape: c:%d  height: %d  width: %d \n", matCls.c, matCls.h, matCls.w);
	printf("reg shape: c:%d  height: %d  width: %d \n", matReg.c, matReg.h, matReg.w);

	ncnn::Mat matOutReg = matReg.reshape(matReg.c * matReg.h * matReg.w);
	ncnn::Mat matOutCls = matCls.reshape(matCls.c * matCls.h * matCls.w);

	std::vector<float> vecReg;
	vecReg.resize(matOutReg.c * matOutReg.h * matOutReg.w);
	std::vector<float> vecCls;
	vecCls.resize(matOutCls.c * matOutCls.h * matOutCls.w);

	int count = 0;
	for (int i = 0; i < vecReg.size(); i++)
	{
		vecReg[i] = matOutReg[i];
		vecCls[i] = matOutCls[i];
		
		if (vecCls[i] > 0.5)
		{
			printf("reg %f\n", vecReg[i]);
			printf("cls %f\n", vecCls[i]);
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
	cv::Mat src_img = img.clone();
	printf("%d %d \n", img.cols, img.rows);
	cv::resize(img, img, cv::Size(300, 300));
	printf("%d %d", img.rows, img.cols);
	char parmFile[] = "../models/ssd/ssd_vgg300_1231.param";
	char binFile[] = "../models/ssd/ssd_vgg300_1231.bin";
	SSDDetector ssdDet;
	ssdDet.loadModel(parmFile, binFile);

	const float fMean[3] = { 104.0f, 117.0f, 123.0f };
	
	double start_time = ncnn::get_current_time();
	ssdDet.detector(img.data, img.cols, img.rows, fMean, NULL, 0);
	double end_time = ncnn::get_current_time();

	std::vector<detInfo> result = ssdDet.getDetectInfo();
	printf("111%d\n", result.size());
	for(int i =0; i < int(result.size()); i++)
	{
		int x1 = int(result[i].box.left * src_img.cols);
		int y1 = int(result[i].box.top * src_img.rows);
		int x2 = int(result[i].box.right * src_img.cols);
		int y2 = int(result[i].box.bottom * src_img.rows);

		cv::rectangle(src_img, cv::Rect(x1, y1, (x2 - x1 + 1), (y2 - y1 + 1)), cv::Scalar(0, 0, 255), 2);
		
	}
	
	//printf("ssd vgg forward spend time:%.3f ms\n", (end_time - start_time));
	 cv::imshow("detect", src_img);
	 cv::waitKey(-1);
	//
	// demo();
	system("pause");
}