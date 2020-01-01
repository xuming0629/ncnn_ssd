#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <opencv2\opencv.hpp>


// #include "common.h"
#include "PriorBox.h"
#include "SSDDetector.h"
#include "net.h"
#include "benchmark.h"
using namespace ncnn_det;

 std::vector<std::string> VOC_CLASSES = {
	"aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair",
	"cow", "diningtable", "dog", "horse",
	"motorbike", "person", "pottedplant",
	"sheep", "sofa", "train", "tvmonitor" };

void ssd_demo()
{

	cv::Mat img = cv::imread("../data/car.jpg");
	if (img.empty())
	{
		printf("image is empty, check path of imgage !!!\n");
		return;
	}
	cv::Mat src_img = img.clone();
	printf("%d %d \n", img.cols, img.rows);
	cv::resize(img, img, cv::Size(300, 300));
	printf("%d %d", img.rows, img.cols);
	char parmFile[] = "../models/ssd/ssd_vgg300.param";
	char binFile[] = "../models/ssd/ssd_vgg300.bin";
	SSDDetector ssdDet;
	ssdDet.loadModel(parmFile, binFile);
	ssdDet.setNumTheads(1);
	const float fMean[3] = { 104.0f, 117.0f, 123.0f };

	double start_time = ncnn::get_current_time();
	ssdDet.detector(img.data, img.cols, img.rows, fMean, NULL, 0);
	double end_time = ncnn::get_current_time();

	std::vector<detInfo> result = ssdDet.getDetectInfo();
	printf("object number is %d\n", result.size());
	for (int i = 0; i < int(result.size()); i++)
	{
		int x1 = int(result[i].box.left * src_img.cols);
		int y1 = int(result[i].box.top * src_img.rows);
		int x2 = int(result[i].box.right * src_img.cols);
		int y2 = int(result[i].box.bottom * src_img.rows);
		
		float score = result[i].score;
		int label = result[i].label;
		cv::rectangle(src_img, cv::Rect(x1, y1, (x2 - x1 + 1), (y2 - y1 + 1)), cv::Scalar(0, 0, 255), 2);
		cv::putText(src_img, VOC_CLASSES[label-1] + " " + std::to_string(score), cv::Point(int(x1), int(y1 - 3)), 2, 1, cv::Scalar(255, 255, 0), 1);
		printf("x1:%d y1:%d x2:%d y2:%d score: %.2f label: %d\n ", x1, y1, x2, y2, score, label);
	}

	printf("ssd vgg forward spend time:%.3f ms\n", (end_time - start_time));
	cv::imshow("detect", src_img);
	cv::waitKey(-1);
	
}
int main()
{
	ssd_demo();
	system("pause");
}