#pragma once
#include<vector>
/***
default voc ssd300
****/
typedef struct ssd_config
{
	int nNumClass = 21;
	int nMinDim = 300;
	std::vector<int> vecMinSizes = { 30, 60, 111, 162, 213, 264 };
	std::vector<int> vecMaxSizes = { 60, 111, 162, 213, 264, 315 };
	std::vector<int> vecFeatureMap = { 38, 19, 10, 5, 3, 1 };
	std::vector<int> vecStep = { 8, 16, 32, 64, 100, 300 };
	std::vector<float> vecVariance = {0.1f, 0.2f };
	std::vector<std::vector<int>> vecAspectRatios = { {2},{2, 3},{2, 3},{2, 3},{2},{2} };

}ssd_conf;

typedef struct rectb
{
	float left;
	float top;
	float right;
	float bottom;
}rect;

typedef struct detResult
{
	rect box;
	float score;
	int label;

}detections;