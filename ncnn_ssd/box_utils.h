#pragma once
#include "common.h"
namespace ncnn_det
{
	void delta2Box(std::vector<detInfo>vecDeltaBoxes,std::vector<rect> vecPriorBoxes,
			std::vector<float>vecVariance,std::vector<detInfo> &vecBoxes);
	void nms(std::vector<detInfo>vecInputInfo, std::vector<detInfo> &vecOutInfo, const float threshold = 0.5);
}