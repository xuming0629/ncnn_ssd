#include "box_utils.h"
#include <math.h>
#include <iostream>
#include <algorithm>
#include <map>

namespace ncnn_det
{
	void delta2Box(std::vector<detInfo>vecDeltaBoxes, std::vector<rect> vecPriorBoxes,
		std::vector<float>vecVariance, std::vector<detInfo> &vecBoxes)
	{
		if (int(vecDeltaBoxes.size()) != int(vecPriorBoxes.size()))
		{
			printf("delta boxes size or prior boxes is bad!!!\n");
			return;
		}
		if (vecVariance.size() != 2)
		{
			printf("variance size is != 2 !!!\\n");
			return;
		}
		if (!vecBoxes.empty())
		{
			vecBoxes.clear();
		}
		for (int i = 0; i < int(vecDeltaBoxes.size()); i++)
		{

			detInfo info = vecDeltaBoxes[i];
			detInfo info1 = info;
			// center
			info.box.left = vecPriorBoxes[i].left + vecPriorBoxes[i].right * (info.box.left * vecVariance[0]);
			info.box.top = vecPriorBoxes[i].top + vecPriorBoxes[i].bottom * (info.box.top * vecVariance[0]);

			// w h
			info.box.right = vecPriorBoxes[i].right  * exp((info.box.right * vecVariance[1]));
			info.box.bottom = vecPriorBoxes[i].bottom * exp((info.box.bottom * vecVariance[1]));



			//box: erea left tot right bottom
			info.area = std::max((info.box.right * info.box.bottom), 0.0f);

			float c_x = info.box.left;
			float c_y = info.box.top;
			float w = info.box.right;
			float h = info.box.bottom;
			info.box.left = c_x - (w / 2.0f);
			info.box.top = c_y - (h / 2.0f);
			info.box.right = c_x + (w / 2.0f);
			info.box.bottom = c_y + (h / 2.0f);
		/*	if (i == 0)
			{
				printf("variance:\n %f %f", vecVariance[0], vecVariance[1]);
				printf("decode1: %f %f %f %f \n", info1.box.left, info1.box.top, info1.box.right, info1.box.bottom);
				printf("decode1: %f %f %f %f \n", info.box.left, info.box.top, info.box.right, info.box.bottom);
				printf("prior: %f %f %f %f \n", vecPriorBoxes[i].left, vecPriorBoxes[i].top, vecPriorBoxes[i].right, vecPriorBoxes[i].bottom);
			}*/
			vecBoxes.push_back(info);
		}

	}
	void nms_ssd(std::vector<detInfo>vecInputInfo, std::vector<detInfo> &vecOutInfo, const float threshold, const int top_k)
	{
		if (vecInputInfo.empty())
		{
			return;
		}
		std::sort(vecInputInfo.begin(), vecInputInfo.end(),
			[](const detInfo& a, const detInfo& b)
		{
			return a.score > b.score;
		});

		int nBoxNum = std::min(top_k, int(vecInputInfo.size()));

		std::vector<int> vecIdxs(nBoxNum, 0);
		for (int i = 0; i < nBoxNum; i++)
		{
			if (vecIdxs[i])
				continue;

			vecOutInfo.push_back(vecInputInfo[i]);


			float area0 = vecInputInfo[i].area;

			for (int j = i + 1; j < nBoxNum; j++)
			{
				if (vecIdxs[j])
					continue;

				float inner_x0 = vecInputInfo[i].box.left > vecInputInfo[j].box.left ? vecInputInfo[i].box.left : vecInputInfo[j].box.left;//std::max(input[i].x1, input[j].x1);
				float inner_y0 = vecInputInfo[i].box.top > vecInputInfo[j].box.top ? vecInputInfo[i].box.top : vecInputInfo[j].box.top;

				float inner_x1 = vecInputInfo[i].box.right < vecInputInfo[j].box.right ? vecInputInfo[i].box.right : vecInputInfo[j].box.right;  //bug fixed ,sorry
				float inner_y1 = vecInputInfo[i].box.bottom < vecInputInfo[j].box.bottom ? vecInputInfo[i].box.bottom : vecInputInfo[j].box.bottom;

				float inner_h = inner_y1 - inner_y0;
				float inner_w = inner_x1 - inner_x0;


				if (inner_h <= 0 || inner_w <= 0)
					continue;

				float inner_area = inner_h * inner_w;



				float area1 = vecInputInfo[j].area;

				float score;

				score = inner_area / (area0 + area1 - inner_area);
				if (score > threshold)
					vecIdxs[j] = 1;
			}

		}
	}
}