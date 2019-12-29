#include "box_utils.h"
#include <math.h>
#include <iostream>
#include <algorithm>
#include <map>

namespace ncnn_det
{
	void delta2Box(std::vector<rect>vecDeltaBoxes, std::vector<rect> vecPriorBoxes,
		std::vector<float>vecVariance, std::vector<rect> &vecBoxes)
	{
		if(int(vecDeltaBoxes.size()) != int(vecPriorBoxes.size()))
		{
			printf("delta boxes size or prior boxes is bad!!!\n");
			return;
		}
		if(vecVariance.size() != 2)
		{
			printf("variance size is != 2 !!!\\n");
			return;
		}
		if(!vecBoxes.empty())
		{
			vecBoxes.clear();
		}
		for (int i = 0; i < int(vecDeltaBoxes.size()); i++)
		{
			rect info = vecDeltaBoxes[i];
			// center
			info.left = vecPriorBoxes[i].left + vecPriorBoxes[i].left * (info.left * vecVariance[0]);
			info.top = vecPriorBoxes[i].top + vecPriorBoxes[i].top * (info.top * vecVariance[1]);

			// w h
			info.right = vecPriorBoxes[i].right  * expf(info.right * vecVariance[0]);
			info.bottom = vecPriorBoxes[i].bottom * expf(info.bottom * vecVariance[1]);

			
			//box: erea left tot right bottom
			// info.area = info.box.right * info.box.bottom;

			info.left = info.left - (info.right / 2.0f);
			info.top = info.top - (info.bottom / 2.0f);
			info.right = info.right + (info.right / 2.0f);
			info.bottom = info.bottom + (info.bottom / 2.0f);
			vecBoxes.push_back(info);
		}

	}
	void nms(std::vector<detInfo>vecInputInfo, std::vector<detInfo> &vecOutInfo , const float threshold)
	{
		/*void Centerface::nms(std::vector<FaceInfo>& input, std::vector<FaceInfo>& output, float nmsthreshold, int type)
		{*/
		if (vecInputInfo.empty()) {
			return;
		}
		std::sort(vecInputInfo.begin(), vecInputInfo.end(),
			[](const detInfo& a, const detInfo& b)
		{
			return a.score < b.score;
		});

		float IOU = 0.0f;
		float maxX = 0.0f;
		float maxY = 0.0f;
		float minX = 0.0f;
		float minY = 0.0f;
		std::vector<int> vPick;
		int nPick = 0;
		std::multimap<float, int> vScores;
		const int num_boxes = vecInputInfo.size();
		vPick.resize(num_boxes);
		for (int i = 0; i < num_boxes; ++i) {
			vScores.insert(std::pair<float, int>(vecInputInfo[i].score, i));
		}
		while (vScores.size() > 0) {
			int last = vScores.rbegin()->second;
			vPick[nPick] = last;
			nPick += 1;
			for (std::multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();) {
				int it_idx = it->second;
				maxX = std::max(vecInputInfo.at(it_idx).box.left, vecInputInfo.at(last).box.left);
				maxY = std::max(vecInputInfo.at(it_idx).box.top, vecInputInfo.at(last).box.top);
				minX = std::min(vecInputInfo.at(it_idx).box.right, vecInputInfo.at(last).box.right);
				minY = std::min(vecInputInfo.at(it_idx).box.bottom, vecInputInfo.at(last).box.bottom);
				//maxX1 and maxY1 reuse 
				maxX = ((minX - maxX + 1) > 0.0f) ? (minX - maxX + 1) : 0.0f;
				maxY = ((minY - maxY + 1) > 0.0f) ? (minY - maxY + 1) : 0.0f;
				//IOU reuse for the area of two bbox
				IOU = maxX * maxY;
				
				IOU = IOU / (vecInputInfo.at(it_idx).area + vecInputInfo.at(last).area - IOU);
				
				if (IOU > threshold) {
					it = vScores.erase(it);
				}
				else {
					it++;
				}
			}
		}

		vPick.resize(nPick);
		vecOutInfo.resize(nPick);
		for (int i = 0; i < nPick; i++) {
			vecOutInfo[i] = vecInputInfo[vPick[i]];
		}

	}
}