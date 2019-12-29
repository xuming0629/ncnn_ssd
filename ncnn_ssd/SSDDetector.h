#pragma once

#include <iostream>
#include "BaseDetector.h"
#include "PriorBox.h"
#include "common.h"
#include <memory>
#include "net.h"
#include "mat.h"

namespace ncnn_det
{
	class PriorBox;
	class SSDDetector:public BaseDetector
	{
	public:
		SSDDetector();
		~SSDDetector();
		virtual bool loadModel(const char * parmFile, const char* binFile);
		virtual void detector(unsigned char *pImage, int nWidth, int nHeight, const float* pMean, const float* pStd, const int dataType);

	private:
		std::vector<detInfo> m_vecDet;
		std::shared_ptr <ncnn::Net> m_SSDNet;
		std::shared_ptr <PriorBox> m_priorBox;
		std::vector<rect> m_vecPriorBoxes;
		float m_confThresh = 0.5;
		float m_nmsThresh = 0.5;
	};
}

