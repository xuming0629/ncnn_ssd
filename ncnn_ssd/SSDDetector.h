#pragma once

#include <iostream>
#include <memory>

#include "common.h"
#include "BaseDetector.h"
#include "net.h"

namespace ncnn_det
{
	class SSDDetector:public BaseDetector
	{
	public:
		SSDDetector();
		~SSDDetector();
		virtual bool loadModel(const char * parmFile, const char* binFile);
		virtual void detector(unsigned char *pImage, const float* mean, const float* std);

	private:
		std::vector<detections> m_vecDet;
		std::shared_ptr <ncnn::Net> m_SSDNet;
		//ncnn::Net m_SSDNet;
	};
}

