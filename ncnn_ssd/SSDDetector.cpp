#include "SSDDetector.h"

namespace ncnn_det
{

	SSDDetector::SSDDetector()
	{
		 m_SSDNet.reset(new ncnn::Net);
	}


	SSDDetector::~SSDDetector()
	{
		m_SSDNet.get()->clear();
		m_SSDNet.reset();
	}
	bool SSDDetector::loadModel(const char * parmFile, const char* binFile)
	{
		if(!m_SSDNet)
		{
			m_SSDNet.reset(new ncnn::Net);
		}
		if (m_SSDNet.get()->load_param(parmFile) || m_SSDNet.get()->load_model(binFile))
		{
			printf(" load model falsed!!!, param or bin file is not exit or bad\n");
			return false;
			
		}

		return true;

	}

	void SSDDetector::detector(unsigned char *pImage, const float* mean, const float* std)
	{

	}
}