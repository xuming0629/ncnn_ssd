#include "BaseDetector.h"
namespace ncnn_det
{
	

	BaseDetector::BaseDetector()
	{
	}


	BaseDetector::~BaseDetector()
	{
	}

	bool BaseDetector::loadModel(const char * parmFile, const char* binFile)
	{
		return false;
	}

	void BaseDetector::detector(unsigned char *pImage, int nWidth, int nHeight, const float* pMean, const float* pStd, const int dataType)
	{

	}
}
