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

	void BaseDetector::detector(unsigned char *pImage,const float* mean, const float* std)
	{

	}
}
