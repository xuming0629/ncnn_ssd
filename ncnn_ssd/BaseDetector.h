#pragma once
namespace ncnn_det
{
	/**
	object detector class base ncnn framework 
	**/
	class BaseDetector
	{
	public:
		BaseDetector();
		~BaseDetector();
		virtual bool loadModel(const char * parmFile, const char* binFile);
		virtual void detector(unsigned char *pImage, int nWidth, int nHeight, const float* pMean, const float* pStd, const int dataType);
		
	};
}

