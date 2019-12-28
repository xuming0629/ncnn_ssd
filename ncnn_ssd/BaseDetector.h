#pragma once
namespace ncnn_det
{
	class BaseDetector
	{
	public:
		BaseDetector();
		~BaseDetector();
		virtual bool loadModel(const char * parmFile, const char* binFile);
		virtual void detector(unsigned char *pImage,const float* mean, const float* std);
		
	};
}

