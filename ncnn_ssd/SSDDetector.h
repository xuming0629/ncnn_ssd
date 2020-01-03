#pragma once

#include <iostream>
#include <memory>
#include "BaseDetector.h"
#include "PriorBox.h"
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
		/**
			@brief load ncnn param file and bin file
			@param paramFile [input] 
			@param binFile	 [input]
		**/
		virtual bool loadModel(const char * parmFile, const char* binFile);
		/**
			@brief ssd model forward, include forward and nms
		**/
		virtual void detector(unsigned char *pImage, int nWidth, int nHeight, const float* pMean, const float* pStd, const int dataType);
		/**
			@brief return detect result 
			@return  std::vector<detInfo>
		**/
		virtual std::vector<detInfo> getDetectInfo();
		/**
			@breif set params of nms
			@param fConfThresh [input]		thresh of cofidence  default = 0.45
			@param fnmsThresh  [input]		thresh of iou		 default = 0.5
			@param nTopK	   [input]		top number of best confidence	default=200
		**/
		void setSSDThresh(const float fConfThresh = 0.45, const float fnmsThresh=0.5, const int nTopK=200);
		
		/**
			@brief set  threads for speed up
			@param  nNumThreads
		**/
		void setNumTheads(const int nNumTheads);

	private:
		std::vector<detInfo> m_vecDet;				// object result
		std::shared_ptr <ncnn::Net> m_SSDNet;		// net
		std::shared_ptr <PriorBox> m_priorBox;		// priorbox class
		std::vector<rect> m_vecPriorBoxes;			// priorbox clas
		std::vector<float> m_vecVariance;			// porobox variance
		float m_confThresh = 0.45;					// confidence thresh
		float m_nmsThresh = 0.5;					// nms thresh
		int m_topK = 200;							// nms top 
		int m_num_threads = 1;						// ext threads for speed up

#if NCNN_VULKAN
		ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
		ncnn::PoolAllocator g_workspace_pool_allocator;
#endif //  NCNN_VULANK
	};
}

