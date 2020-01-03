#include "SSDDetector.h"
#include "box_utils.h"

namespace ncnn_det
{

	SSDDetector::SSDDetector()
	{
		 m_priorBox.reset(new PriorBox);
		 // get prior boxe
		 m_priorBox->computePriorBox(m_vecPriorBoxes);
		 m_priorBox->getVariance(m_vecVariance);
#if NCNN_VULKAN 
		 ncnn::create_gpu_instance();
#endif
	}


	SSDDetector::~SSDDetector()
	{
		m_SSDNet->clear();
		m_SSDNet.reset();
		m_priorBox.reset();
#if NCNN_VULKAN
		ncnn::destroy_gpu_instance();
#endif
	}
	bool SSDDetector::loadModel(const char * parmFile, const char* binFile)
	{
		if(!m_SSDNet)
		{
			m_SSDNet.reset(new ncnn::Net);
		}
		ncnn::Option opt;
		opt.lightmode = true;
		opt.num_threads = m_num_threads;

#if NCNN_VULKAN
		opt.blob_allocator = &g_blob_pool_allocator;
		opt.workspace_allocator = &g_workspace_pool_allocator;
		// use vulkan compute
		if (ncnn::get_gpu_count() != 0)
			opt.use_vulkan_compute = 1;
#endif // NCNN_VULKAN

		m_SSDNet->opt = opt;

		if (m_SSDNet->load_param(parmFile) || m_SSDNet->load_model(binFile))
		{
			printf(" load model falsed!!!, param or bin file is not exit or bad\n");
			return false;
			
		}
		return true;

	}

	void SSDDetector::detector(unsigned char *pImage, int nWidth, int nHeight, const float* pMean, const float* pStd, const int dataType)
	{
		if (!pImage)
		{
			printf("image data is empty!!!\n");
			return;
		}
		ncnn::Mat in = ncnn::Mat::from_pixels(pImage, ncnn::Mat::PIXEL_BGR, nWidth, nHeight);
#ifdef DEBUG
		printf("input size %d %d %d\n", in.c, in.w, in.h);
#endif // DEBUG
		if (pStd == NULL) 
		{
			in.substract_mean_normalize(pMean, 0);
		}
		else
		{
			in.substract_mean_normalize(pMean, pStd);
		}
		

		ncnn::Mat matCls;
		ncnn::Mat matReg;
#ifdef DEBUG
		printf("theads num: %d\n ", m_num_threads);
#endif // DEBUG
		ncnn::Extractor ext = m_SSDNet->create_extractor();
		// ext.set_num_threads(m_num_threads);
		// ext.set_light_mode(true);
#if NCNN_VULKAN
		ext.set_vulkan_compute(true);
#endif // NCNN_VULKAN

		ext.input("input", in);
		ext.extract("reg", matReg);
		ext.extract("cls", matCls);
#ifdef  DEBUG
		printf("cls shape: c:%d  height: %d  width: %d \n", matCls.c, matCls.h, matCls.w);
		printf("reg shape: c:%d  height: %d  width: %d \n", matReg.c, matReg.h, matReg.w);
#endif //  DEBUG

		ncnn::Mat matOutReg = matReg.reshape(matReg.c * matReg.h * matReg.w);
		ncnn::Mat matOutCls = matCls.reshape(matCls.c * matCls.h * matCls.w);
		
		std::vector<float> vecPreCls;
		std::vector<detInfo>vecDelta;
		std::vector<detInfo> vecOutInfo;
		
		for (int i = 0; i < matReg.h; i++)
		{
			rect preReg = { matOutReg[i*4], matOutReg[i*4 +1], matOutReg[i*4 +2], matOutReg[i*4 +3] };
			detInfo detReg;
			detReg.box = preReg;
			vecDelta.push_back(detReg);
		}
		delta2Box(vecDelta, m_vecPriorBoxes, m_vecVariance, vecDelta);
		for(int i = 1; i < CLASS_NUM; i++ )
		{
			std::vector<detInfo> vecDetCls;
			for (int j = 0; j < matCls.h; j++)
			{
				if (matOutCls[i + j * CLASS_NUM] < m_confThresh)
				{
					continue;
				}
				detInfo detCls = vecDelta[j];
				detCls.score = matOutCls[i + j * CLASS_NUM];
				detCls.label = i;
				vecDetCls.push_back(detCls);
			}
			
			nms_ssd(vecDetCls, vecOutInfo);
		}
		m_vecDet.resize(int(vecOutInfo.size()));
		for (int i = 0; i < int(vecOutInfo.size()); i++)
		{
			m_vecDet[i] = vecOutInfo[i];
		}
	}
	std::vector<detInfo> SSDDetector::getDetectInfo()
	{
		return  m_vecDet;
	}

	void SSDDetector::setSSDThresh(const float fConfThresh, const float fnmsThresh, const int nTopK)
	{
		m_confThresh = fConfThresh;
		m_nmsThresh = fnmsThresh;
		m_topK = nTopK;
	}
	void SSDDetector::setNumTheads(const int nNumTheads)
	{
		if (nNumTheads < 0)
		{
			printf("numbers of thead is mask >=1 !!!\n");
			return;
		}
		m_num_threads = nNumTheads;
	}
}
