#include "SSDDetector.h"
#include "box_utils.h"
namespace ncnn_det
{

	SSDDetector::SSDDetector()
	{
		 m_SSDNet.reset(new ncnn::Net);
		 m_priorBox.reset(new PriorBox);
		//  m_matInput.reset(new ncnn::Mat);

		 // get prior boxe
		 m_priorBox.get()->computePriorBox(m_vecPriorBoxes);
	}


	SSDDetector::~SSDDetector()
	{
		m_SSDNet.get()->clear();
		m_SSDNet.reset();
		m_priorBox.reset();
		
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

	void SSDDetector::detector(unsigned char *pImage, int nWidth, int nHeight, const float* pMean, const float* pStd, const int dataType)
	{
		if (!pImage)
		{
			printf("image data is empty!!!\n");
			return;
		}
		ncnn::Mat in = ncnn::Mat::from_pixels(pImage, ncnn::Mat::PIXEL_BGR, nWidth, nHeight);
		printf("input size %d %d %d\n", in.c, in.w, in.h);
		const float fMean[3] = { 104.0f, 117.0f, 123.0f };
		in.substract_mean_normalize(fMean, 0);

		ncnn::Mat matCls;
		ncnn::Mat matReg;

		ncnn::Extractor ext = m_SSDNet.get()->create_extractor();
		ext.set_light_mode(true);
		ext.input("input", in);
		ext.extract("reg", matReg);
		ext.extract("cls", matCls);
		
		printf("cls shape: c:%d  height: %d  width: %d \n", matCls.c, matCls.h, matCls.w);
		printf("reg shape: c:%d  height: %d  width: %d \n", matReg.c, matReg.h, matReg.w);

		ncnn::Mat matOutReg = matReg.reshape(matReg.c * matReg.h * matReg.w);
		std::vector<float> vecReg;
		printf("%f", matOutReg[0]);
		float *pReg = (float*)matReg.data;
		float *pCls = (float*)matCls.data;
		// printf("%f \n", *pReg);
		std::vector<rect> vecPreBoxes;
		std::vector<float> vecPreCls;
		std::vector<detInfo> vecDetInfo;
		vecDetInfo.resize(matCls.h);
		for (int i = 0; i < matReg.h; i++)
		{
			rect preReg = { pReg[0], pReg[1], pReg[2], pReg[3] };

			float  score = 0.0f;
			int label = 0;
			pCls++;
			for (int j = 1; j < CLASS_NUM; j++)
			{
				if (score < * pCls)
				{
					score = *pCls;
					label = j;
				}
			

					pCls++;
							
			}
			/*if(i < 10)
				printf("rec %f %f %f %f\n", pReg[0], pReg[1], pReg[2], pReg[3]);*/
			if (i < (matReg.h - 1))
			{
				pReg += 4;
			}
			vecPreBoxes.push_back(preReg);
			
			detInfo detSingle;
			detSingle.box = preReg;
			detSingle.label = label;
			detSingle.score = score;
			// printf("%f\n", score);
			vecDetInfo[i] = detSingle;
		}
		printf("no decode %f %f %f %f %d %f\n", vecDetInfo[0].box.left, vecDetInfo[0].box.top,
			vecDetInfo[0].box.right, vecDetInfo[0].box.bottom, vecDetInfo[0].label, vecDetInfo[0].score);
		printf("prior %f %f %f %f\n", m_vecPriorBoxes[0].left, m_vecPriorBoxes[0].top,
			m_vecPriorBoxes[0].right, m_vecPriorBoxes[0].bottom);

		printf("box size %d \n", vecPreBoxes.size());
		printf("1 detpre size %d \n", vecDetInfo.size());
		std::vector<float> vecVariance = { 0.1f, 0.2f };
		delta2Box(vecDetInfo, m_vecPriorBoxes, vecVariance, vecDetInfo);
		printf("decode %f %f %f %f %d %f\n", vecDetInfo[0].box.left , vecDetInfo[0].box.top ,
			vecDetInfo[0].box.right , vecDetInfo[0].box.bottom, vecDetInfo[0].label, vecDetInfo[0].score);
		printf("2 detpre size %d \n", vecDetInfo.size());
		std::vector<detInfo> vecOutInfo;
		for (int i = 0; i < vecDetInfo.size(); i++)
		{
			// printf("%f", vecDetInfo[i].area);

		}
		nms(vecDetInfo, vecOutInfo);
		// nms(vecOutInfo, vecOutInfo);
		printf("3 detpre size %d \n", vecOutInfo.size());
		// const int nTopK = 200;
		m_vecDet.resize(int(vecOutInfo.size()));
		for (int i = 0; i < int(vecOutInfo.size()); i++)
		{
			m_vecDet[i] = vecOutInfo[i];
		}
		for(int i =0; i < m_vecDet.size(); i++)
		{
			printf("%f %f %f %f %d %f\n", m_vecDet[0].box.left * 750, vecOutInfo[0].box.top * 1334,
				m_vecDet[0].box.right*750, m_vecDet[0].box.bottom *1334, m_vecDet[0].label, m_vecDet[0].score);
		}
		/*for (int i = 0; i < int(vecPreBoxes.size()); i++)
		{
			if(vecPreBoxes[i].left < 100)
				printf("rec1: %f %f %f %f \n", vecPreBoxes[i].left, vecPreBoxes[i].top,
				vecPreBoxes[i].right, vecPreBoxes[i].bottom);
		}*/

		/*
		

		m_vecDet.resize(matCls.h);
		for(int i = 0; i < matCls.w; i++)
		{
			std::vector<detInfo> vecDetInfo;

			float *pCls = (float*)matCls.data;
			pCls += i;
			

			for (int j = 0; j < matReg.h; j++)
			{
				
				rect preReg = { pReg[0], pReg[1], pReg[2], pReg[3] };
				if (*pCls > m_confThresh)
				{
					detInfo detSingle;
					detSingle.box = preReg;
					detSingle.score = *pCls;
					detSingle.label = i;
					vecDetInfo.push_back(detSingle);
				}
				pReg += 4;
				pCls += CLASS_NUM;
			}
			std::vector<detInfo> vecOutInfo;
			std::vector<float> vecVariance = { 0.1f, 0.2f };
			delta2Box(vecDetInfo, m_vecPriorBoxes, vecVariance, vecOutInfo);
			nms(vecOutInfo, vecOutInfo);
		}*/

	}
	std::vector<detInfo> SSDDetector::getDetectInfo()
	{
		return  m_vecDet;
	}
}
