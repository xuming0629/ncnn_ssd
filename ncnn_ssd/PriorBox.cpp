#include "PriorBox.h"
#include <math.h>

namespace ncnn_det
{
	PriorBox::PriorBox()
	{
	}


	PriorBox::~PriorBox()
	{

	}
	void PriorBox::setConfig(ssd_conf config)
	{
		cfg = config;
	}
	void PriorBox::getVariance(std::vector<float> &vecVariance)
	{
		vecVariance = cfg.vecVariance;

	}
	rect* PriorBox::computePriorBox(int &nCount)
	{
		int image_size = cfg.nMinDim;
		if(! m_vecDeltaBox.empty())
		{
			m_vecDeltaBox.clear();
		}
		for (int k = 0; k < cfg.vecFeatureMap.size(); k++)
		{
			const int nFMSize = cfg.vecFeatureMap[k];
			const float fFK = float(image_size) / float(cfg.vecStep[k]);
			for (int i = 0; i < nFMSize; i++)
			{
				for (int j = 0; j < nFMSize; j++)
				{
					float fCx = float(j + 0.5) / fFK;
					float fCy = float(i + 0.5) / fFK;

					fCx = std::fmaxf(fCx, 0.0f);
					fCy = std::fmaxf(fCy, 0.0f);

					fCx = std::fminf(fCx, 1.0f);
					fCy = std::fminf(fCy, 1.0f);
					
					// ratio = 1 mix size
					int nMinSizeK = cfg.vecMinSizes[k];
					float fMinWH = float(nMinSizeK) / float(image_size);
					fMinWH = std::fmaxf(fMinWH, 0.0f);
					fMinWH = std::fminf(fMinWH, 1.0f);

					rect rec = {fCx, fCy, fMinWH, fMinWH};
					m_vecDeltaBox.push_back(rec);
					
					// ratio = 1 min-max
					float fMinMaxWH = sqrtf(float(fMinWH) * (float(cfg.vecMaxSizes[k]) / float(image_size)));
					
					fMinMaxWH = std::fmaxf(fMinMaxWH, 0.0f);
					fMinMaxWH = std::fminf(fMinMaxWH, 1.0f);

					rec = { fCx, fCy, fMinMaxWH, fMinMaxWH };
					m_vecDeltaBox.push_back(rec);

					for (int m = 0; m < cfg.vecAspectRatios[k].size(); m++)
					{
						float fRatio = sqrtf(float(cfg.vecAspectRatios[k][m]));
						rec = { fCx, fCy,  std::fminf(std::fmaxf(fRatio * fMinWH, 0.0f), 1.0f),  std::fminf(std::fmaxf(fMinWH / fRatio, 0.0f), 1.0f) };
						m_vecDeltaBox.push_back(rec);
						rec = { fCx, fCy,  std::fminf(std::fmaxf(fMinWH / fRatio, 0.0f), 1.0f),  std::fminf(std::fmaxf(fMinWH * fRatio, 0.0f), 1.0f) };
						m_vecDeltaBox.push_back(rec);
					}

				}
			}
		}
		nCount = int(m_vecDeltaBox.size());
		return &m_vecDeltaBox.front();
	}

	void PriorBox::computePriorBox(std::vector<rect> &vecPriorBox)
	{
		int image_size = cfg.nMinDim;
		if (!vecPriorBox.empty())
		{
			vecPriorBox.clear();
		}
		for (int k = 0; k < cfg.vecFeatureMap.size(); k++)
		{
			const int nFMSize = cfg.vecFeatureMap[k];
			const float fFK = float(image_size) / float(cfg.vecStep[k]);
			for (int i = 0; i < nFMSize; i++)
			{
				for (int j = 0; j < nFMSize; j++)
				{
					float fCx = float(j + 0.5) / fFK;
					float fCy = float(i + 0.5) / fFK;

					fCx = std::fmaxf(fCx, 0.0f);
					fCy = std::fmaxf(fCy, 0.0f);

					fCx = std::fminf(fCx, 1.0f);
					fCy = std::fminf(fCy, 1.0f);

					// ratio = 1 mix size
					int nMinSizeK = cfg.vecMinSizes[k];
					float fMinWH = float(nMinSizeK) / float(image_size);
					fMinWH = std::fmaxf(fMinWH, 0.0f);
					fMinWH = std::fminf(fMinWH, 1.0f);

					rect rec = { fCx, fCy, fMinWH, fMinWH };
					vecPriorBox.push_back(rec);

					// ratio = 1 min-max
					float fMinMaxWH = sqrtf(float(fMinWH) * (float(cfg.vecMaxSizes[k]) / float(image_size)));

					fMinMaxWH = std::fmaxf(fMinMaxWH, 0.0f);
					fMinMaxWH = std::fminf(fMinMaxWH, 1.0f);

					rec = { fCx, fCy, fMinMaxWH, fMinMaxWH };
					vecPriorBox.push_back(rec);

					for (int m = 0; m < cfg.vecAspectRatios[k].size(); m++)
					{
						float fRatio = sqrtf(float(cfg.vecAspectRatios[k][m]));
						rec = { fCx, fCy,  std::fminf(std::fmaxf(fRatio * fMinWH, 0.0f), 1.0f),  std::fminf(std::fmaxf(fMinWH / fRatio, 0.0f), 1.0f) };
						vecPriorBox.push_back(rec);
						rec = { fCx, fCy,  std::fminf(std::fmaxf(fMinWH / fRatio, 0.0f), 1.0f),  std::fminf(std::fmaxf(fMinWH * fRatio, 0.0f), 1.0f) };
						vecPriorBox.push_back(rec);
					}

				}
			}
		}
	}



}
