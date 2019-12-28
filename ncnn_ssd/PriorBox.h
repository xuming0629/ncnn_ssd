#pragma once
#include "common.h"
namespace ncnn_det
{
	class PriorBox
	{
	public:
		PriorBox();
		
		~PriorBox();
		void setConfig(ssd_conf config);
		rect * computePriorBox(int &nCount);
	private:
		ssd_conf cfg;
		std::vector<rect> m_vecDeltaBox;
	};
}

