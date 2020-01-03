# ncnn_ssd
ssd.pytorch to ncnn C++

[pytorch ssd](https://github.com/jmu201521121021/ssd.pytorch) fork from [pytorch.ssd](https://github.com/amdegroot/ssd.pytorch)

## 软件环境

- windows 10


- visual studio 2017

- ncnn （cpu or vulkan）

- opencv 3.0

- VulkanSDK-1.1.92.1(option)

  **Download ThirdParty(opencv ncnn vulkan) , please check here [BaiduDrive](https://pan.baidu.com/s/19m3mRaEaRWIFFalVmRyTTw ) 提取码：n3jl , finish ,**move   ThirdParty file to $<ncnn-ssd-root>$

## demo

- load model

  ```c++
  char parmFile[] = "../models/ssd/ssd_vgg300.param";
  char binFile[] = "../models/ssd/ssd_vgg300.bin";
  SSDDetector ssdDet;
  ssdDet.loadModel(parmFile, binFile);
  ```

  ​	


- 设置线程

  ```c++
  ssdDet.setNumTheads(1);
  ```

  ​

- forward

  ```c++
  const float fMean[3] = { 104.0f, 117.0f, 123.0f };
  double start_time = ncnn::get_current_time();
  ssdDet.detector(img.data, img.cols, img.rows, fMean, NULL, 0);
  double end_time = ncnn::get_current_time();
  ```

## Model

- move L2Norm layer and finetuen mAP=74.9 ,download ncnn model ,please check here [[BaiduDrive]](https://pan.baidu.com/s/1KTzZ1Jgr8g9mbPc6QXpT3g ) ,提取码：nd9w

## TODO

- [x] Support gpu
- [ ] Ssd support android 

##  实验结果

## ![demo](https://github.com/jmu201521121021/ncnn_ssd/raw/develop/data/demo.jpg)

![demo](https://github.com/jmu201521121021/ncnn_ssd/raw/develop/data/demo_1.jpg)

