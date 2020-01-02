# ncnn_ssd
ssd.pytorch to ncnn C++

pytorhc ssd: <https://github.com/amdegroot/ssd.pytorch>

## 软件环境

- windows 10


- visual studio 2017
- ncnn （cpu）
- opencv 3.0

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

## TODO

- [ ] Ssd support android 

##  实验结果

## ![demo](data\demo.jpg)



