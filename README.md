# Face Recognition Based on Depth Image and SeetaFace Engine

# Description
基于艾芯相机获取到的深度图像，利用区域生长算法及中科院山世光老师开源的Seetaface人脸识别引擎的人脸识别跟随程序。（很大程度上能避免图像和视频的误识别）
# Environmet

Windows 10

VS2013

Opencv2.4.13

# Contents
* 获取 OpenNI 的相机数据流
* 调用opencv转化成Mat类型数据
* 利用区域生长算法捕获疑似人脸区域
* 通过Seetaface人脸识别引擎进行再次判别，得到最终人脸图像，为感兴趣区域单独显示。

# Result

![image](https://github.com/dongmuliu/FaceRecognition/blob/master/image.gif)   
