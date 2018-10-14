/*****************************************************************************
*                                                                            *
*  OpenNI 2.x Alpha                                                          *
*  Copyright (C) 2012 PrimeSense Ltd.                                        *
*                                                                            *
*  This file is part of OpenNI.                                              *
*                                                                            *
*  Licensed under the Apache License, Version 2.0 (the "License");           *
*  you may not use this file except in compliance with the License.          *
*  You may obtain a copy of the License at                                   *
*                                                                            *
*      http://www.apache.org/licenses/LICENSE-2.0                            *
*                                                                            *
*  Unless required by applicable law or agreed to in writing, software       *
*  distributed under the License is distributed on an "AS IS" BASIS,         *
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
*  See the License for the specific language governing permissions and       *
*  limitations under the License.                                            *
*                                                                            *
*****************************************************************************/

#include <OpenNI.h>
#include "Viewer.h"
#include "Capture.h"
#include "AXonLink.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include "face_detection.h"
#include <iostream>

#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
using namespace openni;
using namespace cv;


Mat image_binary;

Mat img;
Mat img_gray;
Rect face_rect;
int32_t num_face = 0;
seeta::ImageData img_data;
Mat imgdepth;
double secs = 0;
vector<seeta::FaceInfo> faces;
bool depth_img_isvalid = false;
bool color_img_isvalid = false;
bool ir_img_isvalid = false;
bool flag = 0;

int leftdepth = 0;
int middepth = 0;
int rightdepth = 0;
int checkflatresult = 1;
int checkdephtresult = 0;

Mat gray2rainbow(const Mat& scaledGray, int min, int max);
Mat gray2rainbowTestHist(const Mat& scaledGray, int min, int max);
Mat ImageIrCheck;
Mat ImageIrSend;
bool get_true() { return true; }

Capture capture = Capture();


int main(int argc, char** argv)
{
	openni::Status rc = openni::STATUS_OK;

	openni::Device device;
	openni::VideoStream depth, color, ir;


	//��ȡrgb
	seeta::FaceDetection detector("D:/face recognition/FaceDetection/x64/Debug/seeta_fd_frontal_v1.0.bin");
	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);

	int nResolutionColor = 0;
	int nResolutionDepth = 0;
	int lastResolutionX = 0;
	int lastResolutionY = 0;
	const char* deviceURI = openni::ANY_DEVICE;

	if (argc > 1)
	{
		deviceURI = argv[1];
	}

	rc = openni::OpenNI::initialize();
	printf("After initialization:\n%s\n", openni::OpenNI::getExtendedError());

	rc = device.open(deviceURI);                /*�������ʶ��*/
	if (rc != openni::STATUS_OK)
	{
		printf("SimpleViewer: Device open failed:\n%s\n", openni::OpenNI::getExtendedError());
		openni::OpenNI::shutdown();            /*�ر�openni*/
		return 1;
	}

	//--��ȡ�����汾��---------------------------------------
	OniVersion drver;
	int nsize;
	nsize = sizeof(drver);
	device.getProperty(ONI_DEVICE_PROPERTY_DRIVER_VERSION, &drver, &nsize);    /*��ȡ�����汾��*/
	printf("AXon driver version V%d.%d.%d.%d\n", drver.major, drver.minor,
		drver.maintenance, drver.build);

	// get color sensor
	const openni::SensorInfo* info = device.getSensorInfo(openni::SENSOR_COLOR);
	if (info)
	{
		for (int i = 0; i < info->getSupportedVideoModes().getSize(); i++)
		{
			printf("Color info : videomode %d %dx%d FPS %d f %d\n", i,
				info->getSupportedVideoModes()[i].getResolutionX(),
				info->getSupportedVideoModes()[i].getResolutionY(),
				info->getSupportedVideoModes()[i].getFps(),
				info->getSupportedVideoModes()[i].getPixelFormat());
			if ((info->getSupportedVideoModes()[i].getResolutionX() != lastResolutionX) || (info->getSupportedVideoModes()[i].getResolutionY() != lastResolutionY))
			{
				nResolutionColor++;
				lastResolutionX = info->getSupportedVideoModes()[i].getResolutionX();
				lastResolutionY = info->getSupportedVideoModes()[i].getResolutionY();
			}
		}
	}
	lastResolutionX = 0;
	lastResolutionY = 0;
	// get depth sensor
	const openni::SensorInfo* depthinfo = device.getSensorInfo(openni::SENSOR_DEPTH);
	if (depthinfo)
	{

		for (int i = 0; i < depthinfo->getSupportedVideoModes().getSize(); i++)
		{
			printf("\nDepth info: videomode %d %dx%d Fps %d f %d\n", i,                          //�ᱻִ�����
				depthinfo->getSupportedVideoModes()[i].getResolutionX(),
				depthinfo->getSupportedVideoModes()[i].getResolutionY(),
				depthinfo->getSupportedVideoModes()[i].getFps(),
				depthinfo->getSupportedVideoModes()[i].getPixelFormat());
			if ((depthinfo->getSupportedVideoModes()[i].getResolutionX() != lastResolutionX) || (depthinfo->getSupportedVideoModes()[i].getResolutionY() != lastResolutionY))
			{
				nResolutionDepth++;
				lastResolutionX = depthinfo->getSupportedVideoModes()[i].getResolutionX();
				lastResolutionY = depthinfo->getSupportedVideoModes()[i].getResolutionY();
			}
		}
	}

	//------------ Creat depth stream ---------------------------------------------------
	rc = depth.create(device, openni::SENSOR_DEPTH);
	if (rc == openni::STATUS_OK)
	{
		rc = depth.start();                       // start depth 
		if (rc != openni::STATUS_OK)
		{
			printf("SimpleViewer: Couldn't start depth stream:\n%s\n", openni::OpenNI::getExtendedError());
			depth.destroy();
		}
	}
	else
	{
		printf("SimpleViewer: Couldn't find depth stream:\n%s\n", openni::OpenNI::getExtendedError());
	}

	//-------------- Creat Color stream ----------------------------------------------------
	rc = color.create(device, openni::SENSOR_COLOR);
	if (rc == openni::STATUS_OK)
	{
		openni::VideoMode vm;
		vm = color.getVideoMode();
		vm.setResolution(1280, 960);
		color.setVideoMode(vm);
		rc = color.start();  // start color
		if (rc != openni::STATUS_OK)
		{
			printf("SimpleViewer: Couldn't start color stream:\n%s\n", openni::OpenNI::getExtendedError());
			color.destroy();
		}
	}
	else
	{
		printf("SimpleViewer: Couldn't find color stream:\n%s\n", openni::OpenNI::getExtendedError());
	}

	AXonLinkCamParam camParam;
	int dataSize = sizeof(AXonLinkCamParam);
	device.getProperty(AXONLINK_DEVICE_PROPERTY_GET_CAMERA_PARAMETERS, &camParam, &dataSize);
	/*for (int i = 0; i < nResolutionColor; i++)
	{
	printf("astColorParam x =%d\n", camParam.astColorParam[i].ResolutionX);
	printf("astColorParam y =%d\n", camParam.astColorParam[i].ResolutionY);
	printf("astColorParam fx =%.5f\n", camParam.astColorParam[i].fx);
	printf("astColorParam fy =%.5f\n", camParam.astColorParam[i].fy);
	printf("astColorParam cx =%.5f\n", camParam.astColorParam[i].cx);
	printf("astColorParam cy =%.5f\n", camParam.astColorParam[i].cy);
	printf("astColorParam k1 =%.5f\n", camParam.astColorParam[i].k1);
	printf("astColorParam k2 =%.5f\n", camParam.astColorParam[i].k2);
	printf("astColorParam p1 =%.5f\n", camParam.astColorParam[i].p1);
	printf("astColorParam p2 =%.5f\n", camParam.astColorParam[i].p2);
	printf("astColorParam k3 =%.5f\n", camParam.astColorParam[i].k3);
	printf("astColorParam k4 =%.5f\n", camParam.astColorParam[i].k4);
	printf("astColorParam k5 =%.5f\n", camParam.astColorParam[i].k5);
	printf("astColorParam k6 =%.5f\n", camParam.astColorParam[i].k6);
	}
	for (int i = 0; i < nResolutionDepth; i++)
	{
	printf("astDepthParam x =%d\n", camParam.astDepthParam[i].ResolutionX);
	printf("astDepthParam y =%d\n", camParam.astDepthParam[i].ResolutionY);
	printf("astDepthParam fx =%.5f\n", camParam.astDepthParam[i].fx);
	printf("astDepthParam fy =%.5f\n", camParam.astDepthParam[i].fy);
	printf("astDepthParam cx =%.5f\n", camParam.astDepthParam[i].cx);
	printf("astDepthParam cy =%.5f\n", camParam.astDepthParam[i].cy);
	printf("astDepthParam k1 =%.5f\n", camParam.astDepthParam[i].k1);
	printf("astDepthParam k2 =%.5f\n", camParam.astDepthParam[i].k2);
	printf("astDepthParam p1 =%.5f\n", camParam.astDepthParam[i].p1);
	printf("astDepthParam p2 =%.5f\n", camParam.astDepthParam[i].p2);
	printf("astDepthParam k3 =%.5f\n", camParam.astDepthParam[i].k3);
	printf("astDepthParam k4 =%.5f\n", camParam.astDepthParam[i].k4);
	printf("astDepthParam k5 =%.5f\n", camParam.astDepthParam[i].k5);
	printf("astDepthParam k6 =%.5f\n", camParam.astDepthParam[i].k6);
	}
	printf("R = %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n", camParam.stExtParam.R_Param[0], camParam.stExtParam.R_Param[1], camParam.stExtParam.R_Param[2], camParam.stExtParam.R_Param[3], camParam.stExtParam.R_Param[4], camParam.stExtParam.R_Param[5], camParam.stExtParam.R_Param[6], camParam.stExtParam.R_Param[7], camParam.stExtParam.R_Param[8]);
	printf("T = %.5f %.5f %.5f \n", camParam.stExtParam.T_Param[0], camParam.stExtParam.T_Param[1], camParam.stExtParam.T_Param[2]);*/

	//--------------Creat Ir stream--------------------------------------------------
	rc = ir.create(device, openni::SENSOR_IR);
	if (rc == openni::STATUS_OK)
	{
		rc = ir.start();   // start IR 
		if (rc != openni::STATUS_OK)
		{
			printf("SimpleViewer: Couldn't start ir stream:\n%s\n", openni::OpenNI::getExtendedError());
			ir.destroy();
		}
	}
	else
	{
		printf("SimpleViewer: Couldn't find ir stream:\n%s\n", openni::OpenNI::getExtendedError());
	}
	AXonLinkGetExposureLevel value;
	int nSize = sizeof(value);
	ir.getProperty(AXONLINK_STREAM_PROPERTY_EXPOSURE_LEVEL, &value, &nSize);

	printf("\nGet level:custId=%d,max=%d,current=%d\n\n", value.customID, value.maxLevel, value.curLevel);     //�ᱻִ�����


	if (!depth.isValid() || !color.isValid() || !ir.isValid())
	{
		printf("SimpleViewer: No valid streams. Exiting\n");
		openni::OpenNI::shutdown();
		return 2;
	}
	rc = device.setDepthColorSyncEnabled(true);
	if (rc != openni::STATUS_OK)
	{
		printf("start sync failed1\n");
		openni::OpenNI::shutdown();
		return 4;
	}

	//--------------------------------------------
	int changedIndex;
	openni::VideoStream**	m_streams;

	m_streams = new openni::VideoStream*[3];
	m_streams[0] = &depth;
	m_streams[1] = &color;
	m_streams[2] = &ir;
	VideoFrameRef  frameDepth;
	VideoFrameRef  frameColor;
	VideoFrameRef  frameIr;

	cv::Mat mScaledDepth;
	cv::Mat mRgbDepth;
	cv::Mat mScaledIr;
	cv::Mat mImageDepth;
	cv::Mat mImageRGB;
	cv::Mat mImageIr;

	cv::Mat cImageBGR;
	cv::Mat cImageIr;

	cv::Mat mFaceImg;

	int depthunit;

	// ���������ֵ
	int iMaxDepth;
	int iMaxIr;

	Vec3b colorpix = Vec3b(255, 255, 0);
	Vec3b color_value;

	Mat irImage8UC1;
	Mat mImageScale;
	Mat imagedepthROI;
	Mat imageROI;
	Mat imagecolorROI;
	int readIndex = 0;
	int valueCount = 0;     //֡�� ֡������2ʱ��ȡ

	Rect faceRect;			//������
	Rect faceRectcolor;     //��ɫͼ��������
	int phhist[256];	    //��ֱͶӰͳ��
	int pvhist[256];	    //ˮƽͶӰͳ��
	//-------------------------
	// main loop 
	//-------------------------
	while (get_true())
	{
		rc = openni::OpenNI::waitForAnyStream(m_streams, 3, &changedIndex);
		if (rc != openni::STATUS_OK)
		{
			printf("Wait failed\n");
			return 1;
		}

		switch (changedIndex)
		{
		case 0:
			depth.readFrame(&frameDepth);                    //�������֡����
			cout << "\n\n��ǰ��ȡ���֡��Ϣ���£�" << endl;
			printf("[%08llu] depth %d\n", (long long)frameDepth.getTimestamp(), frameDepth.getFrameIndex());
			readIndex++;
			depth_img_isvalid = true;

			depthunit = frameDepth.getVideoMode().getPixelFormat();
			iMaxDepth = depth.getMaxPixelValue();
			//cout << "������ֵΪ��" << iMaxDepth << endl;     //����һ����ֵ 4096

			mImageDepth = cv::Mat(frameDepth.getHeight(), frameDepth.getWidth(), CV_16UC1, (void*)frameDepth.getData());
			//cout << "ͼ��Ŀ�Ϊ��" << mImageDepth.rows << "��"
			//	<< " ͼ��ĸ�Ϊ��" << mImageDepth.cols<<"��"<< endl;
			//cout << "�����Ϣ��ԭʼֵ\n"<< mImageDepth << endl;
			//imshow("srcDepth Image", mImageDepth);                  //mImageDepthΪCV_16UC1ͼƬ

			mRgbDepth = gray2rainbow(mImageDepth, 600, 3600);         //��mImageDepth��ɿ��Ƶ����ͼ(α��ɫͼ)
			//cout << "��һ����� flag:" << flag << endl;
			if (true == capture.findobj(mImageDepth))                 // use orignal depth data
			{
				//capture.isface = true;                 // if no fake result later, keep isface true
				flag = 1;                                //����Ƿ��ҵ�����(����ʱ��flagΪ0)
				if ((true == color_img_isvalid) && (true == ir_img_isvalid))
				{
					faceRect = capture.checkface();                 //faceRect����ֵΪ�沿�ľ���
					ImageIrCheck = Mat(ImageIrSend, faceRect);
					cv::resize(ImageIrCheck, ImageIrCheck, Size(160, 220));
					//imshow("test", ImageIrCheck);
					//if (ImageIrCheck.channels() != 1)
					//	cvtColor(ImageIrCheck, img_gray, cv::COLOR_BGR2GRAY);
					//else
					img_gray = ImageIrCheck;


					img_data.data = img_gray.data;
					img_data.width = img_gray.cols;
					img_data.height = img_gray.rows;
					img_data.num_channels = 1;

					// long t0 = getTickCount();
					faces = detector.Detect(img_data);
					// long t1 = getTickCount();
					// secs = (t1 - t0) / getTickFrequency():


					num_face = static_cast<int32_t>(faces.size());
					if (num_face != 0)                            //�ж��Ƿ�������
					{
						cout << "������" << endl;
						rectangle(mRgbDepth, faceRect, Scalar(255));    //��α��ɫͼ�Ͽ���������
						cout << "�������ɹ���������Depth Image..." << endl;
						namedWindow("Depth Image");
						moveWindow("Depth Image", 70, 420);
						imshow("Depth Image", mRgbDepth);

						// choice the face-like area and show
						imagedepthROI = Mat(mImageDepth, faceRect);         //��ȡ��mImageDepthͼ�е�faceRect�������
						imagedepthROI = capture.get3dface(imagedepthROI);   //ȥ��mImageDepthͼ��rect���еķ��������֣�ֻ������������

						// change CV_16UC1 depth to RGB image
						Point imagedepthROIpointmin = capture.getGrownImage_minPoint();
						Point  imagedepthROIpointmax = capture.getGrownImage_maxPoint();
						int imagedepthROImin = mImageDepth.at<ushort>(imagedepthROIpointmin);
						int imagedepthROImax = mImageDepth.at<ushort>(imagedepthROIpointmax);

						capture.getdepthHist(imagedepthROI, imagedepthROImin, imagedepthROImax, phhist, pvhist);   //�������ֱ��ͼ
						imagedepthROI = gray2rainbow(imagedepthROI, imagedepthROImin, imagedepthROImax);
						// scaling roi depth img for 3D face Identification
						resize(imagedepthROI, imagedepthROI, Size(160, 220));
					}
					else
					{
						cout << "���ж�������" << endl;
						cv::imshow("Depth Image", mRgbDepth);
						//imagedepthROI = Mat::zeros(Size(160, 220), CV_8U);
						imageROI = Mat::zeros(Size(160, 220), CV_8U);
						flag = 0;
					}
				}
				else
				{
					cout << "\n �������ɹ� ��color or irͼ������һ������Ч��(��ʼ����...)" << endl;
					namedWindow("Depth Image");
					moveWindow("Depth Image", 70, 420);
					imshow("Depth Image", mRgbDepth);
					imagedepthROI = Mat::zeros(Size(160, 220), CV_8U);
					//imageROI = Mat::zeros(Size(160, 220), CV_8U);
					flag = 0;
				}
			}
			else
			{
				cout << "������ʧ��!" << endl;
				namedWindow("Depth Image");
				moveWindow("Depth Image", 70, 420);
				imshow("Depth Image", mRgbDepth);
				imagedepthROI = Mat::zeros(Size(160, 220), CV_8U);
				flag = 0;
				valueCount = 0;
			}
			if (flag == 1)           //��⵽�������������������ľֲ�ͼ
			{
				//cout << "�������ɹ���flag == 1�����)���imagedepthROI..." << endl;
				namedWindow("�������");
				moveWindow("�������", 60, 50);
				imshow("�������", imagedepthROI);
			}
			else                   //δ��⵽����������£���������ڵ�����
			{
				//cout << "(������,flag = 0��������imagedepthROI...)" << endl;
				imagecolorROI = Mat::zeros(Size(160, 220), CV_8U);
				namedWindow("�������");
				moveWindow("�������", 60, 50);
				cv::imshow("�������", imagedepthROI);
			}
			break;
		case 1:
			color.readFrame(&frameColor);
			//cout << "\t��ǰ��ȡRGB֡��Ϣ���£�" << endl;
			//printf("[%08llu] color %d\n", (long long)frameColor.getTimestamp(), frameColor.getFrameIndex());
			color_img_isvalid = true;
			//cout << "flag: " << flag << endl << endl;
			mImageRGB = cv::Mat(frameColor.getHeight(), frameColor.getWidth(), CV_8UC3, (void*)frameColor.getData());

			cv::cvtColor(mImageRGB, cImageBGR, CV_RGB2BGR);
			resize(cImageBGR, cImageBGR, Size(640, 480));
			if (flag == 1)
			{
				faceRect = capture.checkface();
				faceRectcolor = Rect(faceRect);

				//�����������ڲ�ɫͼ�Ͽ�������
				faceRectcolor.x -= 45;
				rectangle(cImageBGR, faceRectcolor, Scalar(255));
				namedWindow("Color Image");
				//moveWindow("Color Image", 1300, 50);
				imshow("Color Image", cImageBGR);

				imagecolorROI = Mat(cImageBGR, faceRectcolor);
				resize(imagecolorROI, imagecolorROI, Size(160, 220));
			}
			else
			{
				imagecolorROI = Mat::zeros(Size(160, 220), CV_8U);
				namedWindow("Color Image");
				//moveWindow("Color Image", 1300, 50);
				cv::imshow("Color Image", cImageBGR);
			}
			namedWindow("��ɫͼ��");
			moveWindow("��ɫͼ��", 410, 50);
			cv::imshow("��ɫͼ��", imagecolorROI);
			cv::waitKey(10);


			break;
		case 2:
			ir.readFrame(&frameIr);
			//cout << "\n\t��ǰ��ȡir֡" << endl;
			//printf("[%08llu] ir %d\n", (long long)frameIr.getTimestamp(), frameIr.getFrameIndex());
			ir_img_isvalid = true;
			//cout << "flag: " << flag << endl;
			iMaxIr = ir.getMaxPixelValue();
			//	cout << "iMaxIr��" << iMaxIr << endl;
			mImageIr = cv::Mat(frameIr.getHeight(), frameIr.getWidth(), CV_8UC1, (void *)frameIr.getData());
			ImageIrSend = mImageIr.clone();
			if (flag == 1)
			{
				faceRect = capture.checkface();

				rectangle(mImageIr, faceRect, Scalar(255));
				namedWindow("Ir Image");
				moveWindow("Ir Image", 720, 420);
				cv::imshow("Ir Image", mImageIr);
				imageROI = Mat(mImageIr, faceRect);
				cv::resize(imageROI, imageROI, Size(160, 220));
			}
			else
			{
				imageROI = Mat::zeros(Size(160, 220), CV_8U);
				namedWindow("Ir Image");
				moveWindow("Ir Image", 720, 420);
				cv::imshow("Ir Image", mImageIr);
			}
			namedWindow("�Ҷ�ͼ��");
			moveWindow("�Ҷ�ͼ��", 240, 50);
			cv::imshow("�Ҷ�ͼ��", imageROI);
			cv::waitKey(10);

			break;
		default:
			printf("Error in wait\n");
		}

		//-----------------------------------------------------
		// ��ֹ��ݼ�
		if (cv::waitKey(10) == 'q')
			break;
	}
	// �ر�������
	depth.destroy();
	color.destroy();
	ir.destroy();
	// �ر��豸
	device.close();
	// ���ر�OpenNI
	openni::OpenNI::shutdown();
	return 0;
}

//�� CV_16UC1 ���ͼ ת����α��ɫͼ
Mat gray2rainbow(const Mat& scaledGray, int min, int max)
{
	Mat outputRainbow(scaledGray.size(), CV_8UC3);       //��ʼ����һ��outputRainbow�Ĳ�ɫͼ��
	unsigned short grayValue;
	float tempvalue;

	float par = (float)255 / (max - min);


	for (int y = 0; y < scaledGray.rows; y++)
		for (int x = 0; x < scaledGray.cols; x++)
		{

			grayValue = scaledGray.at<ushort>(y, x);
			if ((grayValue > 0) && (grayValue < min))        //���ܻ�����ҵ���min��������������Сֵ
			{
				tempvalue = (float)min;
			}
			else if (grayValue > max)                     //Ҳ���ܻ�����ҵ���max���������������ֵ
			{
				tempvalue = 0;
			}
			else
			{
				tempvalue = (float)(grayValue - min);
			}
			tempvalue = tempvalue*par;          //Ϊ�˰����ֵ�滮��(0~255֮��)
			/*
			* color    R   G   B   gray
			* red      255 0   0   255
			* orange   255 127 0   204
			* yellow   255 255 0   153
			* green    0   255 0   102
			* cyan     0   255 255 51
			* blue     0   0   255 0
			*/

			Vec3b& pixel = outputRainbow.at<Vec3b>(y, x);
			tempvalue = 256 - tempvalue;

			if ((tempvalue <= 0) || (tempvalue >= 255))
			{
				pixel[0] = 0;
				pixel[1] = 0;
				pixel[2] = 0;
			}
			else if (tempvalue <= 51)
			{
				pixel[0] = 255;
				pixel[1] = (unsigned char)(tempvalue * 5);
				pixel[2] = 0;
			}
			else if (tempvalue <= 102)
			{
				tempvalue -= 51;
				pixel[0] = 255 - (unsigned char)(tempvalue * 5);
				pixel[1] = 255;
				pixel[2] = 0;
			}
			else if (tempvalue <= 153)
			{
				tempvalue -= 102;
				pixel[0] = 0;
				pixel[1] = 255;
				pixel[2] = (unsigned char)(tempvalue * 5);
			}
			else if (tempvalue <= 204)
			{
				tempvalue -= 153;
				pixel[0] = 0;
				pixel[1] = 255 - static_cast<unsigned char>(tempvalue * 128.0 / 51 + 0.5);
				pixel[2] = 255;
			}
			else if (tempvalue < 255)
			{
				tempvalue -= 204;
				pixel[0] = 0;
				pixel[1] = 127 - static_cast<unsigned char>(tempvalue * 127.0 / 51 + 0.5);
				pixel[2] = 255;
			}
		}

	return outputRainbow;
}