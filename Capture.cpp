/*****************************************************************************
*                                                                            *
*  Copyright (C) 2018 QSVision Ltd.                                          *
*                                                                            *
*  This file is used for real human face capture                             *
*                                                                            *
*  Licensed under the Apache License, Version 2.0 (the "License");           *
*  you may not use this file except in compliance with the License.          *
*  You may obtain a copy of the License at                                   *
*                                                                            *
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
#include <string>

#include "face_detection.h"



#include <iostream>
#include <fstream>

#include "opencv2/opencv.hpp"
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Capture.h"

#include <OpenNI.h>
#include "Viewer.h"
#include "AXonLink.h"

#define facedepthmin 100
#define facedepthmax 250
int avgfacedepth;
int topdepth=0;

ofstream file("depthface_Hist_average.txt", ios::out);


//#define GL_WIN_SIZE_X	1280
//#define GL_WIN_SIZE_Y	1024
//#define TEXTURE_SIZE	512
//
//#define DEFAULT_DISPLAY_MODE	DISPLAY_MODE_DEPTH
//
//#define MIN_NUM_CHUNKS(data_size, chunk_size)	((((data_size)-1) / (chunk_size) + 1))
//#define MIN_CHUNKS_SIZE(data_size, chunk_size)	(MIN_NUM_CHUNKS(data_size, chunk_size) * (chunk_size))

Capture::Capture()
{

}
Capture::~Capture()
{


}

bool Capture::findobj(cv::Mat imgdepth)  // use orignal depth data
{
	Mat depth = imgdepth.clone();
	Mat depth_zip;
	depth.convertTo(depth_zip, CV_8U, 1.0 / 6, -170);
	//depth.convertTo(depth_zip, CV_8U, 1.0 / 6, -100);

	//���˳�����
	Mat depth_cut;
	cutImage(depth_zip, &depth_cut);
	//������鿴������Ϣ
	//namedWindow("Display");
	//setMouseCallback("Display", on_mouse, &depth_cut);
	//imshow("Display", depth_cut);


	//α��ɫͼ��
	Mat depth_rainbow;
	applyColorMap(depth_zip, depth_rainbow, COLORMAP_HSV);
	//imshow("depth_rainbow", depth_rainbow);

	//ֱ��ͼͳ��
	//myCalcHist(depth_cut, 0);

	//�ұ���
	//outstandingDetect(depth_zip, 5);

	//����ͶӰ  vertical hisgram
	Mat depth_down;
	shadowDown(depth_cut, &depth_down, 0);
	Point maxP;
	minMaxLoc(depth_down, 0, 0, 0, &maxP);
	

	//�������
	//int minLoc;
	//int min;
	//����һ��
	//drawDepthLineCol(depth_cut, maxP.x, &minLoc, &min, 0);

	//����ߵ�
	 // int topY = findTop(depth_cut, maxP.x, 10, 10);
	int topY = findtestTop(depth_cut, maxP.x, 10, 10,100);
	if (topY != -1)
	{
		//��ȡ����
		Mat depth_head;
		Point minHPt = Point();
		cutBody(depth_cut, &depth_head, topY, 100, maxP.x, 60, &minHPt);
		//circle(depth_cut, minHPt, 3, Scalar(255));

		//imshow("depth_cut", depth_cut);
		//imshow("head", depth_headpoint);


		//��������
		Rect re;
		Mat depth_grow;
		Point2i pt = Point2i(minHPt);
		Point minPt;
		Point maxPt;
		int dis_min;
		int dis_max;
		int area_th = depth_cut.at<uchar>(pt.y, pt.x)*(-103) + 22000;		//�������������ɾ���ȷ��

		int result = RegionGrow(depth_cut, &depth_grow, pt, 30, area_th, 5000, &re, &dis_min, &minPt, &dis_max, &maxPt);
		result = secondOrderCheck(depth, depth_grow, re, 5, (float)0.915,0.3f);
		if (!result)
		{
			//�������С
			
			int standardWidth = (int)(dis_min * (-0.37) + 180);
			re.x -= (standardWidth - re.width) / 2;
			int standardHeight = (int)(standardWidth * 1.3);
			re.y -= (standardHeight - re.height) / 2;
			re.height = standardHeight;
			re.width = standardWidth;

			if (re.x < depth_grow.cols / 4)
				re.x = depth_grow.cols / 4;
			if (re.x > depth_grow.cols / 4.0 * 3)
				re.x = (int)(depth_grow.cols / 4.0 * 3);

			if (re.y < 0)
				re.y = 0;
			if (re.y > depth_grow.rows - 1)
				re.y = depth_grow.rows - 1;

			if (re.x + re.width > depth_grow.cols - 1)
				re.width = depth_grow.cols - 1 - re.x;
			if (re.y + re.height > depth_grow.rows - 1)
				re.height = depth_grow.rows - 1 - re.y;
			//imshow("depth_grow", depth_grow);


			//Ѱ�ҽ������
			objface.rect = re;
			objface.mask = depth_grow.clone();
			objface.depthmax = dis_max;
			objface.depthmin = dis_min;
			
			/*avgfacedepth = (dis_max + dis_min) / 2;

			if (avgfacedepth < facedepthmin)
				cout << "����̫��������Զһ��" << endl;
			if (avgfacedepth  > facedepthmax)
				cout << "�����һ��" << endl;*/
			

			objface.minPt = Point(minPt);
			objface.maxPt = Point(maxPt);

			//rectangle(depth_zip, re, Scalar(255));

			return true;
		}

	}
	return false;
}
//������
//������ imgdepth ԭʼ���ͼ��
//����ֵ�� 

int Capture::DetectLivingBody(Mat imgdepthsolve)
{
	int mindepth=0;
	int count = 0;
	for (int i = 0; i < imgdepthsolve.cols; i++)
	{
		for (int j = 0; j <imgdepthsolve.rows; j++)
		{
			if (imgdepthsolve.at<ushort>(i, j) > imgdepthsolve.at<ushort>(i, j + 1))
			{
				mindepth = imgdepthsolve.at<ushort>(i, j + 1);
			}
		}
		if (((imgdepthsolve.at<ushort>(i, 0) - mindepth) > 5) & ((imgdepthsolve.at<ushort>(i, imgdepthsolve.cols) - mindepth) > 5))
			{
				count++;
			}
			else
			{
				count = 0;
			}
			if (count > 10)
			{
				return 1;
			}
		
	}

	return 0;
}

Rect Capture::checkface(void)
{
	if (objface.rect.width != 0 && objface.rect.height != 0)
		return objface.rect;

	return Rect();
}
cv::Mat Capture::cropir(cv::Mat imgir)
{
	cv::Mat out_ir_img = imgir.clone();


	return out_ir_img;
}

Mat Capture::getGrownImage(void)
{
	Mat dst = Mat(objface.mask, objface.rect);
	return dst.clone();
}

int Capture::getGrownImage_minval(void)
{
	return objface.depthmin;
}
int Capture::getGrownImage_maxval(void)
{

	return objface.depthmax;
}

Point Capture::getGrownImage_minPoint(void)
{
	return objface.minPt;
}
Point Capture::getGrownImage_maxPoint(void)
{
	return objface.maxPt;
}

//ɽ����
//������	src �Ҷ�ͼ��
void Capture::outstandingDetect(Mat src, int th)
{
	Mat matDst = Mat::zeros(Size(src.cols, src.rows), CV_8U);

	int max_x;
	int max_y;
	int max_val = 255;

	//�жϷ���
	int DIR[8][2] = { { -th, -th }, { 0, -th }, { th, -th }, { th, 0 }, { th, th }, { 0, th }, { -th, th }, { -th, 0 } };

	for (int i = th; i < src.rows - th; i++)
	{
		for (int j = th; j < src.cols - th; j++)
		{
			int val = src.at<uchar>(i, j);
			if (val != 0)
			{
				int k;
				for (k = 0; k < 8; k++)
				{
					int com = src.at<uchar>(i + DIR[k][0], j + DIR[k][1]);
					if (com == 0)
						break;
					if ((int)com - (int)val <= 0)
					{
						break;
					}
				}
				if (k == 8)
				{
					matDst.at<uchar>(i, j) = 255;
					if (max_val > val)
					{
						max_val = val;
						max_x = j;
						max_y = i;
					}
				}

			}
		}
	}

	circle(matDst, Point(max_x, max_y), 3, Scalar(255));

	imshow("outstanding", matDst);
}


//ƽ����
//������	src �Ҷ�ͼ��
//������	th	��ֵ
void Capture::flatDetect(Mat src, int th)
{
	Mat matDst = Mat::zeros(Size(src.cols, src.rows), CV_8U);
	//Mat matDst = src.clone();

	//�жϷ���
	int DIR[8][2] = { { -1, -1 }, { 0, -1 }, { 1, -1 }, { 1, 0 }, { 1, 1 }, { 0, 1 }, { -1, 1 }, { -1, 0 } };

	for (int i = 1; i < matDst.rows - 1; i++)
	{
		for (int j = 1; j < matDst.cols - 1; j++)
		{
			int val = src.at<uchar>(i, j);
			if (val != 0)
			{
				int k;
				for (k = 0; k < 8; k++)
				{
					int com = src.at<uchar>(i + DIR[k][0], j + DIR[k][1]);
					if (abs(com - val) > th)
					{
						break;
					}
				}
				if (k == 8)
					matDst.at<uchar>(i, j) = 255;
			}
		}
	}

	imshow("flat detect", matDst);

}


//����ͶӰ
//������	src ����Ҷ�ͼ��
//������	dst ���һάͳ������
//������	image �Ƿ���ʾͳ��ͼ
//���أ�	��
Mat Capture::shadowDown(Mat src, Mat* dst, int image)
{

	Mat shadow_down = Mat::zeros(Size(src.cols, 1), CV_16U);
	//����ͶӰ
	for (int j = 0; j < src.cols; j++)
	{
		for (int i = 0; i < src.rows; i++)
		{
			if (src.at<uchar>(i, j) != 0)
			{
				shadow_down.at<ushort>(0, j)++;
			}

		}
	}

	if (image)
	{
		//�����ȴ���һ���ڵ׵�ͼ��Ϊ�˿�����ʾ��ɫ�����Ըû���ͼ����һ��8λ��3ͨ��ͼ��    
		Mat drawImage = Mat::zeros(Size(src.size().width, src.size().height), CV_8UC3);

		for (int i = 1; i < src.cols; i++)
		{
			int value = shadow_down.at<ushort>(0, i - 1);
			line(drawImage, Point(i, drawImage.rows - 1), Point(i, drawImage.rows - 1 - value), Scalar(0, 0, 255));
		}
		line(drawImage, Point(0, drawImage.rows - 1), Point(0, drawImage.rows - 1 - 0), Scalar(0, 0, 255));
		imshow("down_shadow", drawImage);
	}


	*dst = shadow_down.clone();

	return shadow_down.clone();
}

//����ͶӰ��Ȩֵ
//������	src ����Ҷ�ͼ��
//������	dst ���һάͳ������
//������	image �Ƿ���ʾͳ��ͼ
//���أ�	��
Mat Capture::shadowDownPar(Mat src, Mat* dst, int image)
{

	Mat shadow_down = Mat::zeros(Size(src.cols, 1), CV_32S);
	//����ͶӰ
	for (int j = 0; j < src.cols; j++)
	{
		for (int i = 0; i < src.rows; i++)
		{
			if (src.at<uchar>(i, j) != 0)
			{
				shadow_down.at<int>(0, j) += (255 - src.at<uchar>(i, j));
			}

		}
		shadow_down.at<int>(0, j) /= src.cols;
	}

	if (image)
	{
		//�����ȴ���һ���ڵ׵�ͼ��Ϊ�˿�����ʾ��ɫ�����Ըû���ͼ����һ��8λ��3ͨ��ͼ��    
		Mat drawImage = Mat::zeros(Size(src.size().width, src.size().height), CV_8UC3);

		for (int i = 1; i < src.cols; i++)
		{
			int value = shadow_down.at<int>(0, i - 1);
			line(drawImage, Point(i, drawImage.rows - 1), Point(i, drawImage.rows - 1 - value), Scalar(0, 0, 255));
		}
		line(drawImage, Point(0, drawImage.rows - 1), Point(0, drawImage.rows - 1 - 0), Scalar(0, 0, 255));
		imshow("down_shadow", drawImage);
	}


	*dst = shadow_down.clone();

	return shadow_down.clone();
}

//����ͶӰ
//������	src ����Ҷ�ͼ��
//������	dst ���һάͳ������
//������	image �Ƿ���ʾͳ��ͼ
//���أ�	��
Mat Capture::shadowLeft(Mat src, Mat* dst, int image)
{

	Mat shadow_left = Mat::zeros(Size(src.rows, 1), CV_16U);
	//����ͶӰ
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<uchar>(i, j) != 0)
			{
				shadow_left.at<ushort>(0, i)++;
			}

		}
	}

	if (image)
	{
		//�����ȴ���һ���ڵ׵�ͼ��Ϊ�˿�����ʾ��ɫ�����Ըû���ͼ����һ��8λ��3ͨ��ͼ��    
		Mat drawImage = Mat::zeros(Size(src.size().width, src.size().height), CV_8UC3);

		for (int i = 1; i < src.rows; i++)
		{
			int value = shadow_left.at<ushort>(0, i - 1);
			line(drawImage, Point(drawImage.cols - 1, i), Point(drawImage.cols - 1 - value, i), Scalar(0, 0, 255));
		}
		//line(drawImage, Point(drawImage.cols - 1, 0), Point(drawImage.cols - 1 - 0, 0), Scalar(0, 0, 255));
		imshow("left_shadow", drawImage);
	}


	*dst = shadow_left.clone();

	return shadow_left.clone();
}

//����ͶӰ����Ȩ
//������	src ����Ҷ�ͼ��
//������	dst ���һάͳ������
//������	image �Ƿ���ʾͳ��ͼ
//���أ�	��
Mat Capture::shadowLeftPar(Mat src, Mat* dst, int image)
{

	Mat shadow_left = Mat::zeros(Size(src.rows, 1), CV_32S);
	//����ͶӰ
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<uchar>(i, j) != 0)
			{
				shadow_left.at<int>(0, i) += (255 - src.at<uchar>(i, j));
			}

		}
		shadow_left.at<int>(0, i) /= src.rows;
	}


	if (image)
	{
		//�����ȴ���һ���ڵ׵�ͼ��Ϊ�˿�����ʾ��ɫ�����Ըû���ͼ����һ��8λ��3ͨ��ͼ��    
		Mat drawImage = Mat::zeros(Size(src.size().width, src.size().height), CV_8UC3);

		for (int i = 1; i < src.rows; i++)
		{
			int value = shadow_left.at<int>(0, i - 1);
			line(drawImage, Point(drawImage.cols - 1, i), Point(drawImage.cols - 1 - value, i), Scalar(0, 0, 255));
		}
		//line(drawImage, Point(drawImage.cols - 1, 0), Point(drawImage.cols - 1 - 0, 0), Scalar(0, 0, 255));
		imshow("left_shadow", drawImage);
	}


	*dst = shadow_left.clone();

	return shadow_left.clone();
}

//����ֱ��ͼ
//������	imageGray �Ҷ�ͼ��
//������	isShow	-0 ������
//					-1 ����
//����ֵ��	�Ҷ�ֱ��ͼ����
MatND Capture::myCalcHist(Mat imageGray, int isShow)
{
	//����ֱ��ͼ
	int channels = 0;
	MatND dstHist;
	int histSize[] = { 256 };
	float midRanges[] = { 0, 256 };
	const float *ranges[] = { midRanges };
	calcHist(&imageGray, 1, &channels, Mat(), dstHist, 1, histSize, ranges, true, false);



	if (isShow)
	{
		//����ֱ��ͼ,�����ȴ���һ���ڵ׵�ͼ��Ϊ�˿�����ʾ��ɫ�����Ըû���ͼ����һ��8λ��3ͨ��ͼ��    
		Mat drawImage = Mat::zeros(Size(257, 257), CV_8UC3);
		//�κ�һ��ͼ���ĳ�����ص��ܸ����п��ܻ�ܶ࣬���������������ͼ��ĳߴ磬  
		//������Ҫ�ȶԸ������з�Χ�����ƣ���minMaxLoc�������õ�����ֱ��ͼ������ص�������    
		double g_dHistMaxValue;
		Point maxLoc;
		dstHist.at<float>(0) = 0;
		minMaxLoc(dstHist, 0, &g_dHistMaxValue, &maxLoc, 0);
		normalize(dstHist, dstHist, 0, 256, NORM_MINMAX, -1, Mat());
		//�����صĸ������ϵ�ͼ������Χ��    
		for (int i = 1; i < 256; i++)
		{
			//int value = cvRound(dstHist.at<float>(i) * 256 * 0.9 / g_dHistMaxValue);
			int value = (int)dstHist.at<float>(i - 1);
			line(drawImage, Point(i, drawImage.rows - 1), Point(i, drawImage.rows - 1 - value), Scalar(0, 0, 255));
		}
		line(drawImage, Point(0, drawImage.rows - 1), Point(0, drawImage.rows - 1 - 0), Scalar(0, 0, 255));
		imshow("histROI", drawImage);
	}


	return dstHist;
}

//���������㷨
//������	src �Ҷ�ͼ
//������	dst ������ͼ
//������	pt	���ӵ�
//������	th	������ֵ
//������	area_max �������ֵ
//������	area_min ������Сֵ
//������	re	��Χ����
//������	distance_min ��С��������
//������	distanceMinPt ��С��λ��
//������	distance_max ����������
//������	distanceMaxPt ����λ��
//���أ�	�ɹ�0 ʧ��-1
int Capture::RegionGrow(Mat src, Mat* dst, Point2i pt, int th, int area_max, int area_min, Rect* re, int* distance_min, Point* distanceMinPt,int* distance_max,Point* distanceMaxPt)
{
	Point2i ptGrowing;								//��������λ��
	int nGrowLable = 0;								//����Ƿ�������
	int nSrcValue = 0;								//�������Ҷ�ֵ
	int nCurValue = 0;								//��ǰ������Ҷ�ֵ
	Mat matDst = Mat::zeros(src.size(), CV_8UC1);	//����һ���հ��������Ϊ��ɫ
	//��������˳������
	int DIR[8][2] = { { -1, -1 }, { 0, -1 }, { 1, -1 }, { 1, 0 }, { 1, 1 }, { 0, 1 }, { -1, 1 }, { -1, 0 } };
	int left = pt.x;
	int right = pt.x;
	int top = pt.y;
	int bottom = pt.y;
	int counter = 0;								//��¼�������
	queue<Point2i> vcGrowPt;						//������ջ
	vcGrowPt.push(pt);								//��������ѹ��ջ��
	matDst.at<uchar>(pt.y, pt.x) = 255;				//���������
	nSrcValue = src.at<uchar>(pt.y, pt.x);			//��¼������ĻҶ�ֵ
	*distance_min = 255;								//Ѱ�������
	*distance_max = 0;


	while (!vcGrowPt.empty())						//����ջ��Ϊ��������
	{
		pt = vcGrowPt.front();						//ȡ��һ��������
		vcGrowPt.pop();

		//�ֱ�԰˸������ϵĵ��������
		for (int i = 0; i < 8; ++i)
		{
			ptGrowing.x = pt.x + DIR[i][0];
			ptGrowing.y = pt.y + DIR[i][1];
			//����Ƿ��Ǳ�Ե��
			if (ptGrowing.x < 0 || ptGrowing.y < 0 || ptGrowing.x >(src.cols - 1) || (ptGrowing.y > src.rows - 1))
				continue;

			nGrowLable = matDst.at<uchar>(ptGrowing.y, ptGrowing.x);		//��ǰ��������ĻҶ�ֵ

			if (nGrowLable == 0)					//�����ǵ㻹û�б�����
			{
				nCurValue = src.at<uchar>(ptGrowing.y, ptGrowing.x);
				if (nCurValue != 0)
				{
					if (abs(nSrcValue - nCurValue) < th)					//����ֵ��Χ��������
					{
						matDst.at<uchar>(ptGrowing.y, ptGrowing.x) = 255;		//���Ϊ��ɫ
						vcGrowPt.push(ptGrowing);					//����һ��������ѹ��ջ��
						if (left > ptGrowing.x)
							left = ptGrowing.x;
						if (right < ptGrowing.x)
							right = ptGrowing.x;
						if (top > ptGrowing.y)
							top = ptGrowing.y;
						if (bottom < ptGrowing.y)
							bottom = ptGrowing.y;

						if (*distance_min > nCurValue)
						{
							*distance_min = nCurValue;
							distanceMinPt->x = ptGrowing.x;
							distanceMinPt->y = ptGrowing.y;
						}
						else if (*distance_max < nCurValue)
						{
							*distance_max = nCurValue;
							distanceMaxPt->x = ptGrowing.x;
							distanceMaxPt->y = ptGrowing.y;
						}

						

						//�޶������������
						if (++counter > area_max)
						{
							//cout << "X-Axis: " << xAxisCounter << "\tY-Axis: " << yAsisCounter << endl;
							//cout << "x:" << (double)xAxisCounter / counter << "y: " << (double)yAsisCounter / counter << endl;
							printf("Big area: %d\n", counter);
							*re = Rect(Point(left, top), Point(right, bottom));
							*dst = matDst.clone();


							return 0;
						}
					}
				}

			}
		}
	}
	//cout << "X-Axis: " << xAxisCounter << "\tY-Axis: " << yAsisCounter << endl;
	//cout << "x:" << (double)xAxisCounter / counter << "y: " << (double)yAsisCounter / counter << endl;
	printf("area: %d\n", counter);
	*re = Rect(Point(left, top), Point(right, bottom));

	*dst = matDst.clone();


	if (counter < area_min)
	{
		printf("Small!\n");
		return -1;
	}
	return 0;
}

//���׹���+��Ч���ر�������
//���룺	src ԭʼ���ͼ��ushort
//���룺	row �����������
//���룺	re  ����������
//���룺	orderTh �����ݶ���ֵ
//���룺	th	������ֵ
//���룺	areaTh ��Ч����ٷֱ���ֵ
//���أ�	0 �ɹ� -1����
int Capture::secondOrderCheck(Mat src, Mat row,Rect re, int orderTh,float th,float areaTh)
{
	long yAsisCounter = 0;
	long xAxisCounter = 0;

	int reLeft = re.y + 1;
	int reRight = re.y + re.height - 1;
	int reTop = re.x + 1;
	int reBottom = re.x + re.width - 1;

	int area = 0;
	

	for (int i = reTop; i < reBottom; i++)
	{
		for (int j = reLeft; j < reRight; j++)
		{
			if (row.at<uchar>(i, j) != 0)
			{
				ushort nCurValue = src.at<ushort>(i, j);
				//ͳ�ƶ���
				if (abs(src.at<ushort>(i - 1, j) + src.at<ushort>(i + 1, j) - 2 * nCurValue) <orderTh)
				{
					yAsisCounter++;
				}
				if (abs(src.at<ushort>(i, j + 1) + src.at<ushort>(i, j - 1) - 2 * nCurValue) < orderTh)
				{
					xAxisCounter++;
				}
				area++;
			}
			
		}
	}

	cout << "x:" << (double)xAxisCounter / area << " y:" << (double)yAsisCounter / area << endl;

	//���ƽ��
	if (std::abs((double)xAxisCounter / area) > th)
		return -1;
	if (std::abs((double)yAsisCounter / area) > th)
		return -1;

	//�����Ч����
	cout <<"area%:" << (double)area / re.width / re.height <<" "<< areaTh << endl;
	//if ((double)area / re.width / re.height < areaTh)
		//return -1;

	int fiftyArea = 0;			//50%����Ч����ͳ��
	reLeft = re.x + re.width / 4+5;
	reRight = re.x + re.width / 4 * 3 -5;
	reTop = re.y + re.height / 4 + 5;
	reBottom = re.y + re.height / 4 * 3 -5;
	for (int i = reTop; i < reBottom; i++)
	{
		for (int j = reLeft; j < reRight; j++)
		{
			if (row.at<uchar>(i, j) != 0)
			{
				fiftyArea++;
			}
		}
	}

	cout << (double)fiftyArea / (re.height-20) / (re.width-20) * 4 << endl;
	if ((double)fiftyArea / (re.height-20) / (re.width-20) * 4 < 0.93)
		return -1;

	return 0;
}

//���׹���+��Ч���ر�������
//���룺	src ԭʼ���ͼ��ushort
//���룺	row �����������
//���룺	re  ����������
//���룺	orderTh �����ݶ���ֵ
//���룺	th	������ֵ
//���룺	areaTh ��Ч����ٷֱ���ֵ
//���أ�	0 �ɹ� -1����
int Capture::secondOrderCheckFace(Mat src, Mat row, Rect re, int orderTh, float th, float areaTh)
{
	long yAsisCounter = 0;
	long xAxisCounter = 0;

	int reLeft = re.y + 1;
	int reRight = re.y + re.height - 1;
	int reTop = re.x + 1;
	int reBottom = re.x + re.width - 1;

	int area = 0;

	reLeft = reLeft < 1 ? 1 : reLeft;
	reRight = reRight >= row.size().width-1 ? row.size().width - 2 : reRight;
	reTop = reTop < 1 ? 1 : reTop;
	reBottom = reBottom >= row.size().height - 1 ? row.size().height - 2 : reBottom;

	for (int i = reTop; i < reBottom; i++)
	{
		for (int j = reLeft; j < reRight; j++)
		{
			if (row.at<uchar>(i, j) != 0)
			{
				ushort nCurValue = src.at<ushort>(i, j);
				//ͳ�ƶ���
				if (abs(src.at<ushort>(i - 1, j) + src.at<ushort>(i + 1, j) - 2 * nCurValue) <orderTh)
				{
					yAsisCounter++;
				}
				if (abs(src.at<ushort>(i, j + 1) + src.at<ushort>(i, j - 1) - 2 * nCurValue) < orderTh)
				{
					xAxisCounter++;
				}
				area++;
			}

		}
	}

	cout << "x:" << (double)xAxisCounter / area << " y:" << (double)yAsisCounter / area << endl;

	//���ƽ��
	if (std::abs((double)xAxisCounter / area) > th)
		return -1;
	if (std::abs((double)yAsisCounter / area) > th)
		return -1;

	//�����Ч����
	cout << "area%:" << (double)area / re.width / re.height << " " << areaTh << endl;
	//if ((double)area / re.width / re.height < areaTh)
	//return -1;
#if 0
	int fiftyArea = 0;			//50%����Ч����ͳ��
	reLeft = re.x + re.width / 4 + 5;
	reRight = re.x + re.width / 4 * 3 - 5;
	reTop = re.y + re.height / 4 + 5;
	reBottom = re.y + re.height / 4 * 3 - 5;
	for (int i = reTop; i < reBottom; i++)
	{
		for (int j = reLeft; j < reRight; j++)
		{
			if (row.at<uchar>(i, j) != 0)
			{
				fiftyArea++;
			}
		}
	}

	cout << (double)fiftyArea / (re.height - 20) / (re.width - 20) * 4 << endl;
	if ((double)fiftyArea / (re.height - 20) / (re.width - 20) * 4 < 0.93)
		return -1;
#endif
	return 0;
}


//����255��0
//����	src �Ҷ�ͼ��
//����  dst ������
void Capture::cutImage(Mat src, Mat* dst)
{
	*dst = src.clone();
	for (int i = 0; i < dst->rows; i++)
	{
		for (int j = 0; j < dst->cols; j++)
		{
			if (j < dst->cols / 4 || j >(dst->cols / 4 * 3))
				dst->at<uchar>(i, j) = 0;
			else if (dst->at<uchar>(i, j) == 255)
				dst->at<uchar>(i, j) = 0;
		}
	}
}

//����һ���������
//������	src  �Ҷ�ͼ��
//������	line ����
//������	minLoc ������Сλ��
//������	min	 ������Сֵ
//������	dra	 0����ʾͼ�� 1��ʾͼ��
//����ֵ��	NULL
void Capture::drawDepthLineCol(Mat src, int lineNum, int *minLoc, int *min, int dra)
{

	//�����ȴ���һ���ڵ׵�ͼ��Ϊ�˿�����ʾ��ɫ�����Ըû���ͼ����һ��8λ��3ͨ��ͼ��    
	Mat drawImage = Mat::zeros(Size(256, src.size().height), CV_8UC3);


	*min = 255;
	*minLoc = -1;

	for (int i = 1; i < src.rows; i++)
	{
		int value = src.at<uchar>(i, lineNum);
		if (dra)
			line(drawImage, Point(drawImage.cols - 1, i), Point(drawImage.cols - 1 - value, i), Scalar(0, 0, 255));
		if (value != 0)
		{
			if (*min > value)
			{
				*min = value;
				*minLoc = i;
			}
		}
	}

	if (dra)
	{
		line(drawImage, Point(drawImage.cols - 1, 0), Point(drawImage.cols - 1 - 0, 0), Scalar(0, 0, 255));
		imshow("row", drawImage);
	}
}


//��һ������ߵ�
//���룺	src �Ҷ�ͼ��
//���룺	col ������
//���룺	scaler ����������Χ
//���룺	th ��ֵ
//���أ�	�к�
int Capture::findTop(Mat src, int col, int scaler, int th)
{
	for (int i = 0; i < src.rows; i++)
	{
		int avaliCounter = 0;
		for (int j = 0; j <= scaler; j++)
		{
			//���ҷ�Χ
			int rightdepth = src.at<uchar>(i, col + j);
			if (rightdepth != 0)
			{
				avaliCounter++;

				/*if (rightdepth>topdepth)
				    topdepth=rightdepth;*/
			}
			//if (src.at<uchar>(i, col + j) != 0)

			//int leftdepth = src.at<uchar>(i, col + j);
			int temp = col - j;
			if (temp < 0)
				temp = 0;
			int leftdepth = src.at<uchar>(i, temp);
			if (leftdepth != 0)
			{
				avaliCounter++;

				/*if (leftdepth>topdepth)
				  topdepth = leftdepth;*/
			}
		}
		
	/*	if (topdepth < facedepthmin)
			cout << "����̫��������Զ��" << endl;
		if ((topdepth + 150)>facedepthmax)
			cout << "����̫Զ���������" << endl;*/
		if (avaliCounter > th)
		{
			cout << "��ߵ�" << topdepth << endl;
			/*if (topdepth < facedepthmin)
				cout << "����̫��������Զ��" << endl;
			if ((topdepth + 80)>facedepthmax)
				cout << "����̫Զ���������" << endl; */

			return i;
		}
	
	}
	cout << "��ߵ�" << topdepth << endl;
	return -1;
}
int Capture::findtestTop(Mat src, int col, int scaler, int lowth,int highth)
{
	for (int i = 0; i < src.rows; i++)
	{
		int avaliCounter = 0;
		for (int j = 0; j <= scaler; j++)
		{
			//���ҷ�Χ
			int rightdepth = src.at<uchar>(i, col + j);
			if (rightdepth != 0)
			{
				avaliCounter++;

				/*if (rightdepth>topdepth)
				topdepth=rightdepth;*/
			}
			//if (src.at<uchar>(i, col + j) != 0)

			//int leftdepth = src.at<uchar>(i, col + j);
			int temp = col - j;
			if (temp < 0)
				temp = 0;
			int leftdepth = src.at<uchar>(i, temp);
			if (leftdepth != 0)
			{
				avaliCounter++;

				/*if (leftdepth>topdepth)
				topdepth = leftdepth;*/
			}
		}

		/*	if (topdepth < facedepthmin)
		cout << "����̫��������Զ��" << endl;
		if ((topdepth + 150)>facedepthmax)
		cout << "����̫Զ���������" << endl;*/
		if (avaliCounter > lowth && avaliCounter < highth)
		{
			//cout << "��ߵ�" << topdepth << endl;
			/*if (topdepth < facedepthmin)
			cout << "����̫��������Զ��" << endl;
			if ((topdepth + 80)>facedepthmax)
			cout << "����̫Զ���������" << endl; */

			return i;
		}

	}
	cout << "��ߵ�" << topdepth << endl;
	return -1;
}


//��ȡͷ����������
//���룺	src �Ҷ�ͼ��
//���룺	dst ���ͼ����
//���룺	top ͷ��y����
//���룺	th  ͷ��������ֵ
//���룺	middle ���ĵ�x
//���룺	width Ѱ����������ҿ��
//���룺	pt	���������
void Capture::cutBody(Mat src, Mat *dst, int top, int th, int middle, int width, Point* pt)
{
	*dst = Mat::zeros(Size(src.size().width, src.size().height), CV_8U);
	uchar min = 255;
	int topRange;

	if (top + th > src.rows)
		topRange = src.rows;
	else
		topRange = top + th;

	for (int i = 0; i < topRange; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			dst->at<uchar>(i, j) = src.at<uchar>(i, j);

			if (j > middle - width && j < middle + width && src.at<uchar>(i, j) != 0)
			{
				if (min > src.at<uchar>(i, j))
				{
					min = src.at<uchar>(i, j);
					pt->x = j;
					pt->y = i;
				}
			}
		}
	}
}
void Capture::avgimage(Mat avgdepth)
{
	int k = 0;
	for (int i = 0; i < avgdepth.rows; i++)
	{
		for (int j = 0; j < avgdepth.cols; j++)
		{
			k += avgdepth.at<uchar>(i, j);
		}
	}
	k = k / (avgdepth.rows*avgdepth.cols);
	cout << "k=" << k << endl;
	myCalcHist(avgdepth, 1);
}

Mat Capture::get3dface(Mat imageroi)
{
	//double minval = 0.0, maxval = 0.0;
	Mat mask;
	//minMaxIdx(imageroi, &minval, &maxval);  // get the min and max depth in face img

	mask = getGrownImage();

	for (int i = 0; i < mask.rows; i++)
	{
		for (int j = 0; j < mask.cols; j++)
		{
			if (0 == mask.at<uchar>(i, j) )
			{
				imageroi.at<ushort>(i, j) = 0;
			}
		}
	}
	//imagedepthROI =capture.enhancedepth(imagedepthROI);
	return imageroi.clone();
}


Mat Capture::enhancedepth(Mat imagecolor)
{
	double minv = 0.0, maxv = 0.0;
	double* minp = &minv;
	double* maxp = &maxv;
	minMaxIdx(imagecolor, minp, maxp);
	double para = 255 / (maxv - minv);
	cout << "para" << para << " " << maxv << " " << minv << endl;
	for (int i = 0; i < imagecolor.rows; i++)
	{
		for (int j = 0; j < imagecolor.cols; j++)
		{
			imagecolor.at<ushort>(i, j) = (ushort)((imagecolor.at<ushort>(i, j)-(ushort)minv) *para);
		}

	}
	return imagecolor.clone();
}

Mat Capture::getMask()
{
	return objface.mask.clone();
}

void Capture::resetFaceSignal(void)
{

	Capture::isface = false;
}
void Capture::setFaceSignal(void)
{

	Capture::isface = true;
}
void Capture::getdepthHist(Mat srcimg, int mindepth, int maxdepth, int *phhist, int *pvhist)
{
	unsigned short grayValue;
	float tempvalue;
    int hist[256] = { 0 };
   int tempvaluemax=0;
   int shiftvalue;
   int allvalue[26] = {0};
   int avgvalue[26] = {0};
	float par = (float)255 / (maxdepth - mindepth);
	//float par = (maxdepth - mindepth) / (float)255;
	for (int i = 0; i < 256; i++)
	{
		phhist[i] = 0;
		pvhist[i] = 0;
	}


	for (int y = 0; y < srcimg.rows; y++)
	{

		for (int x = 0; x < srcimg.cols; x++)
		{
			grayValue = srcimg.at<ushort>(y, x);
			if ((grayValue > 0) && (grayValue < mindepth))
			{
				tempvalue = (float)mindepth;
			}
			else if (grayValue > maxdepth||grayValue==0)
			{
				tempvalue = 0;
			}
			else
			{
				tempvalue = (float)(grayValue - mindepth);
			}
			tempvalue = tempvalue*par;
			//cout << "tempvalue" << tempvalue << endl;
			shiftvalue = (int)tempvalue;

			if (tempvalue >0)
			{
				phhist[y]++;
				pvhist[x]++;
				if (shiftvalue < 255)
				{
					hist[shiftvalue]++;
				}
				
			}

		}	
	}



	//hist��һ������;
	for (int i = 0; i < 256; i++)
	{
		if (tempvaluemax < hist[i])
			tempvaluemax = hist[i];
	//	cout << "hist" << hist[i] << endl;
	}
	float histpara = 255.0f / tempvaluemax;

	//�����ȴ���һ���ڵ׵�ͼ��Ϊ�˿�����ʾ��ɫ�����Ըû���ͼ����һ��8λ��3ͨ��ͼ��    
	Mat drawImage = Mat::zeros(Size(257,257), CV_8UC3);

	//for (int i = 1; i < 256; i++)
	//{
	//	//int value = shadow_down.at<ushort>(0, i - 1);
	//	int value = hist[i];
	//	value = (int)(value * histpara);
	//	//cout << "value" << value << endl;
	//	line(drawImage, Point(i, drawImage.rows - 1), Point(i, drawImage.rows - 1 - value), Scalar(0, 0, 255));
	//}
	//line(drawImage, Point(0, drawImage.rows - 1), Point(0, drawImage.rows - 1 - 0), Scalar(0, 0, 255));
	//imshow("hist_zip", drawImage);

	//ofstream file("depthface_Hist_average.txt", ios::out);

	cout << "average:";
	for (int i = 0; i < 25; i++)
	{
		for (int j = 1; j < 11; j++)
		{
			allvalue[i] += hist[i * 10 + j];
		}
		avgvalue[i] = allvalue[i] / 10;

		file << int(avgvalue[i]) << ",";
		cout << allvalue[i] << ",";

	}

	// face or fake check: test1
	for (int i = 20; i < 25; i++)
	{
		if (avgvalue[i] < 20)
		{
			resetFaceSignal();
			break;

		}
	}
	file << "\n";
	cout << "\n" ;
	for (int i = 0; i < 25; i++)
	{
		for (int j = 1; j < 11; j++)
		{
			hist[i*10+j]=avgvalue[i];
		}
	}
	for (int i = 1; i < 256; i++)
	{
		//int value = shadow_down.at<ushort>(0, i - 1);
		int value = hist[i];
		value = (int)(value * histpara);
		//cout << "value" << value << endl;
		line(drawImage, Point(i, drawImage.rows - 1), Point(i, drawImage.rows - 1 - value), Scalar(0, 0, 255));
	}

	line(drawImage, Point(0, drawImage.rows - 1), Point(0, drawImage.rows - 1 - 0), Scalar(0, 0, 255));
	imshow("hist_zip", drawImage);

	return;
}
