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

	//过滤超出点
	Mat depth_cut;
	cutImage(depth_zip, &depth_cut);
	//鼠标点击查看像素信息
	//namedWindow("Display");
	//setMouseCallback("Display", on_mouse, &depth_cut);
	//imshow("Display", depth_cut);


	//伪彩色图像
	Mat depth_rainbow;
	applyColorMap(depth_zip, depth_rainbow, COLORMAP_HSV);
	//imshow("depth_rainbow", depth_rainbow);

	//直方图统计
	//myCalcHist(depth_cut, 0);

	//找鼻子
	//outstandingDetect(depth_zip, 5);

	//向下投影  vertical hisgram
	Mat depth_down;
	shadowDown(depth_cut, &depth_down, 0);
	Point maxP;
	minMaxLoc(depth_down, 0, 0, 0, &maxP);
	

	//找最近点
	//int minLoc;
	//int min;
	//绘制一列
	//drawDepthLineCol(depth_cut, maxP.x, &minLoc, &min, 0);

	//找最高点
	 // int topY = findTop(depth_cut, maxP.x, 10, 10);
	int topY = findtestTop(depth_cut, maxP.x, 10, 10,100);
	if (topY != -1)
	{
		//截取身体
		Mat depth_head;
		Point minHPt = Point();
		cutBody(depth_cut, &depth_head, topY, 100, maxP.x, 60, &minHPt);
		//circle(depth_cut, minHPt, 3, Scalar(255));

		//imshow("depth_cut", depth_cut);
		//imshow("head", depth_headpoint);


		//区域生长
		Rect re;
		Mat depth_grow;
		Point2i pt = Point2i(minHPt);
		Point minPt;
		Point maxPt;
		int dis_min;
		int dis_max;
		int area_th = depth_cut.at<uchar>(pt.y, pt.x)*(-103) + 22000;		//生长区域上限由距离确定

		int result = RegionGrow(depth_cut, &depth_grow, pt, 30, area_th, 5000, &re, &dis_min, &minPt, &dis_max, &maxPt);
		result = secondOrderCheck(depth, depth_grow, re, 5, (float)0.915,0.3f);
		if (!result)
		{
			//调整框大小
			
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


			//寻找结果保存
			objface.rect = re;
			objface.mask = depth_grow.clone();
			objface.depthmax = dis_max;
			objface.depthmin = dis_min;
			
			/*avgfacedepth = (dis_max + dis_min) / 2;

			if (avgfacedepth < facedepthmin)
				cout << "距离太近，请离远一点" << endl;
			if (avgfacedepth  > facedepthmax)
				cout << "请离近一点" << endl;*/
			

			objface.minPt = Point(minPt);
			objface.maxPt = Point(maxPt);

			//rectangle(depth_zip, re, Scalar(255));

			return true;
		}

	}
	return false;
}
//检测活体
//参数： imgdepth 原始深度图像
//返回值： 

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

//山峰检测
//参数：	src 灰度图像
void Capture::outstandingDetect(Mat src, int th)
{
	Mat matDst = Mat::zeros(Size(src.cols, src.rows), CV_8U);

	int max_x;
	int max_y;
	int max_val = 255;

	//判断方向
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


//平面检测
//参数：	src 灰度图像
//参数：	th	阈值
void Capture::flatDetect(Mat src, int th)
{
	Mat matDst = Mat::zeros(Size(src.cols, src.rows), CV_8U);
	//Mat matDst = src.clone();

	//判断方向
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


//向下投影
//参数：	src 输入灰度图像
//参数：	dst 输出一维统计数据
//参数：	image 是否显示统计图
//返回：	无
Mat Capture::shadowDown(Mat src, Mat* dst, int image)
{

	Mat shadow_down = Mat::zeros(Size(src.cols, 1), CV_16U);
	//向下投影
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
		//首先先创建一个黑底的图像，为了可以显示彩色，所以该绘制图像是一个8位的3通道图像    
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

//向下投影带权值
//参数：	src 输入灰度图像
//参数：	dst 输出一维统计数据
//参数：	image 是否显示统计图
//返回：	无
Mat Capture::shadowDownPar(Mat src, Mat* dst, int image)
{

	Mat shadow_down = Mat::zeros(Size(src.cols, 1), CV_32S);
	//向下投影
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
		//首先先创建一个黑底的图像，为了可以显示彩色，所以该绘制图像是一个8位的3通道图像    
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

//向左投影
//参数：	src 输入灰度图像
//参数：	dst 输出一维统计数据
//参数：	image 是否显示统计图
//返回：	无
Mat Capture::shadowLeft(Mat src, Mat* dst, int image)
{

	Mat shadow_left = Mat::zeros(Size(src.rows, 1), CV_16U);
	//向左投影
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
		//首先先创建一个黑底的图像，为了可以显示彩色，所以该绘制图像是一个8位的3通道图像    
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

//向左投影带加权
//参数：	src 输入灰度图像
//参数：	dst 输出一维统计数据
//参数：	image 是否显示统计图
//返回：	无
Mat Capture::shadowLeftPar(Mat src, Mat* dst, int image)
{

	Mat shadow_left = Mat::zeros(Size(src.rows, 1), CV_32S);
	//向左投影
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
		//首先先创建一个黑底的图像，为了可以显示彩色，所以该绘制图像是一个8位的3通道图像    
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

//计算直方图
//参数：	imageGray 灰度图像
//参数：	isShow	-0 不绘制
//					-1 绘制
//返回值：	灰度直方图数组
MatND Capture::myCalcHist(Mat imageGray, int isShow)
{
	//计算直方图
	int channels = 0;
	MatND dstHist;
	int histSize[] = { 256 };
	float midRanges[] = { 0, 256 };
	const float *ranges[] = { midRanges };
	calcHist(&imageGray, 1, &channels, Mat(), dstHist, 1, histSize, ranges, true, false);



	if (isShow)
	{
		//绘制直方图,首先先创建一个黑底的图像，为了可以显示彩色，所以该绘制图像是一个8位的3通道图像    
		Mat drawImage = Mat::zeros(Size(257, 257), CV_8UC3);
		//任何一个图像的某个像素的总个数有可能会很多，甚至超出所定义的图像的尺寸，  
		//所以需要先对个数进行范围的限制，用minMaxLoc函数来得到计算直方图后的像素的最大个数    
		double g_dHistMaxValue;
		Point maxLoc;
		dstHist.at<float>(0) = 0;
		minMaxLoc(dstHist, 0, &g_dHistMaxValue, &maxLoc, 0);
		normalize(dstHist, dstHist, 0, 256, NORM_MINMAX, -1, Mat());
		//将像素的个数整合到图像的最大范围内    
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

//区域生长算法
//参数：	src 灰度图
//参数：	dst 输出结果图
//参数：	pt	种子点
//参数：	th	生长阈值
//参数：	area_max 区域最大值
//参数：	area_min 区域最小值
//参数：	re	范围矩形
//参数：	distance_min 最小距离点距离
//参数：	distanceMinPt 最小点位置
//参数：	distance_max 最大距离点距离
//参数：	distanceMaxPt 最大点位置
//返回：	成功0 失败-1
int Capture::RegionGrow(Mat src, Mat* dst, Point2i pt, int th, int area_max, int area_min, Rect* re, int* distance_min, Point* distanceMinPt,int* distance_max,Point* distanceMaxPt)
{
	Point2i ptGrowing;								//待生长点位置
	int nGrowLable = 0;								//标记是否生长过
	int nSrcValue = 0;								//生长起点灰度值
	int nCurValue = 0;								//当前生长点灰度值
	Mat matDst = Mat::zeros(src.size(), CV_8UC1);	//创建一个空白区域，填充为黑色
	//生长方向顺序数据
	int DIR[8][2] = { { -1, -1 }, { 0, -1 }, { 1, -1 }, { 1, 0 }, { 1, 1 }, { 0, 1 }, { -1, 1 }, { -1, 0 } };
	int left = pt.x;
	int right = pt.x;
	int top = pt.y;
	int bottom = pt.y;
	int counter = 0;								//记录生长面积
	queue<Point2i> vcGrowPt;						//生长点栈
	vcGrowPt.push(pt);								//将生长点压入栈中
	matDst.at<uchar>(pt.y, pt.x) = 255;				//标记生长点
	nSrcValue = src.at<uchar>(pt.y, pt.x);			//记录生长点的灰度值
	*distance_min = 255;								//寻找最近点
	*distance_max = 0;


	while (!vcGrowPt.empty())						//生长栈不为空则生长
	{
		pt = vcGrowPt.front();						//取出一个生长点
		vcGrowPt.pop();

		//分别对八个方向上的点进行生长
		for (int i = 0; i < 8; ++i)
		{
			ptGrowing.x = pt.x + DIR[i][0];
			ptGrowing.y = pt.y + DIR[i][1];
			//检查是否是边缘点
			if (ptGrowing.x < 0 || ptGrowing.y < 0 || ptGrowing.x >(src.cols - 1) || (ptGrowing.y > src.rows - 1))
				continue;

			nGrowLable = matDst.at<uchar>(ptGrowing.y, ptGrowing.x);		//当前待生长点的灰度值

			if (nGrowLable == 0)					//如果标记点还没有被生长
			{
				nCurValue = src.at<uchar>(ptGrowing.y, ptGrowing.x);
				if (nCurValue != 0)
				{
					if (abs(nSrcValue - nCurValue) < th)					//在阈值范围内则生长
					{
						matDst.at<uchar>(ptGrowing.y, ptGrowing.x) = 255;		//标记为白色
						vcGrowPt.push(ptGrowing);					//将下一个生长点压入栈中
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

						

						//限定生长区域面积
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

//二阶过滤+有效像素比例过滤
//输入：	src 原始深度图像ushort
//输入：	row 区域生长结果
//输入：	re  区域生长框
//输入：	orderTh 二阶梯度阈值
//输入：	th	过滤阈值
//输入：	areaTh 有效面积百分比阈值
//返回：	0 成功 -1错误
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
				//统计二阶
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

	//检查平面
	if (std::abs((double)xAxisCounter / area) > th)
		return -1;
	if (std::abs((double)yAsisCounter / area) > th)
		return -1;

	//检查有效区域
	cout <<"area%:" << (double)area / re.width / re.height <<" "<< areaTh << endl;
	//if ((double)area / re.width / re.height < areaTh)
		//return -1;

	int fiftyArea = 0;			//50%框有效像素统计
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

//二阶过滤+有效像素比例过滤
//输入：	src 原始深度图像ushort
//输入：	row 区域生长结果
//输入：	re  区域生长框
//输入：	orderTh 二阶梯度阈值
//输入：	th	过滤阈值
//输入：	areaTh 有效面积百分比阈值
//返回：	0 成功 -1错误
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
				//统计二阶
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

	//检查平面
	if (std::abs((double)xAxisCounter / area) > th)
		return -1;
	if (std::abs((double)yAsisCounter / area) > th)
		return -1;

	//检查有效区域
	cout << "area%:" << (double)area / re.width / re.height << " " << areaTh << endl;
	//if ((double)area / re.width / re.height < areaTh)
	//return -1;
#if 0
	int fiftyArea = 0;			//50%框有效像素统计
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


//超出255置0
//输入	src 灰度图像
//输入  dst 输出结果
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

//绘制一列深度趋势
//参数：	src  灰度图像
//参数：	line 列数
//参数：	minLoc 非零最小位置
//参数：	min	 非零最小值
//参数：	dra	 0不显示图像 1显示图像
//返回值：	NULL
void Capture::drawDepthLineCol(Mat src, int lineNum, int *minLoc, int *min, int dra)
{

	//首先先创建一个黑底的图像，为了可以显示彩色，所以该绘制图像是一个8位的3通道图像    
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


//在一列找最高点
//输入：	src 灰度图像
//输入：	col 过滤列
//输入：	scaler 左右搜索范围
//输入：	th 阈值
//返回：	行号
int Capture::findTop(Mat src, int col, int scaler, int th)
{
	for (int i = 0; i < src.rows; i++)
	{
		int avaliCounter = 0;
		for (int j = 0; j <= scaler; j++)
		{
			//左右范围
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
			cout << "距离太近，请离远点" << endl;
		if ((topdepth + 150)>facedepthmax)
			cout << "距离太远，请离近点" << endl;*/
		if (avaliCounter > th)
		{
			cout << "最高点" << topdepth << endl;
			/*if (topdepth < facedepthmin)
				cout << "距离太近，请离远点" << endl;
			if ((topdepth + 80)>facedepthmax)
				cout << "距离太远，请离近点" << endl; */

			return i;
		}
	
	}
	cout << "最高点" << topdepth << endl;
	return -1;
}
int Capture::findtestTop(Mat src, int col, int scaler, int lowth,int highth)
{
	for (int i = 0; i < src.rows; i++)
	{
		int avaliCounter = 0;
		for (int j = 0; j <= scaler; j++)
		{
			//左右范围
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
		cout << "距离太近，请离远点" << endl;
		if ((topdepth + 150)>facedepthmax)
		cout << "距离太远，请离近点" << endl;*/
		if (avaliCounter > lowth && avaliCounter < highth)
		{
			//cout << "最高点" << topdepth << endl;
			/*if (topdepth < facedepthmin)
			cout << "距离太近，请离远点" << endl;
			if ((topdepth + 80)>facedepthmax)
			cout << "距离太远，请离近点" << endl; */

			return i;
		}

	}
	cout << "最高点" << topdepth << endl;
	return -1;
}


//截取头部以下区域
//输入：	src 灰度图像
//输入：	dst 输出图像结果
//输入：	top 头顶y坐标
//输入：	th  头顶往下阈值
//输入：	middle 中心点x
//输入：	width 寻找最近点左右宽度
//输入：	pt	最近点坐标
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



	//hist归一化处理;
	for (int i = 0; i < 256; i++)
	{
		if (tempvaluemax < hist[i])
			tempvaluemax = hist[i];
	//	cout << "hist" << hist[i] << endl;
	}
	float histpara = 255.0f / tempvaluemax;

	//首先先创建一个黑底的图像，为了可以显示彩色，所以该绘制图像是一个8位的3通道图像    
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
