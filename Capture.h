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

#ifndef _CAPTURE_H_
#define _CAPTURE_H_




#include <OpenNI.h>
#include <queue>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//#define MAX_DEPTH 10000

using namespace cv;
using namespace std;

class ObjectFace
{
public:
	int dist;
	int depthmin;
	int depthmax;
	Mat mask;
	Rect rect;
	Point minPt;
	Point maxPt;
	int HHist_depth[256];
	int VHist_depth[256];
};



class Capture
{
public:
	Capture();
	~Capture();

	int cropX,cropY,cropwidth,crophigh;

	cv::Mat  depthbuf1;
	cv::Mat  depthbuf2;
	cv::Mat  depthbuf3;
	int		 depthcount;  // use 3 frame for  face check one time;

	bool findobj(cv::Mat imgdepth);  // use orignal depth data
	Rect checkface(void);
	Mat getGrownImage(void);
	cv::Mat cropir(cv::Mat imgir);
	void avgimage(Mat avgdepth); 
	Mat enhancedepth(Mat imagecolor);
	Mat get3dface(Mat imageroi);
	int getGrownImage_minval(void);
	int getGrownImage_maxval(void);
	Point getGrownImage_minPoint(void);
	Point getGrownImage_maxPoint(void);
	MatND myCalcHist(Mat imageGray, int isShow);
	void getdepthHist(Mat srcimg, int mindepth, int maxdepth, int *phhist, int *pvhist);
	Mat getMask();
	int secondOrderCheck(Mat src, Mat row, Rect re, int orderTh, float th,float areaTh);
	int findtestTop(Mat src, int col, int scaler, int lowth, int highth);
	bool findcolorobj(Mat imgcolor);
	ObjectFace objface;
	bool isface;
	int DetectLivingBody(Mat imgdepth);
	int secondOrderCheckFace(Mat src, Mat row, Rect re, int orderTh, float th, float areaTh);

protected:


private:
	void cutImage(Mat src, Mat* dst);
	void drawDepthLineCol(Mat src, int lineNum, int *minLoc, int *min, int dra);
	void flatDetect(Mat src, int th);
	//MatND myCalcHist(Mat imageGray, int isShow);
	void on_mouse(int EVENT, int x, int y, int flags, void* userdata);
	void outstandingDetect(Mat src, int th);
	int RegionGrow(Mat src, Mat* dst, Point2i pt, int th, int area_max, int area_min, Rect* re, int* distance_min, Point* distanceMinPt, int* distance_max, Point* distanceMaxPt);
	Mat shadowDown(Mat src, Mat* dst, int image);
	Mat shadowDownPar(Mat src, Mat* dst, int image);
	Mat shadowLeft(Mat src, Mat* dst, int image);
	Mat shadowLeftPar(Mat src, Mat* dst, int image);
	int findTop(Mat src, int col, int scaler,int th);
	void cutBody(Mat src, Mat *dst, int top, int th);
	void cutBody(Mat src, Mat *dst, int top, int th, int middle, int width, Point* pt);
	
	void Capture::resetFaceSignal(void);
	void Capture::setFaceSignal(void);

};


#endif  //_CAPTURE_H_
