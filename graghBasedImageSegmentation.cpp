// graghBasedImageSegmentation.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <vector>  
#include <ctime>  
#include "opencv2/opencv.hpp"  
#include "segmentimage.h"  

using namespace std;
using namespace cv;




//膨胀
void Pengzhang1(Mat src, Mat& dst, int elementsize1 = 3)
{
	Mat element1 = getStructuringElement(MORPH_RECT, Size(elementsize1 * 2 + 1, elementsize1 * 2 + 1));//保证是奇数
	dilate(src, dst, element1);//膨胀
}

void Fushi1(Mat src, Mat& dst, int elementsize2 = 3)
{
	Mat element2 = getStructuringElement(MORPH_RECT, Size(elementsize2 * 2 + 1, elementsize2 * 2 + 1));//保证是奇数
	erode(src, dst, element2);//腐蚀
}

//*im: image to segment.
//* sigma : to smooth the image.
//* c : constant for treshold function.
//* min_size : minimum component size(enforced by post - processing stage).
//* num_ccs : number of connected components in the segmentation.
//void SegImage(cv::Mat& image, cv::Mat& result, float sigma = 0.5, float k = 500, int min_size = 100)
//{
//	Mat grayImg;
//	result.create(image.size(), image.type());
//	int num_ccs;
//	cvtColor(image, grayImg, COLOR_BGR2GRAY);
//	image.convertTo(image, CV_32FC3);
//	result = segment_image(image, sigma, k, min_size, &num_ccs);
//
//}
void drawAllContours1(vector<vector<Point>> contours, vector<Vec4i> hierarchy, Mat& dstImage)
{
	int index = 0;
	for (; index >= 0; index = hierarchy[index][0])
	{
		Scalar color(rand() & 255, rand() & 255, rand() & 255);
		drawContours(dstImage, contours, index, color, FILLED, 8, hierarchy);
	}
}

#include <iostream>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main4()
{
	const int MAX_CLUSTERS = 5;
	Vec3b colorTab[] =
	{
		Vec3b(0, 0, 255),
		Vec3b(0, 255, 0),
		Vec3b(255, 100, 100),
		Vec3b(255, 0, 255),
		Vec3b(0, 255, 255)
	};
	Mat data, labels;
	Mat pic = imread("dustbin1.jpg");//0表示灰度读入，1表示rgb读入 scene1 rabit1 dustbin1 photo1  threeObject1 door1  door3 flower1 eagle1 am1.bmp

	for (int i = 0; i < pic.rows; i++)     //像素点线性排列
		for (int j = 0; j < pic.cols; j++)
		{
			Vec3b point = pic.at<Vec3b>(i, j);
			Mat tmp = (Mat_<float>(1, 3) << point[0], point[1], point[2]);
			data.push_back(tmp);
		}

	//根据浏览图片，确定k=3
	kmeans(data, 30, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 6, KMEANS_RANDOM_CENTERS);

	int n = 0;
	//显示聚类结果，不同的类别用不同的颜色显示
	for (int i = 0; i < pic.rows; i++)
		for (int j = 0; j < pic.cols; j++)
		{
			int clusterIdx = labels.at<int>(n);
			pic.at<Vec3b>(i, j) = colorTab[clusterIdx];
			n++;
		}
	imshow("K-means聚类算法", pic);
	waitKey(0);

	return 0;
}
//int main5(int argc, _TCHAR* argv[])
//{
//	Mat Image1, Image2;
//	Mat LoadImg1, LoadImg2;
//	Mat grayImg1, grayImg2;
//	Mat SegImg1, SegImg2;
//	Mat PengzhangImg1, PengzhangImg2;
//	Mat FushiImg1, FushiImg2;
//	Mat CannyImg1, CannyImg2;
//	Mat ContoursImg1, ContoursImg2;
//	Mat BlurImg1, BlurImg2;
//
//	LoadImg1 = imread("eagle1.jpg", 1);//0表示灰度读入，1表示rgb读入 scene1 rabit1 dustbin1 photo1  threeObject1 door1  door3 flower1 eagle1 am1.bmp
//	LoadImg2 = imread("eagle2.jpg", 1);
//
//	Image1 = LoadImg1.clone();
//	Image2 = LoadImg2.clone();
//
//	//显示图像  
//	imshow("image1", Image1);
//	imshow("image2", Image2);
//
//
//	//*im: image to segment.
//	//	* sigma : to smooth the image.
//	//	* c : constant for treshold function.
//	//	* min_size : minimum component size(enforced by post - processing stage).
//	//	* num_ccs : number of connected components in the segmentation.
//	SegImage(Image1, SegImg1, 9,  450, 120);
//	SegImage(Image2, SegImg2, 9, 450, 120);
//
//	//imwrite("Seg1.jpg", SegImg1);
//	//imwrite("Seg2.jpg", SegImg2);
//
//	imshow("SegImg1", SegImg1);
//	imshow("SegImg2", SegImg2);
//
//	//Fushi1(SegImg1, FushiImg1, 10);
//	//Pengzhang1(FushiImg1, PengzhangImg1, 10);
//
//	//Fushi1(SegImg2, FushiImg2, 10);
//	//Pengzhang1(FushiImg2, PengzhangImg2, 10);
//
//	//imshow("PengzhangFhushi1", PengzhangImg1); 
//	//imshow("PengzhangFhushi2", PengzhangImg2);
//
//
//
//	//均值滤波	//blur(PengzhangImg1, BlurImg1, Size(1, 1));
//	//imshow("blurImg1", BlurImg1);
//
//	//blur(PengzhangImg2, BlurImg2, Size(1, 1));
//	//imshow("blurImg2", BlurImg2);
//
//
//
//	//canny边缘提取
//	Canny(SegImg1, CannyImg1, 90, 200, 3);
//	Canny(SegImg2, CannyImg2, 90, 200, 3);
//	imshow("CannyImg1", CannyImg1);
//	imshow("CannyImg2", CannyImg2);
//
//
//	Mat shapeMatchImg1 = LoadImg1.clone();
//	Mat shapeMatchImg2 = LoadImg2.clone();
//	vector<vector<Point>> contours1, contours2;
//	vector<Vec4i> hierarchy1, hierarchy2;
//	ContoursImg1.create(CannyImg1.size(), CannyImg1.type());
//	ContoursImg2.create(CannyImg2.size(), CannyImg2.type());
//
//	findContours(CannyImg1, contours1, hierarchy1, RETR_CCOMP, CHAIN_APPROX_TC89_KCOS);
//	findContours(CannyImg2, contours2, hierarchy2, RETR_CCOMP, CHAIN_APPROX_TC89_KCOS);
//	drawAllContours1(contours1, hierarchy1, ContoursImg1);
//	drawAllContours1(contours2, hierarchy2, ContoursImg2);
//	imshow("ContoursImg1", ContoursImg1);
//	imshow("ContoursImg2", ContoursImg2);
//
//
//	srand((int)time(0));
//	vector<int> matchContours1;
//	vector<int> matchContours2;
//	for (int i = 0; i < contours1.size(); i++)//遍历待测试图像的轮廓
//	{
//		int Most_Similar_No = -1;
//		float Min_Score = 9999;
//		for (int j = 0; j < contours2.size(); j++)//遍历待测试图像的轮廓
//		{
//			//返回此轮廓与模版轮廓之间的相似度,a0越小越相似
//			float score = matchShapes(contours1[i], contours2[j], CONTOURS_MATCH_I3, 0);
//			//cout << "模版轮廓"<<i<<"与待测试图像轮廓" << j << "的相似度:" << score << endl;//输出两个轮廓间的相似度
//			if (score < Min_Score)//如果此轮廓与模版轮廓的相似度小于0.1
//			{
//				Min_Score = score;
//				Most_Similar_No = j;
//			}
//		}
//		if (Most_Similar_No != -1 && Min_Score <= 4)//如果此轮廓与模版轮廓的相似度小于0.1
//		{
//			int r = rand() % 255;
//			int g = rand() % 255;
//			int b = rand() % 255;
//
//			matchContours1.push_back(i);
//			matchContours2.push_back(Most_Similar_No);
//
//
//			drawContours(shapeMatchImg1, contours1, i, Scalar(r, g, b), 2, 8);//则在待测试图像上画出此轮廓
//			drawContours(shapeMatchImg2, contours2, Most_Similar_No, Scalar(r, g, b), 2, 8);//则在待测试图像上画出此轮廓
//
//		}
//
//	}
//
//	imshow("shapeMatchImg1", shapeMatchImg1);
//	imshow("shapeMatchImg2", shapeMatchImg2);
//
//
//	////此函数等待按键，按键盘任意键就返回  
//	waitKey(0);
//	return 0;
//}
//
