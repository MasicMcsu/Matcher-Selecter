
#include <iostream>
#include <opencv2/opencv.hpp>  //头文件
#include <opencv2/xfeatures2d.hpp>
#include "GMS.h"
#include "SEGRANSAC.h"
#include "RANSAC.h"
#include "segmentimage.h"  
#include "segmentgraph.h"

#include<stdio.h>
#include<stdlib.h>
#include<time.h>

using namespace cv;  //包含cv命名空间
using namespace std;


int main()
{

		//----------------------------------read Images-------------------------------------------------------------------
		Mat Image1 = imread("F:\\pro\\特征匹配\\上传\\Project方法比较new使用特殊库\\Project方法比较\\1.jpg", 1);//eagle1   door1 House1 flower1 FourObjects1 leaf1 stone1 rabit1 threeObject1 scene1
		Mat Image2 = imread("F:\\pro\\特征匹配\\上传\\Project方法比较new使用特殊库\\Project方法比较\\2.jpg", 1);

		//----------------------------------ORB match-------------------------------------------------------------------
		Ptr<Feature2D> orb = ORB::create(1000);
		orb.dynamicCast<cv::ORB>()->setFastThreshold(20);
		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
		std::vector<KeyPoint> orb_keypoints_1, orb_keypoints_2;
		Mat orb_descriptors_1, orb_descriptors_2;
		orb->detectAndCompute(Image1, noArray(), orb_keypoints_1, orb_descriptors_1);
		orb->detectAndCompute(Image2, noArray(), orb_keypoints_2, orb_descriptors_2);
		std::vector<DMatch> orb_matches_origional;
		matcher->match(orb_descriptors_1, orb_descriptors_2, orb_matches_origional);

		//draw
		Mat orb_img_matches_withoutConst;
		cv::drawMatches(Image1, orb_keypoints_1, Image2, orb_keypoints_2, orb_matches_origional, orb_img_matches_withoutConst);
		cv::imshow("orb-Matches Without Constraint", orb_img_matches_withoutConst);
		std::cout << endl << "ORB" << endl;


		Mat newImgorb_img_matches_withoutConst;
		resize(orb_img_matches_withoutConst, newImgorb_img_matches_withoutConst, Size(orb_img_matches_withoutConst.cols / 4, orb_img_matches_withoutConst.rows / 4), (0, 0), (0, 0), 3); //缩小
		cv::imshow("orb-Matches Without Constraint", newImgorb_img_matches_withoutConst);



		//---------------------------Select---------------------------
		vector<DMatch> GMSSegRANSAC_orb_matches_with_ShapeConstT1, GMSSegRANSAC_orb_matches_with_ShapeConstT2, GMSSegRANSAC_orb_matches_with_ShapeConstT3;

		int GMSSegRANSAC_shaixuan_right_num_T1, GMSSegRANSAC_shaixuan_num_T1, GMSSegRANSAC_origional_right_num_T1;

		MatchesSelecter(0, Image1, Image2, 0.2, orb_matches_origional,
			orb_keypoints_1, orb_keypoints_2,
			GMSSegRANSAC_orb_matches_with_ShapeConstT1);



		Mat frameGMSSegRANSACMatches;
		drawMatches(Image1, orb_keypoints_1, Image2, orb_keypoints_2, GMSSegRANSAC_orb_matches_with_ShapeConstT1, frameGMSSegRANSACMatches, Scalar::all(-1), Scalar::all(-1),
			std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		resize(frameGMSSegRANSACMatches, frameGMSSegRANSACMatches, Size(frameGMSSegRANSACMatches.cols / 4, frameGMSSegRANSACMatches.rows / 4), (0, 0), (0, 0), 3);
		cv::imshow("GMSSegRANSAC", frameGMSSegRANSACMatches);
		waitKey(0);


		return 0;
}
