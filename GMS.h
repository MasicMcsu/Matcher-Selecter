#pragma once
#ifndef SEG6_H  
#define SEG6_H  

#include <opencv2/opencv.hpp>  //头文件
#include <opencv2/features2d.hpp>
#include <vector>
using namespace cv;  //包含cv命名空间
using namespace std;



void GMS(Mat Image1, Mat Image2, float alpha,int cellnum,
	const vector<DMatch> sift_matches_origional, vector<DMatch>& Inliers,
	vector<KeyPoint> sift_keypoints_1, vector<KeyPoint>  sift_keypoints_2);
void GMSForVoter(Mat Image1, Mat Image2, float alpha, int cellnum,
	const vector<DMatch> sift_matches_origional, vector<bool>& IsRobust,
	vector<KeyPoint> sift_keypoints_1, vector<KeyPoint>  sift_keypoints_2);
void MyGMS(Mat Image1, Mat Image2, vector<KeyPoint> sift_keypoints_1, vector<KeyPoint>  sift_keypoints_2,
	const vector<DMatch> sift_matches_origional,
	vector<DMatch>& InliersT1, vector<DMatch>& InliersT2, vector<DMatch>& InliersT3, vector<DMatch>& InliersT4, vector<DMatch>& InliersT5,
	vector<DMatch>& InliersT6, vector<DMatch>& InliersT7, vector<DMatch>& InliersT8, vector<DMatch>& InliersT9, vector<DMatch>& InliersT10
);
#endif  