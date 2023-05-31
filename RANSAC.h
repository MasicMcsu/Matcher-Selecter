#pragma once
#ifndef RANSAC_H  
#define RANSAC_H  

#include <opencv2/opencv.hpp>  //头文件
#include <opencv2/features2d.hpp>
#include <vector>
using namespace cv;  //包含cv命名空间
using namespace std;


//void ransac(Mat Image1, Mat Image2, const vector<DMatch> sift_matches_origional, vector<DMatch>& sift_matches_with_ShapeConst, vector<KeyPoint> sift_keypoints_1, vector<KeyPoint>  sift_keypoints_2);
void ransac(Mat Image1, Mat Image2, const vector<DMatch> sift_matches_origional, vector<DMatch>& sift_matches_with_ShapeConst, vector<KeyPoint> sift_keypoints_1, vector<KeyPoint>  sift_keypoints_2);

void MyRANSAC(const vector<DMatch> sift_matches_origional, vector<KeyPoint> sift_keypoints_1, vector<KeyPoint>  sift_keypoints_2,
	vector<DMatch>& InliersT1, vector<DMatch>& InliersT2, vector<DMatch>& InliersT3, vector<DMatch>& InliersT4, vector<DMatch>& InliersT5,
	vector<DMatch>& InliersT6, vector<DMatch>& InliersT7, vector<DMatch>& InliersT8, vector<DMatch>& InliersT9, vector<DMatch>& InliersT10);
#endif  