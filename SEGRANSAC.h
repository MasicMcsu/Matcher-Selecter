#pragma once
#ifndef SEG7_H  
#define SEG7_H  

#include <opencv2/opencv.hpp>  //头文件
#include <opencv2/features2d.hpp>
#include "segmentimage.h"  
#include "segmentgraph.h"
#include <vector>
using namespace cv;  //包含cv命名空间
using namespace std;

//void Seg7GMSSEGRANSAC(int IsFixFund, Mat Image1, Mat Image2, float apha, const vector<DMatch> sift_matches_origional, vector<DMatch>& Inliers, vector<KeyPoint> sift_keypoints_1, vector<KeyPoint>  sift_keypoints_2);

void MatchesSelecter(int IsFixFund, Mat Image1, Mat Image2, float apha,
	const vector<DMatch> sift_matches_origional,
	vector<KeyPoint> sift_keypoints_1, vector<KeyPoint>  sift_keypoints_2,
	vector<DMatch>& InliersT1);
#endif  