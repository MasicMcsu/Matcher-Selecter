

#include"RANSAC.h"

bool SatisfyEpipolarConst1(Mat m_Fundamental, int no, float T, const vector<DMatch> sift_matches_origional, vector<KeyPoint> sift_keypoints_1, vector<KeyPoint>  sift_keypoints_2)
{

	Point2f m1= sift_keypoints_1[sift_matches_origional[no].queryIdx].pt;
	Point2f	m2= sift_keypoints_2[sift_matches_origional[no].trainIdx].pt;
	const double* F = m_Fundamental.ptr<double>();

	double a, b, c, d1, d2, s1, s2, err;

	a = F[0] * m1.x + F[1] * m1.y + F[2];
	b = F[3] * m1.x + F[4] * m1.y + F[5];
	c = F[6] * m1.x + F[7] * m1.y + F[8];

	s2 = 1. / (a * a + b * b);
	d2 = m2.x * a + m2.y * b + c;

	a = F[0] * m2.x + F[3] * m2.y + F[6];
	b = F[1] * m2.x + F[4] * m2.y + F[7];
	c = F[2] * m2.x + F[5] * m2.y + F[8];

	s1 = 1. / (a * a + b * b);
	d1 = m1.x * a + m1.y * b + c;

	err = (float)std::max(d1 * d1 * s1, d2 * d2 * s2);

	if (err <= T*T)
		return true;
	else
		return false;

}


void ransac(float conf, const vector<DMatch> sift_matches_origional, vector<DMatch>& sift_matches_with_ShapeConst, vector<KeyPoint> sift_keypoints_1, vector<KeyPoint>  sift_keypoints_2)
{

	int ptCount = (int)sift_matches_origional.size();
	Mat p1(ptCount, 2, CV_32F);
	Mat p2(ptCount, 2, CV_32F);

	// 把Keypoint转换为Mat
	Point2f pt;
	for (int i = 0; i < ptCount; i++)
	{
		pt = sift_keypoints_1[sift_matches_origional[i].queryIdx].pt;
		p1.at<float>(i, 0) = pt.x;
		p1.at<float>(i, 1) = pt.y;

		pt = sift_keypoints_2[sift_matches_origional[i].trainIdx].pt;
		p2.at<float>(i, 0) = pt.x;
		p2.at<float>(i, 1) = pt.y;
	}

	// 用RANSAC方法计算F
	// Mat m_Fundamental;
	// 上面这个变量是基本矩阵
	vector<uchar> m_RANSACStatus;
	Mat m_Fundamental = findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC, 3.0, conf);
	//std::cout << "RANSAC fund:"<<endl<<m_Fundamental << endl;


	//去掉一部分点
	sift_matches_with_ShapeConst = sift_matches_origional;
	int kkk = 0;
	for (auto it = sift_matches_with_ShapeConst.begin(); it != sift_matches_with_ShapeConst.end();)
	{

		if (m_RANSACStatus[kkk] != 0)
		{
			it++;
		}
		else
		{
			it = sift_matches_with_ShapeConst.erase(it);

		}
		kkk++;
		if (it == sift_matches_with_ShapeConst.end())
			break;
	}
}

void MyRANSAC(const vector<DMatch> sift_matches_origional, vector<KeyPoint> sift_keypoints_1, vector<KeyPoint>  sift_keypoints_2,
	vector<DMatch>& InliersT1, vector<DMatch>& InliersT2, vector<DMatch>& InliersT3, vector<DMatch>& InliersT4, vector<DMatch>& InliersT5,
	vector<DMatch>& InliersT6, vector<DMatch>& InliersT7, vector<DMatch>& InliersT8, vector<DMatch>& InliersT9, vector<DMatch>& InliersT10)
{
	ransac(0.2, sift_matches_origional, InliersT1, sift_keypoints_1, sift_keypoints_2);
	ransac(0.3, sift_matches_origional, InliersT2, sift_keypoints_1, sift_keypoints_2);
	ransac(0.4, sift_matches_origional, InliersT3, sift_keypoints_1, sift_keypoints_2);
	ransac(0.5, sift_matches_origional, InliersT4, sift_keypoints_1, sift_keypoints_2);
	ransac(0.6, sift_matches_origional, InliersT5, sift_keypoints_1, sift_keypoints_2);
	ransac(0.7, sift_matches_origional, InliersT6, sift_keypoints_1, sift_keypoints_2);
	ransac(0.8, sift_matches_origional, InliersT7, sift_keypoints_1, sift_keypoints_2);
	ransac(0.9, sift_matches_origional, InliersT8, sift_keypoints_1, sift_keypoints_2);
	ransac(0.95, sift_matches_origional, InliersT9, sift_keypoints_1, sift_keypoints_2);
	ransac(0.99, sift_matches_origional, InliersT10, sift_keypoints_1, sift_keypoints_2);
}


void ransac(Mat Image1, Mat Image2, const vector<DMatch> sift_matches_origional, vector<DMatch>& sift_matches_with_ShapeConst, vector<KeyPoint> sift_keypoints_1, vector<KeyPoint>  sift_keypoints_2)
{
	
	int ptCount = (int)sift_matches_origional.size();
	Mat p1(ptCount, 2, CV_32F);
	Mat p2(ptCount, 2, CV_32F);

	// 把Keypoint转换为Mat
	Point2f pt;
	for (int i = 0; i < ptCount; i++)
	{
		pt = sift_keypoints_1[sift_matches_origional[i].queryIdx].pt;
		p1.at<float>(i, 0) = pt.x;
		p1.at<float>(i, 1) = pt.y;

		pt = sift_keypoints_2[sift_matches_origional[i].trainIdx].pt;
		p2.at<float>(i, 0) = pt.x;
		p2.at<float>(i, 1) = pt.y;
	}

	// 用RANSAC方法计算F
	// Mat m_Fundamental;
	// 上面这个变量是基本矩阵
	vector<uchar> m_RANSACStatus;
	Mat m_Fundamental = findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC);
	//std::cout << "RANSAC fund:"<<endl<<m_Fundamental << endl;

	
	//去掉一部分点
	sift_matches_with_ShapeConst = sift_matches_origional;
	int kkk = 0;
	for (auto it = sift_matches_with_ShapeConst.begin(); it != sift_matches_with_ShapeConst.end();)
	{

		if (m_RANSACStatus[kkk] != 0)
		{
			it++;
		}
		else
		{
			it = sift_matches_with_ShapeConst.erase(it);

		}
		kkk++;
		if (it == sift_matches_with_ShapeConst.end())
			break;
	}
}