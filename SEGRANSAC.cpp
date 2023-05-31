#include"GMS.h"
#include"SEGRANSAC.h"



class MyPointMatch//对应点相关的信息
{
public:
	Point2f Pa;//在图A中的点坐标
	Point2f Pb;//在图B中的点坐标

	int SegBlockANO;//在图像1中图像块的编码
	int SegBlockBNO;//在图像2中图像块的编码

	Point2i PInSegBlockA1;//使用划分方法1，在图像块A中的相对位置坐标
	Point2i PInSegBlockA2;//使用划分方法2，在图像块A中的相对位置坐标

	Point2i PInSegBlockB1;//使用划分方法1，在图像块B中的相对位置坐标
	Point2i PInSegBlockB2;//使用划分方法2，在图像块B中的相对位置坐标

	float distance;
	int imgIdx;
	int queryIdx;
	int trainIdx;

	bool IsAdd;
	int wight;
	
};

class Seg7Block//对应点相关的信息
{
public:
	int SegBlockNO;
	int correspondSeg;
	int correspondSegNo;
	//int SegBlockANO;//在图像1中图像块的编码
	//int SegBlockBNO;//在图像2中图像块的编码

	vector<MyPointMatch> p;//指针指向具体的点对
	vector<int> NO;//指针指向具体的点对 
	vector<int> NO1;//指针指向具体的点对 
	Mat m_Fundamental;

	bool HaveFundamental;
	//图像A中
	Point2f LeftUpA;//左上角的特征点
	Point2f RightDownA;//右下角的特征点
	//图像B中
	Point2f LeftUpB;//左上角的特征点
	Point2f RightDownB;//右下角的特征点


	/////////////////////在所在图像中的划分
	vector<cv::Point2f> colTopA1;//划分方式1
	vector<cv::Point2f> colDownA1;
	vector<cv::Point2f> rawLeftA1;
	vector<cv::Point2f> rawRightA1;

	vector<cv::Point2f> colTopA2;//划分方式2
	vector<cv::Point2f> colDownA2;
	vector<cv::Point2f> rawLeftA2;
	vector<cv::Point2f> rawRightA2;


	/////////////////////在另一个图像中的划分
	vector<cv::Point2f> colTopB1;//划分方式1
	vector<cv::Point2f> colDownB1;
	vector<cv::Point2f> rawLeftB1;
	vector<cv::Point2f> rawRightB1;

	vector<cv::Point2f> colTopB2;//划分方式2
	vector<cv::Point2f> colDownB2;
	vector<cv::Point2f> rawLeftB2;
	vector<cv::Point2f> rawRightB2;


};


class JuLei//对应点相关的信息
{
public:
	vector<int> NO;//指针指向具体的点对 
	Mat m_Fundamental;
	int lastNoNum;
	bool IsHaveFund;
};

bool SatisfyEpipolarConst3(Mat m_Fundamental, int no, float T, const vector<DMatch> sift_matches_origional, vector<KeyPoint> sift_keypoints_1, vector<KeyPoint>  sift_keypoints_2)
{
	if (m_Fundamental.rows == 0 ||m_Fundamental.cols == 0)
		return false;


	Point2f m1 = sift_keypoints_1[sift_matches_origional[no].queryIdx].pt;
	Point2f	m2 = sift_keypoints_2[sift_matches_origional[no].trainIdx].pt;
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

	if (err <= T * T)
		return true;
	else
		return false;

}




void func(int IsFixFund, float conf, vector<Seg7Block>& SegBlockA, vector<Seg7Block>& SegBlockB, vector<MyPointMatch> correspoints,
	vector<KeyPoint> sift_keypoints_1, vector<KeyPoint>  sift_keypoints_2, vector<DMatch> sift_matches_origional, vector<DMatch>& Inliers);


void MatchesSelecter(int IsFixFund, Mat Image1, Mat Image2, float apha,
	const vector<DMatch> sift_matches_origional,
	vector<KeyPoint> sift_keypoints_1, vector<KeyPoint>  sift_keypoints_2,
	vector<DMatch>& InliersT1)
{


	//---------------------------------------图像分割--------------------------------------------------------------------------------
	int width1 = Image1.size().width;
	int height1 = Image1.size().height;

	int width2 = Image2.size().width;
	int height2 = Image2.size().height;

	universe* u1 = new universe(width1 * height1);
	universe* u2 = new universe(width2 * height2);

	int* num1 = new int(0);
	int* num2 = new int(0);

	Mat segimg1 = Image1.clone();
	Mat segimg2 = Image2.clone();

	/*
	* Segment an image
	* Returns a color image representing the segmentation.
	* sigma: to smooth the image.
	* min_size: minimum component size (enforced by post-processing stage).
	* SegImage(cv::Mat& image, universe* u, int* num, float sigma , float k , int min_size)
	*/

	SegImage(segimg1, u1, num1, 4, 200, 0);
	SegImage(segimg2, u2, num2, 4, 200, 0);

	vector<bool> IsRobust(sift_matches_origional.size(), false);






	GMSForVoter( Image1,  Image2,  6, 20, sift_matches_origional, IsRobust, sift_keypoints_1, sift_keypoints_2);



	vector<MyPointMatch> correspoints;
	vector<Seg7Block> SegBlockA;
	vector<Seg7Block> SegBlockB;


	for (int i = 0; i < sift_matches_origional.size(); i++)
	{

		//初始化对应点对 
		MyPointMatch tempCorrespondPoint;
		tempCorrespondPoint.Pa = sift_keypoints_1[sift_matches_origional[i].queryIdx].pt;
		tempCorrespondPoint.Pb = sift_keypoints_2[sift_matches_origional[i].trainIdx].pt;

		tempCorrespondPoint.SegBlockANO = u1->find(int(tempCorrespondPoint.Pa.y) * width1 + int(tempCorrespondPoint.Pa.x));
		tempCorrespondPoint.SegBlockBNO = u2->find(int(tempCorrespondPoint.Pb.y) * width2 + int(tempCorrespondPoint.Pb.x));

		tempCorrespondPoint.IsAdd = false;
		if(IsRobust[i]==true)
			tempCorrespondPoint.wight = 3;
		else
			tempCorrespondPoint.wight = 0.3;

		correspoints.push_back(tempCorrespondPoint);

		//分别计算两幅图中有多少个含有特征点的图像块
		//图A中的
		bool IsFindSegBlockANO = false;
		for (int j = 0; j < SegBlockA.size(); j++)
		{
			if (tempCorrespondPoint.SegBlockANO == SegBlockA[j].SegBlockNO)
			{
				IsFindSegBlockANO = true;
				SegBlockA[j].NO.push_back(i);
				break;
			}
		}
		if (IsFindSegBlockANO == false)
		{
			Seg7Block tempSegBlock;
			tempSegBlock.SegBlockNO = tempCorrespondPoint.SegBlockANO;
			tempSegBlock.NO.push_back(i);
			tempSegBlock.HaveFundamental = false;
			SegBlockA.push_back(tempSegBlock);
		}

		bool IsFindSegBlockBNO = false;
		for (int j = 0; j < SegBlockB.size(); j++)
		{
			if (tempCorrespondPoint.SegBlockBNO == SegBlockB[j].SegBlockNO)
			{
				IsFindSegBlockBNO = true;
				SegBlockB[j].NO.push_back(i);
				break;
			}
		}
		if (IsFindSegBlockBNO == false)
		{
			Seg7Block tempSegBlock;
			tempSegBlock.SegBlockNO = tempCorrespondPoint.SegBlockBNO;
			tempSegBlock.NO.push_back(i);
			tempSegBlock.HaveFundamental = false;
			SegBlockB.push_back(tempSegBlock);
		}
	}


	for (int i = 0; i < SegBlockA.size(); i++)
	{
		if (SegBlockA[i].NO.size() > 0)
		{
			vector<int> Voter(SegBlockB.size());
			for (int j = 0; j < SegBlockA[i].NO.size(); j++)
			{
				for (int k = 0; k < SegBlockB.size(); k++)
				{
					if (correspoints[SegBlockA[i].NO[j]].SegBlockBNO == SegBlockB[k].SegBlockNO)
					{
						Voter[k]+= correspoints[SegBlockA[i].NO[j]].wight;
						break;
					}
				}
			}

			std::vector<int>::iterator voternum = std::max_element(std::begin(Voter), std::end(Voter));
			int correspondSeg = std::distance(std::begin(Voter), voternum);

			SegBlockA[i].correspondSeg = SegBlockB[correspondSeg].SegBlockNO;
			SegBlockA[i].correspondSegNo = correspondSeg;

			float tau = apha * sqrt((float)SegBlockA[i].NO.size());
			
			if ((float)(*voternum) > tau)
			{
				for (auto it = SegBlockA[i].NO.begin(); it != SegBlockA[i].NO.end(); )
				{
					if (correspoints[*it].SegBlockBNO == SegBlockB[correspondSeg].SegBlockNO)
					{
						it++;
					}
					else
					{
						it = SegBlockA[i].NO.erase(it);
					}
					if (it == SegBlockA[i].NO.end())
						break;
				}
			}
			else
			{
				vector <int>().swap(SegBlockA[i].NO);
			}
		}
	}
	for (int i = 0; i < SegBlockB.size(); i++)
	{
		if (SegBlockB[i].NO.size() > 0)
		{
			vector<int> Voter(SegBlockA.size());
			for (int j = 0; j < SegBlockB[i].NO.size(); j++)
			{
				for (int k = 0; k < SegBlockA.size(); k++)
				{
					if (correspoints[SegBlockB[i].NO[j]].SegBlockANO == SegBlockA[k].SegBlockNO)
					{
						Voter[k] += correspoints[SegBlockB[i].NO[j]].wight;
						break;
					}
				}
			}
			std::vector<int>::iterator voternum = std::max_element(std::begin(Voter), std::end(Voter));
			int correspondSeg = std::distance(std::begin(Voter), voternum);
			SegBlockB[i].correspondSeg = SegBlockA[correspondSeg].SegBlockNO;
			SegBlockB[i].correspondSegNo = correspondSeg;
			float tau = apha * sqrt((float)SegBlockB[i].NO.size());
			
			if ((float)(*voternum) > tau)
			{
				for (auto it = SegBlockB[i].NO.begin(); it != SegBlockB[i].NO.end(); )
				{
					if (correspoints[*it].SegBlockANO == SegBlockA[correspondSeg].SegBlockNO)
					{
						it++;
					}
					else
					{
						it = SegBlockB[i].NO.erase(it);
					}
					if (it == SegBlockB[i].NO.end())
						break;
				}
			}
			else
			{
				vector <int>().swap(SegBlockB[i].NO);
			}
		}
	}

	func(IsFixFund, 0.1, SegBlockA, SegBlockB, correspoints, sift_keypoints_1, sift_keypoints_2, sift_matches_origional, InliersT1);

	delete u1;
	delete u2;
}


void func(int IsFixFund, float conf, vector<Seg7Block>& SegBlockA, vector<Seg7Block>& SegBlockB, vector<MyPointMatch> correspoints,
	vector<KeyPoint> sift_keypoints_1, vector<KeyPoint>  sift_keypoints_2, vector<DMatch> sift_matches_origional, vector<DMatch>& Inliers)
{
	if (IsFixFund == 0)
	{
		//当有多个的时候基础矩阵的时候
		vector <JuLei> FundMatch;
		for (int i = 0; i < SegBlockA.size(); i++)
		{
			if (SegBlockA[i].NO.size() >= 7)//10
			{
				std::vector<Point2f>CELLA;
				std::vector<Point2f>CELLB;

				for (int j = 0; j < SegBlockA[i].NO.size(); j++)
				{
					CELLA.push_back(sift_keypoints_1[sift_matches_origional[SegBlockA[i].NO[j]].queryIdx].pt);
					CELLB.push_back(sift_keypoints_2[sift_matches_origional[SegBlockA[i].NO[j]].trainIdx].pt);
				}

				vector<uchar> inliersMask;
				if (SegBlockA[i].NO.size() >= 7 && SegBlockA[i].NO.size() <= 14)
					SegBlockA[i].m_Fundamental = findFundamentalMat(CELLA, CELLB, inliersMask, FM_LMEDS, 3.0, conf);
				else
					SegBlockA[i].m_Fundamental = findFundamentalMat(CELLA, CELLB, inliersMask, FM_RANSAC, 3.0, conf);



				if (SegBlockA[i].m_Fundamental.rows == 0 || SegBlockA[i].m_Fundamental.cols == 0)
					SegBlockA[i].HaveFundamental = false;
				else
					SegBlockA[i].HaveFundamental = true;
				if (SegBlockA[i].HaveFundamental == true)
				{
					JuLei temp;
					temp.m_Fundamental = SegBlockA[i].m_Fundamental;
					temp.lastNoNum = 0;
					temp.IsHaveFund = true;
					FundMatch.push_back(temp);
				}


				for (size_t j = 0; j < inliersMask.size(); j++)
				{
					if (inliersMask[j])
					{
						correspoints[SegBlockA[i].NO[j]].IsAdd = true;
					}
				}
			}

		}

		for (int i = 0; i < SegBlockB.size(); i++)
		{
			if (SegBlockB[i].NO.size() >= 7)//10
			{
				std::vector<Point2f>CELLA;
				std::vector<Point2f>CELLB;

				for (int j = 0; j < SegBlockB[i].NO.size(); j++)
				{
					CELLA.push_back(sift_keypoints_1[sift_matches_origional[SegBlockB[i].NO[j]].queryIdx].pt);
					CELLB.push_back(sift_keypoints_2[sift_matches_origional[SegBlockB[i].NO[j]].trainIdx].pt);
				}
				vector<uchar> inliersMask;
				if (SegBlockB[i].NO.size() >= 7 && SegBlockB[i].NO.size() <= 14)
					SegBlockB[i].m_Fundamental = findFundamentalMat(CELLA, CELLB, inliersMask, FM_LMEDS, 3.0, conf);
				else
					SegBlockB[i].m_Fundamental = findFundamentalMat(CELLA, CELLB, inliersMask, FM_RANSAC, 3.0, conf);


				if (SegBlockB[i].m_Fundamental.rows == 0 || SegBlockB[i].m_Fundamental.cols == 0)
					SegBlockB[i].HaveFundamental = false;
				else
					SegBlockB[i].HaveFundamental = true;

				if (SegBlockB[i].HaveFundamental == true)
				{
					JuLei temp;
					temp.m_Fundamental = SegBlockB[i].m_Fundamental;
					temp.lastNoNum = 0;
					temp.IsHaveFund = true;
					FundMatch.push_back(temp);
				}
				for (size_t j = 0; j < inliersMask.size(); j++)
				{
					if (inliersMask[j] && correspoints[SegBlockB[i].NO[j]].IsAdd == false)
					{
						correspoints[SegBlockB[i].NO[j]].IsAdd = true;
					}
				}
			}
		}

		for (int l = 0; l < 1; l++)
		{
			int t = 3;
			//if (l < 7)
			//	t = 5;
			//else
			//	t = 3;

			for (int i = 0; i < FundMatch.size(); i++)
			{
				//if (FundMatch[i].IsHaveFund==true)
				{
					vector <int>().swap(FundMatch[i].NO);
					int num = 0;
					for (int j = 0; j < sift_matches_origional.size(); j++)
					{
						if (SatisfyEpipolarConst3(FundMatch[i].m_Fundamental, j, t, sift_matches_origional, sift_keypoints_1, sift_keypoints_2))
						{
							FundMatch[i].NO.push_back(j);
							num++;
						}
					}


					if (FundMatch[i].NO.size() < 7)// ((FundMatch[i].NO.size() < (sift_matches_gms.size()/35)) || (FundMatch[i].NO.size()<7))//150
					{
						FundMatch[i].IsHaveFund = false;
						continue;
					}
					else
					{
						FundMatch[i].IsHaveFund = true;
					}

					std::vector<Point2f>CELLA;
					std::vector<Point2f>CELLB;

					for (int j = 0; j < FundMatch[i].NO.size(); j++)
					{
						CELLA.push_back(sift_keypoints_1[sift_matches_origional[FundMatch[i].NO[j]].queryIdx].pt);
						CELLB.push_back(sift_keypoints_2[sift_matches_origional[FundMatch[i].NO[j]].trainIdx].pt);
					}
					vector<uchar> inliersMask;
					if (FundMatch[i].NO.size() >= 7 && FundMatch[i].NO.size() <= 14)
						FundMatch[i].m_Fundamental = findFundamentalMat(CELLA, CELLB, inliersMask, FM_LMEDS, 3.0, conf);
					else
						FundMatch[i].m_Fundamental = findFundamentalMat(CELLA, CELLB, inliersMask, FM_RANSAC, 3.0, conf);
				}
			}

		}




		for (int i = 0; i < sift_matches_origional.size(); i++)
		{
			for (int j = 0; j < FundMatch.size(); j++)
			{
				if (FundMatch[j].IsHaveFund && FundMatch[j].NO.size() > 15 && SatisfyEpipolarConst3(FundMatch[j].m_Fundamental, i, 3.0, sift_matches_origional, sift_keypoints_1, sift_keypoints_2))//35
				{
					Inliers.push_back(sift_matches_origional[i]);
					break;
				}
			}
		}
	}
	else
	{
		//当有一个基础矩阵的时候
		std::vector<Point2f>CELLA;
		std::vector<Point2f>CELLB;
		for (int i = 0; i < SegBlockA.size(); i++)
		{
			if (SegBlockA[i].NO.size() >= 7)//10
			{
				for (int j = 0; j < SegBlockA[i].NO.size(); j++)
				{
					CELLA.push_back(sift_keypoints_1[sift_matches_origional[SegBlockA[i].NO[j]].queryIdx].pt);
					CELLB.push_back(sift_keypoints_2[sift_matches_origional[SegBlockA[i].NO[j]].trainIdx].pt);
					correspoints[SegBlockA[i].NO[j]].IsAdd = true;
				}
			}
		}

		for (int i = 0; i < SegBlockB.size(); i++)
		{
			if (SegBlockB[i].NO.size() >= 7)//10
			{
				for (int j = 0; j < SegBlockB[i].NO.size(); j++)
				{
					if (correspoints[SegBlockB[i].NO[j]].IsAdd == false)
					{
						CELLA.push_back(sift_keypoints_1[sift_matches_origional[SegBlockB[i].NO[j]].queryIdx].pt);
						CELLB.push_back(sift_keypoints_2[sift_matches_origional[SegBlockB[i].NO[j]].trainIdx].pt);
					}
				}
			}
		}


		vector<uchar> inliersMask;
		Mat m_Fundamental;
		if (CELLA.size() >= 7 && CELLA.size() <= 14)
			m_Fundamental = findFundamentalMat(CELLA, CELLB, inliersMask, FM_LMEDS, 3.0, conf);
		else
			m_Fundamental = findFundamentalMat(CELLA, CELLB, inliersMask, FM_RANSAC, 3.0, conf);

		//sift_matches_gms  sift_matches_origional
		for (int i = 0; i < sift_matches_origional.size(); i++)
		{
			if (SatisfyEpipolarConst3(m_Fundamental, i, 3.0, sift_matches_origional, sift_keypoints_1, sift_keypoints_2))//35
			{
				Inliers.push_back(sift_matches_origional[i]);
			}
		}
	}
}