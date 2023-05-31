

#include"GMS.h"
//GMS

class Cell
{
public:
	int imgNo;//所在图像编号
	//Point2i coord;//该格子所在图像的坐标
	vector<int> p;//指针指向具体的点对
	int corrCell;//对应的最佳匹配块
};

void vfc(const vector<DMatch> sift_matches_origional, vector<KeyPoint> sift_keypoints_1, vector<KeyPoint>  sift_keypoints_2,
	vector<DMatch>& InliersT1, vector<DMatch>& InliersT2, vector<DMatch>& InliersT3, vector<DMatch>& InliersT4, vector<DMatch>& InliersT5,
	vector<DMatch>& InliersT6, vector<DMatch>& InliersT7, vector<DMatch>& InliersT8, vector<DMatch>& InliersT9, vector<DMatch>& InliersT10);


void MyGMS(Mat Image1, Mat Image2,vector<KeyPoint> sift_keypoints_1, vector<KeyPoint>  sift_keypoints_2,
	const vector<DMatch> sift_matches_origional, 
	vector<DMatch>& InliersT1, vector<DMatch>& InliersT2, vector<DMatch>& InliersT3, vector<DMatch>& InliersT4, vector<DMatch>& InliersT5,
	vector<DMatch>& InliersT6, vector<DMatch>& InliersT7, vector<DMatch>& InliersT8, vector<DMatch>& InliersT9, vector<DMatch>& InliersT10
	)
{
	GMS(Image1, Image2, 1, 20, sift_matches_origional, InliersT1, sift_keypoints_1, sift_keypoints_2);
	GMS(Image1, Image2, 2, 20, sift_matches_origional, InliersT2, sift_keypoints_1, sift_keypoints_2);
	GMS(Image1, Image2, 3, 20, sift_matches_origional, InliersT3, sift_keypoints_1, sift_keypoints_2);
	GMS(Image1, Image2, 4, 20, sift_matches_origional, InliersT4, sift_keypoints_1, sift_keypoints_2);
	GMS(Image1, Image2, 5, 20, sift_matches_origional, InliersT5, sift_keypoints_1, sift_keypoints_2);
	GMS(Image1, Image2, 6, 20, sift_matches_origional, InliersT6, sift_keypoints_1, sift_keypoints_2);
	GMS(Image1, Image2, 11, 20, sift_matches_origional, InliersT7, sift_keypoints_1, sift_keypoints_2);
	GMS(Image1, Image2, 12, 20, sift_matches_origional, InliersT8, sift_keypoints_1, sift_keypoints_2);
	GMS(Image1, Image2, 13, 20, sift_matches_origional, InliersT9, sift_keypoints_1, sift_keypoints_2);
	GMS(Image1, Image2, 14, 20, sift_matches_origional, InliersT10, sift_keypoints_1, sift_keypoints_2);
}
void GMS(Mat Image1, Mat Image2, float alpha,int cellnum,
	const vector<DMatch> sift_matches_origional, vector<DMatch>& Inliers,
	vector<KeyPoint> sift_keypoints_1, vector<KeyPoint>  sift_keypoints_2)
{
	vector<DMatch> tempInliers;
	vector <DMatch>().swap(Inliers);
	vector<DMatch> matches = sift_matches_origional;
	int wid1 = Image1.size().width;
	int hei1 = Image1.size().height;

	int wid2 = Image2.size().width;
	int hei2 = Image2.size().height;



	int CELLA_WIDTH = ((Image1.size().width % cellnum == 0) ? (Image1.size().width / cellnum) : (Image1.size().width / cellnum + 1));//一个细胞的宽度
	int CELLA_HEIGHT = ((Image1.size().height % cellnum == 0) ? (Image1.size().height / cellnum) : (Image1.size().height / cellnum + 1));//一个细胞的高度
	int CELLA_NUM_ROW = ((Image1.size().height % CELLA_HEIGHT == 0) ? (Image1.size().height / CELLA_HEIGHT) : (Image1.size().height / CELLA_HEIGHT + 1));//图a中细胞的行数
	int CELLA_NUM_COL = ((Image1.size().width % CELLA_WIDTH == 0) ? (Image1.size().width / CELLA_WIDTH) : (Image1.size().width / CELLA_WIDTH + 1));//图b中细胞的列数

	int CELLB_WIDTH = ((Image2.size().width % cellnum == 0) ? (Image2.size().width / cellnum) : (Image2.size().width / cellnum + 1));//一个细胞的宽度
	int CELLB_HEIGHT = ((Image2.size().height % cellnum == 0) ? (Image2.size().height / cellnum) : (Image2.size().height / cellnum + 1));//一个细胞的高度
	int CELLB_NUM_ROW = ((Image2.size().height % CELLB_HEIGHT == 0) ? (Image2.size().height / CELLB_HEIGHT) : (Image2.size().height / CELLB_HEIGHT + 1));//图a中细胞的行数
	int CELLB_NUM_COL = ((Image2.size().width % CELLB_WIDTH == 0) ? (Image2.size().width / CELLB_WIDTH) : (Image2.size().width / CELLB_WIDTH + 1));//图b中细胞的列数

	int HALFA_HEIGHT = CELLA_HEIGHT / 2;
	int HALFA_WIDTH = CELLA_WIDTH / 2;

	int HALFB_HEIGHT = CELLB_HEIGHT / 2;
	int HALFB_WIDTH = CELLB_WIDTH / 2;

	//划分方式一
	vector<Cell> cellA(CELLA_NUM_ROW * CELLA_NUM_COL);//图A中的表格
	vector<Cell> cellB(CELLB_NUM_ROW * CELLB_NUM_COL);//图B中的表格
	//划分方式二
	vector<Cell> cellC((CELLA_NUM_ROW + 1) * (CELLA_NUM_COL + 1));//图A中的表格
	vector<Cell> cellD((CELLB_NUM_ROW + 1) * (CELLB_NUM_COL + 1));//图B中的表格

	vector<bool> IsAdd(matches.size());//表示对应的匹配有没有添加到内点集合中

	for (int i = 0; i < matches.size(); i++)
	{
		int temprow = int(sift_keypoints_1[matches[i].queryIdx].pt.y) / CELLA_HEIGHT;
		int tempcol = int(sift_keypoints_1[matches[i].queryIdx].pt.x) / CELLA_WIDTH;
		cellA[temprow * CELLA_NUM_ROW + tempcol].p.push_back(i);

		temprow = int(sift_keypoints_1[matches[i].queryIdx].pt.y + HALFA_HEIGHT) / CELLA_HEIGHT;
		tempcol = int(sift_keypoints_1[matches[i].queryIdx].pt.x + HALFA_WIDTH) / CELLA_WIDTH;
		cellC[temprow * CELLA_NUM_ROW + tempcol].p.push_back(i);

		IsAdd[i] = false;
	}

	for (int i = 0; i < CELLA_NUM_ROW * CELLA_NUM_COL; i++)
	{
		if (cellA[i].p.size() > 0)
		{
			vector<int> Voter(CELLB_NUM_ROW * CELLB_NUM_COL);
			for (int j = 0; j < cellA[i].p.size(); j++)
			{
				int col = int(sift_keypoints_2[matches[cellA[i].p[j]].trainIdx].pt.x) / CELLB_WIDTH;
				int row = int(sift_keypoints_2[matches[cellA[i].p[j]].trainIdx].pt.y) / CELLB_HEIGHT;
				Voter[row * CELLB_NUM_ROW + col]++;
			}

			std::vector<int>::iterator biggest = std::max_element(std::begin(Voter), std::end(Voter));
			int correspondCell = std::distance(std::begin(Voter), biggest);
			int i5 = *biggest;
			int i5num = cellA[i].p.size();
			int num = 1;

			int i1 = 0;
			int i1num = 0;
			int tempif = i - CELLA_NUM_COL - 1;
			if (tempif >= 0 && correspondCell - CELLB_NUM_COL - 1 >= 0 && correspondCell - CELLB_NUM_COL - 1 < CELLB_NUM_ROW * CELLB_NUM_COL)//i1
			{
				i1num = cellA[tempif].p.size();
				num++;

				for (int j = 0; j < cellA[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[matches[cellA[tempif].p[j]].trainIdx].pt.x) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[matches[cellA[tempif].p[j]].trainIdx].pt.y) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell - CELLB_NUM_COL - 1)
						i1++;
				}
			}

			int i2 = 0;
			int i2num = 0;

			tempif = i - CELLA_NUM_COL;
			if (tempif >= 0 && correspondCell - CELLB_NUM_COL >= 0 && correspondCell - CELLB_NUM_COL < CELLB_NUM_ROW * CELLB_NUM_COL)//i1
			{
				i2num = cellA[tempif].p.size();
				num++;

				for (int j = 0; j < cellA[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[matches[cellA[tempif].p[j]].trainIdx].pt.x) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[matches[cellA[tempif].p[j]].trainIdx].pt.y) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell - CELLB_NUM_COL)
						i2++;
				}
			}

			int i3 = 0;
			int i3num = 0;
			tempif = i - CELLA_NUM_COL + 1;
			if (tempif >= 0 && correspondCell - CELLB_NUM_COL + 1 >= 0 && correspondCell - CELLB_NUM_COL + 1 < CELLB_NUM_ROW * CELLB_NUM_COL)//i3
			{
				i3num = cellA[tempif].p.size();
				num++;
				for (int j = 0; j < cellA[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[matches[cellA[tempif].p[j]].trainIdx].pt.x) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[matches[cellA[tempif].p[j]].trainIdx].pt.y) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell - CELLB_NUM_COL + 1)
						i3++;
				}
			}

			int i4 = 0;
			int i4num = 0;

			tempif = i - 1;
			if (tempif >= 0 && correspondCell - 1 >= 0 && correspondCell - 1 < CELLB_NUM_ROW * CELLB_NUM_COL)//i4
			{
				i4num = cellA[tempif].p.size();
				num++;
				for (int j = 0; j < cellA[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[matches[cellA[tempif].p[j]].trainIdx].pt.x) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[matches[cellA[tempif].p[j]].trainIdx].pt.y) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell - 1)
						i4++;
				}
			}

			int i6 = 0;
			int i6num = 0;
			tempif = i + 1;
			if (tempif < CELLA_NUM_ROW * CELLA_NUM_COL && correspondCell + 1 >= 0 && correspondCell + 1 < CELLB_NUM_ROW * CELLB_NUM_COL)//i4
			{
				i6num = cellA[tempif].p.size();
				num++;
				for (int j = 0; j < cellA[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[matches[cellA[tempif].p[j]].trainIdx].pt.x) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[matches[cellA[tempif].p[j]].trainIdx].pt.y) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell + 1)
						i6++;
				}
			}

			int i7 = 0;
			int i7num = 0;
			tempif = i + CELLA_NUM_COL - 1;
			if (tempif < CELLA_NUM_ROW * CELLA_NUM_COL && correspondCell + CELLB_NUM_COL - 1 >= 0 && correspondCell + CELLB_NUM_COL - 1 < CELLB_NUM_ROW * CELLB_NUM_COL)//i4
			{
				i7num = cellA[tempif].p.size();
				num++;
				for (int j = 0; j < cellA[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[matches[cellA[tempif].p[j]].trainIdx].pt.x) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[matches[cellA[tempif].p[j]].trainIdx].pt.y) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell + CELLB_NUM_COL - 1)
						i7++;
				}
			}

			int i8 = 0;
			int i8num = 0;
			tempif = i + CELLA_NUM_COL;
			if (tempif < CELLA_NUM_ROW * CELLA_NUM_COL && correspondCell + CELLB_NUM_COL >= 0 && correspondCell + CELLB_NUM_COL < CELLB_NUM_ROW * CELLB_NUM_COL)//i4
			{
				i8num = cellA[tempif].p.size();
				num++;
				for (int j = 0; j < cellA[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[matches[cellA[tempif].p[j]].trainIdx].pt.x) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[matches[cellA[tempif].p[j]].trainIdx].pt.y) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell + CELLB_NUM_COL)
						i8++;
				}
			}

			int i9 = 0;
			int i9num = 0;
			tempif = i + CELLA_NUM_COL + 1;
			if (tempif < CELLA_NUM_ROW * CELLA_NUM_COL && correspondCell + CELLB_NUM_COL + 1 >= 0 && correspondCell + CELLB_NUM_COL + 1 < CELLB_NUM_ROW * CELLB_NUM_COL)//i4
			{
				i9num = cellA[tempif].p.size();
				num++;
				for (int j = 0; j < cellA[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[matches[cellA[tempif].p[j]].trainIdx].pt.x) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[matches[cellA[tempif].p[j]].trainIdx].pt.y) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell + CELLB_NUM_COL + 1)
						i9++;
				}
			}

			int sum = i1 + i2 + i3 + i4 + i5 + i6 + i7 + i8 + i9;
			float tau = alpha * sqrt((float)(i1num + i2num + i3num + i4num + i5num + i6num + i7num + i8num + i9num) / ((float)num));
			if (sum > tau)
			{
				for (auto it = cellA[i].p.begin(); it != cellA[i].p.end(); )
				{

					int col = int(sift_keypoints_2[matches[*it].trainIdx].pt.x) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[matches[*it].trainIdx].pt.y) / CELLB_HEIGHT;
					if (row * CELLB_NUM_COL + col == correspondCell)
					{
						Inliers.push_back(matches[*it]);
						IsAdd[*it] = true;
					}
					it++;
					if (it == cellA[i].p.end())
						break;
				}
			}
			else
			{
				vector <int>().swap(cellA[i].p);
			}
		}
	}

	//////////////////////////////////

	CELLA_NUM_ROW++;
	CELLA_NUM_COL++;
	CELLB_NUM_ROW++;
	CELLB_NUM_COL++;

	for (int i = 0; i < CELLA_NUM_ROW * CELLA_NUM_COL; i++)
	{
		if (cellC[i].p.size() > 0)
		{
			vector<int> Voter(CELLB_NUM_ROW * CELLB_NUM_COL);
			for (int j = 0; j < cellC[i].p.size(); j++)
			{
				int col = int(sift_keypoints_2[matches[cellC[i].p[j]].trainIdx].pt.x + HALFB_WIDTH) / CELLB_WIDTH;
				int row = int(sift_keypoints_2[matches[cellC[i].p[j]].trainIdx].pt.y + HALFB_HEIGHT) / CELLB_HEIGHT;
				Voter[row * CELLB_NUM_ROW + col]++;
			}

			std::vector<int>::iterator biggest = std::max_element(std::begin(Voter), std::end(Voter));
			int correspondCell = std::distance(std::begin(Voter), biggest);
			int i5 = *biggest;
			int i5num = cellC[i].p.size();
			int num = 1;

			int i1 = 0;
			int i1num = 0;
			int tempif = i - CELLA_NUM_COL + 1 - 1;
			if (tempif >= 0 && correspondCell - CELLB_NUM_COL - 1 >= 0 && correspondCell - CELLB_NUM_COL - 1 < CELLB_NUM_ROW * CELLB_NUM_COL)//i1
			{
				i1num = cellC[tempif].p.size();
				num++;

				for (int j = 0; j < cellC[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[matches[cellC[tempif].p[j]].trainIdx].pt.x + HALFB_WIDTH) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[matches[cellC[tempif].p[j]].trainIdx].pt.y + HALFB_HEIGHT) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell - CELLB_NUM_COL - 1)
						i1++;
				}
			}

			int i2 = 0;
			int i2num = 0;

			tempif = i - CELLA_NUM_COL;
			if (tempif >= 0 && correspondCell - CELLB_NUM_COL >= 0 && correspondCell - CELLB_NUM_COL < CELLB_NUM_ROW * CELLB_NUM_COL)//i1
			{
				i2num = cellC[tempif].p.size();
				num++;

				for (int j = 0; j < cellC[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[matches[cellC[tempif].p[j]].trainIdx].pt.x + HALFB_WIDTH) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[matches[cellC[tempif].p[j]].trainIdx].pt.y + HALFB_HEIGHT) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell - CELLB_NUM_COL)
						i2++;
				}
			}

			int i3 = 0;
			int i3num = 0;
			tempif = i - CELLA_NUM_COL + 1;
			if (tempif >= 0 && correspondCell - CELLB_NUM_COL + 1 >= 0 && correspondCell - CELLB_NUM_COL + 1 < CELLB_NUM_ROW * CELLB_NUM_COL)//i3
			{
				i3num = cellC[tempif].p.size();
				num++;
				for (int j = 0; j < cellC[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[matches[cellC[tempif].p[j]].trainIdx].pt.x + HALFB_WIDTH) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[matches[cellC[tempif].p[j]].trainIdx].pt.y + HALFB_HEIGHT) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell - CELLB_NUM_COL + 1)
						i3++;
				}
			}

			int i4 = 0;
			int i4num = 0;

			tempif = i - 1;
			if (tempif >= 0 && correspondCell - 1 >= 0 && correspondCell - 1 < CELLB_NUM_ROW * CELLB_NUM_COL)//i4
			{
				i4num = cellC[tempif].p.size();
				num++;
				for (int j = 0; j < cellC[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[matches[cellC[tempif].p[j]].trainIdx].pt.x + HALFB_WIDTH) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[matches[cellC[tempif].p[j]].trainIdx].pt.y + HALFB_HEIGHT) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell - 1)
						i4++;
				}
			}

			int i6 = 0;
			int i6num = 0;
			tempif = i + 1;
			if (tempif < CELLA_NUM_ROW * CELLA_NUM_COL && correspondCell + 1 >= 0 && correspondCell + 1 < CELLB_NUM_ROW * CELLB_NUM_COL)//i4
			{
				i6num = cellC[tempif].p.size();
				num++;
				for (int j = 0; j < cellC[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[matches[cellC[tempif].p[j]].trainIdx].pt.x + HALFB_WIDTH) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[matches[cellC[tempif].p[j]].trainIdx].pt.y + HALFB_HEIGHT) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell + 1)
						i6++;
				}
			}

			int i7 = 0;
			int i7num = 0;
			tempif = i + CELLA_NUM_COL - 1;
			if (tempif < CELLA_NUM_ROW * CELLA_NUM_COL && correspondCell + CELLB_NUM_COL - 1 >= 0 && correspondCell + CELLB_NUM_COL - 1 < CELLB_NUM_ROW * CELLB_NUM_COL)//i4
			{
				i7num = cellC[tempif].p.size();
				num++;
				for (int j = 0; j < cellC[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[matches[cellC[tempif].p[j]].trainIdx].pt.x + HALFB_WIDTH) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[matches[cellC[tempif].p[j]].trainIdx].pt.y + HALFB_HEIGHT) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell + CELLB_NUM_COL - 1)
						i7++;
				}
			}

			int i8 = 0;
			int i8num = 0;
			tempif = i + CELLA_NUM_COL;
			if (tempif < CELLA_NUM_ROW * CELLA_NUM_COL && correspondCell + CELLB_NUM_COL >= 0 && correspondCell + CELLB_NUM_COL < CELLB_NUM_ROW * CELLB_NUM_COL)//i4
			{
				i8num = cellC[tempif].p.size();
				num++;
				for (int j = 0; j < cellC[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[matches[cellC[tempif].p[j]].trainIdx].pt.x + HALFB_WIDTH) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[matches[cellC[tempif].p[j]].trainIdx].pt.y + HALFB_HEIGHT) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell + CELLB_NUM_COL)
						i8++;
				}
			}

			int i9 = 0;
			int i9num = 0;
			tempif = i + CELLA_NUM_COL + 1;
			if (tempif < CELLA_NUM_ROW * CELLA_NUM_COL && correspondCell + CELLB_NUM_COL + 1 >= 0 && correspondCell + CELLB_NUM_COL + 1 < CELLB_NUM_ROW * CELLB_NUM_COL)//i4
			{
				i9num = cellC[tempif].p.size();
				num++;
				for (int j = 0; j < cellC[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[matches[cellC[tempif].p[j]].trainIdx].pt.x + HALFB_WIDTH) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[matches[cellC[tempif].p[j]].trainIdx].pt.y + HALFB_HEIGHT) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell + CELLB_NUM_COL + 1)
						i9++;
				}
			}

			int sum = i1 + i2 + i3 + i4 + i5 + i6 + i7 + i8 + i9;
			float tau = alpha * sqrt((float)(i1num + i2num + i3num + i4num + i5num + i6num + i7num + i8num + i9num) / ((float)num));
			if (sum > tau)
			{
				for (auto it = cellC[i].p.begin(); it != cellC[i].p.end(); )
				{
					int col = int(sift_keypoints_2[matches[*it].trainIdx].pt.x + HALFB_WIDTH) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[matches[*it].trainIdx].pt.y + HALFB_HEIGHT) / CELLB_HEIGHT;
					if (row * CELLB_NUM_COL + col == correspondCell && IsAdd[*it] == false)
					{
						Inliers.push_back(matches[*it]);
						IsAdd[*it] =true;
					}
					it++;
					if (it == cellC[i].p.end())
						break;
				}
			}
			else
			{
				vector <int>().swap(cellC[i].p);
			}
		}
	}
}












//-----------------------------GMS for Voter-------------


void GMSForVoter(Mat Image1, Mat Image2, float alpha, int cellnum,
	const vector<DMatch> sift_matches_origional, vector<bool>& IsRobust,
	vector<KeyPoint> sift_keypoints_1, vector<KeyPoint>  sift_keypoints_2)
{

	int wid1 = Image1.size().width;
	int hei1 = Image1.size().height;

	int wid2 = Image2.size().width;
	int hei2 = Image2.size().height;



	int CELLA_WIDTH = ((Image1.size().width % cellnum == 0) ? (Image1.size().width / cellnum) : (Image1.size().width / cellnum + 1));//一个细胞的宽度
	int CELLA_HEIGHT = ((Image1.size().height % cellnum == 0) ? (Image1.size().height / cellnum) : (Image1.size().height / cellnum + 1));//一个细胞的高度
	int CELLA_NUM_ROW = ((Image1.size().height % CELLA_HEIGHT == 0) ? (Image1.size().height / CELLA_HEIGHT) : (Image1.size().height / CELLA_HEIGHT + 1));//图a中细胞的行数
	int CELLA_NUM_COL = ((Image1.size().width % CELLA_WIDTH == 0) ? (Image1.size().width / CELLA_WIDTH) : (Image1.size().width / CELLA_WIDTH + 1));//图b中细胞的列数

	int CELLB_WIDTH = ((Image2.size().width % cellnum == 0) ? (Image2.size().width / cellnum) : (Image2.size().width / cellnum + 1));//一个细胞的宽度
	int CELLB_HEIGHT = ((Image2.size().height % cellnum == 0) ? (Image2.size().height / cellnum) : (Image2.size().height / cellnum + 1));//一个细胞的高度
	int CELLB_NUM_ROW = ((Image2.size().height % CELLB_HEIGHT == 0) ? (Image2.size().height / CELLB_HEIGHT) : (Image2.size().height / CELLB_HEIGHT + 1));//图a中细胞的行数
	int CELLB_NUM_COL = ((Image2.size().width % CELLB_WIDTH == 0) ? (Image2.size().width / CELLB_WIDTH) : (Image2.size().width / CELLB_WIDTH + 1));//图b中细胞的列数

	int HALFA_HEIGHT = CELLA_HEIGHT / 2;
	int HALFA_WIDTH = CELLA_WIDTH / 2;

	int HALFB_HEIGHT = CELLB_HEIGHT / 2;
	int HALFB_WIDTH = CELLB_WIDTH / 2;

	//划分方式一
	vector<Cell> cellA(CELLA_NUM_ROW * CELLA_NUM_COL);//图A中的表格
	vector<Cell> cellB(CELLB_NUM_ROW * CELLB_NUM_COL);//图B中的表格
	//划分方式二
	vector<Cell> cellC((CELLA_NUM_ROW + 1) * (CELLA_NUM_COL + 1));//图A中的表格
	vector<Cell> cellD((CELLB_NUM_ROW + 1) * (CELLB_NUM_COL + 1));//图B中的表格

	vector<bool> IsAdd(sift_matches_origional.size());//表示对应的匹配有没有添加到内点集合中

	for (int i = 0; i < sift_matches_origional.size(); i++)
	{
		int temprow = int(sift_keypoints_1[sift_matches_origional[i].queryIdx].pt.y) / CELLA_HEIGHT;
		int tempcol = int(sift_keypoints_1[sift_matches_origional[i].queryIdx].pt.x) / CELLA_WIDTH;
		cellA[temprow * CELLA_NUM_ROW + tempcol].p.push_back(i);

		temprow = int(sift_keypoints_1[sift_matches_origional[i].queryIdx].pt.y + HALFA_HEIGHT) / CELLA_HEIGHT;
		tempcol = int(sift_keypoints_1[sift_matches_origional[i].queryIdx].pt.x + HALFA_WIDTH) / CELLA_WIDTH;
		cellC[temprow * CELLA_NUM_ROW + tempcol].p.push_back(i);

		IsAdd[i] = false;
	}

	for (int i = 0; i < CELLA_NUM_ROW * CELLA_NUM_COL; i++)
	{
		if (cellA[i].p.size() > 0)
		{
			vector<int> Voter(CELLB_NUM_ROW * CELLB_NUM_COL);
			for (int j = 0; j < cellA[i].p.size(); j++)
			{
				int col = int(sift_keypoints_2[sift_matches_origional[cellA[i].p[j]].trainIdx].pt.x) / CELLB_WIDTH;
				int row = int(sift_keypoints_2[sift_matches_origional[cellA[i].p[j]].trainIdx].pt.y) / CELLB_HEIGHT;
				Voter[row * CELLB_NUM_ROW + col]++;
			}

			std::vector<int>::iterator biggest = std::max_element(std::begin(Voter), std::end(Voter));
			int correspondCell = std::distance(std::begin(Voter), biggest);
			int i5 = *biggest;
			int i5num = cellA[i].p.size();
			int num = 1;

			int i1 = 0;
			int i1num = 0;
			int tempif = i - CELLA_NUM_COL - 1;
			if (tempif >= 0 && correspondCell - CELLB_NUM_COL - 1 >= 0 && correspondCell - CELLB_NUM_COL - 1 < CELLB_NUM_ROW * CELLB_NUM_COL)//i1
			{
				i1num = cellA[tempif].p.size();
				num++;

				for (int j = 0; j < cellA[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[sift_matches_origional[cellA[tempif].p[j]].trainIdx].pt.x) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[sift_matches_origional[cellA[tempif].p[j]].trainIdx].pt.y) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell - CELLB_NUM_COL - 1)
						i1++;
				}
			}

			int i2 = 0;
			int i2num = 0;

			tempif = i - CELLA_NUM_COL;
			if (tempif >= 0 && correspondCell - CELLB_NUM_COL >= 0 && correspondCell - CELLB_NUM_COL < CELLB_NUM_ROW * CELLB_NUM_COL)//i1
			{
				i2num = cellA[tempif].p.size();
				num++;

				for (int j = 0; j < cellA[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[sift_matches_origional[cellA[tempif].p[j]].trainIdx].pt.x) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[sift_matches_origional[cellA[tempif].p[j]].trainIdx].pt.y) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell - CELLB_NUM_COL)
						i2++;
				}
			}

			int i3 = 0;
			int i3num = 0;
			tempif = i - CELLA_NUM_COL + 1;
			if (tempif >= 0 && correspondCell - CELLB_NUM_COL + 1 >= 0 && correspondCell - CELLB_NUM_COL + 1 < CELLB_NUM_ROW * CELLB_NUM_COL)//i3
			{
				i3num = cellA[tempif].p.size();
				num++;
				for (int j = 0; j < cellA[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[sift_matches_origional[cellA[tempif].p[j]].trainIdx].pt.x) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[sift_matches_origional[cellA[tempif].p[j]].trainIdx].pt.y) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell - CELLB_NUM_COL + 1)
						i3++;
				}
			}

			int i4 = 0;
			int i4num = 0;

			tempif = i - 1;
			if (tempif >= 0 && correspondCell - 1 >= 0 && correspondCell - 1 < CELLB_NUM_ROW * CELLB_NUM_COL)//i4
			{
				i4num = cellA[tempif].p.size();
				num++;
				for (int j = 0; j < cellA[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[sift_matches_origional[cellA[tempif].p[j]].trainIdx].pt.x) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[sift_matches_origional[cellA[tempif].p[j]].trainIdx].pt.y) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell - 1)
						i4++;
				}
			}

			int i6 = 0;
			int i6num = 0;
			tempif = i + 1;
			if (tempif < CELLA_NUM_ROW * CELLA_NUM_COL && correspondCell + 1 >= 0 && correspondCell + 1 < CELLB_NUM_ROW * CELLB_NUM_COL)//i4
			{
				i6num = cellA[tempif].p.size();
				num++;
				for (int j = 0; j < cellA[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[sift_matches_origional[cellA[tempif].p[j]].trainIdx].pt.x) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[sift_matches_origional[cellA[tempif].p[j]].trainIdx].pt.y) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell + 1)
						i6++;
				}
			}

			int i7 = 0;
			int i7num = 0;
			tempif = i + CELLA_NUM_COL - 1;
			if (tempif < CELLA_NUM_ROW * CELLA_NUM_COL && correspondCell + CELLB_NUM_COL - 1 >= 0 && correspondCell + CELLB_NUM_COL - 1 < CELLB_NUM_ROW * CELLB_NUM_COL)//i4
			{
				i7num = cellA[tempif].p.size();
				num++;
				for (int j = 0; j < cellA[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[sift_matches_origional[cellA[tempif].p[j]].trainIdx].pt.x) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[sift_matches_origional[cellA[tempif].p[j]].trainIdx].pt.y) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell + CELLB_NUM_COL - 1)
						i7++;
				}
			}

			int i8 = 0;
			int i8num = 0;
			tempif = i + CELLA_NUM_COL;
			if (tempif < CELLA_NUM_ROW * CELLA_NUM_COL && correspondCell + CELLB_NUM_COL >= 0 && correspondCell + CELLB_NUM_COL < CELLB_NUM_ROW * CELLB_NUM_COL)//i4
			{
				i8num = cellA[tempif].p.size();
				num++;
				for (int j = 0; j < cellA[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[sift_matches_origional[cellA[tempif].p[j]].trainIdx].pt.x) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[sift_matches_origional[cellA[tempif].p[j]].trainIdx].pt.y) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell + CELLB_NUM_COL)
						i8++;
				}
			}

			int i9 = 0;
			int i9num = 0;
			tempif = i + CELLA_NUM_COL + 1;
			if (tempif < CELLA_NUM_ROW * CELLA_NUM_COL && correspondCell + CELLB_NUM_COL + 1 >= 0 && correspondCell + CELLB_NUM_COL + 1 < CELLB_NUM_ROW * CELLB_NUM_COL)//i4
			{
				i9num = cellA[tempif].p.size();
				num++;
				for (int j = 0; j < cellA[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[sift_matches_origional[cellA[tempif].p[j]].trainIdx].pt.x) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[sift_matches_origional[cellA[tempif].p[j]].trainIdx].pt.y) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell + CELLB_NUM_COL + 1)
						i9++;
				}
			}

			int sum = i1 + i2 + i3 + i4 + i5 + i6 + i7 + i8 + i9;
			float tau = alpha * sqrt((float)(i1num + i2num + i3num + i4num + i5num + i6num + i7num + i8num + i9num) / ((float)num));
			if (sum > tau)
			{
				for (auto it = cellA[i].p.begin(); it != cellA[i].p.end(); )
				{
					int col = int(sift_keypoints_2[sift_matches_origional[*it].trainIdx].pt.x) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[sift_matches_origional[*it].trainIdx].pt.y) / CELLB_HEIGHT;
					if (row * CELLB_NUM_COL + col == correspondCell)
					{
						IsRobust[*it] = true;
					}
					it++;
					if (it == cellA[i].p.end())
						break;
				}
			}
			else
			{
				vector <int>().swap(cellA[i].p);
			}
		}
	}

	//////////////////////////////////

	CELLA_NUM_ROW++;
	CELLA_NUM_COL++;
	CELLB_NUM_ROW++;
	CELLB_NUM_COL++;

	for (int i = 0; i < CELLA_NUM_ROW * CELLA_NUM_COL; i++)
	{
		if (cellC[i].p.size() > 0)
		{
			vector<int> Voter(CELLB_NUM_ROW * CELLB_NUM_COL);
			for (int j = 0; j < cellC[i].p.size(); j++)
			{
				int col = int(sift_keypoints_2[sift_matches_origional[cellC[i].p[j]].trainIdx].pt.x + HALFB_WIDTH) / CELLB_WIDTH;
				int row = int(sift_keypoints_2[sift_matches_origional[cellC[i].p[j]].trainIdx].pt.y + HALFB_HEIGHT) / CELLB_HEIGHT;
				Voter[row * CELLB_NUM_ROW + col]++;
			}

			std::vector<int>::iterator biggest = std::max_element(std::begin(Voter), std::end(Voter));
			int correspondCell = std::distance(std::begin(Voter), biggest);
			int i5 = *biggest;
			int i5num = cellC[i].p.size();
			int num = 1;

			int i1 = 0;
			int i1num = 0;
			int tempif = i - CELLA_NUM_COL + 1 - 1;
			if (tempif >= 0 && correspondCell - CELLB_NUM_COL - 1 >= 0 && correspondCell - CELLB_NUM_COL - 1 < CELLB_NUM_ROW * CELLB_NUM_COL)//i1
			{
				i1num = cellC[tempif].p.size();
				num++;

				for (int j = 0; j < cellC[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[sift_matches_origional[cellC[tempif].p[j]].trainIdx].pt.x + HALFB_WIDTH) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[sift_matches_origional[cellC[tempif].p[j]].trainIdx].pt.y + HALFB_HEIGHT) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell - CELLB_NUM_COL - 1)
						i1++;
				}
			}

			int i2 = 0;
			int i2num = 0;

			tempif = i - CELLA_NUM_COL;
			if (tempif >= 0 && correspondCell - CELLB_NUM_COL >= 0 && correspondCell - CELLB_NUM_COL < CELLB_NUM_ROW * CELLB_NUM_COL)//i1
			{
				i2num = cellC[tempif].p.size();
				num++;

				for (int j = 0; j < cellC[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[sift_matches_origional[cellC[tempif].p[j]].trainIdx].pt.x + HALFB_WIDTH) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[sift_matches_origional[cellC[tempif].p[j]].trainIdx].pt.y + HALFB_HEIGHT) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell - CELLB_NUM_COL)
						i2++;
				}
			}

			int i3 = 0;
			int i3num = 0;
			tempif = i - CELLA_NUM_COL + 1;
			if (tempif >= 0 && correspondCell - CELLB_NUM_COL + 1 >= 0 && correspondCell - CELLB_NUM_COL + 1 < CELLB_NUM_ROW * CELLB_NUM_COL)//i3
			{
				i3num = cellC[tempif].p.size();
				num++;
				for (int j = 0; j < cellC[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[sift_matches_origional[cellC[tempif].p[j]].trainIdx].pt.x + HALFB_WIDTH) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[sift_matches_origional[cellC[tempif].p[j]].trainIdx].pt.y + HALFB_HEIGHT) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell - CELLB_NUM_COL + 1)
						i3++;
				}
			}

			int i4 = 0;
			int i4num = 0;

			tempif = i - 1;
			if (tempif >= 0 && correspondCell - 1 >= 0 && correspondCell - 1 < CELLB_NUM_ROW * CELLB_NUM_COL)//i4
			{
				i4num = cellC[tempif].p.size();
				num++;
				for (int j = 0; j < cellC[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[sift_matches_origional[cellC[tempif].p[j]].trainIdx].pt.x + HALFB_WIDTH) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[sift_matches_origional[cellC[tempif].p[j]].trainIdx].pt.y + HALFB_HEIGHT) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell - 1)
						i4++;
				}
			}

			int i6 = 0;
			int i6num = 0;
			tempif = i + 1;
			if (tempif < CELLA_NUM_ROW * CELLA_NUM_COL && correspondCell + 1 >= 0 && correspondCell + 1 < CELLB_NUM_ROW * CELLB_NUM_COL)//i4
			{
				i6num = cellC[tempif].p.size();
				num++;
				for (int j = 0; j < cellC[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[sift_matches_origional[cellC[tempif].p[j]].trainIdx].pt.x + HALFB_WIDTH) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[sift_matches_origional[cellC[tempif].p[j]].trainIdx].pt.y + HALFB_HEIGHT) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell + 1)
						i6++;
				}
			}

			int i7 = 0;
			int i7num = 0;
			tempif = i + CELLA_NUM_COL - 1;
			if (tempif < CELLA_NUM_ROW * CELLA_NUM_COL && correspondCell + CELLB_NUM_COL - 1 >= 0 && correspondCell + CELLB_NUM_COL - 1 < CELLB_NUM_ROW * CELLB_NUM_COL)//i4
			{
				i7num = cellC[tempif].p.size();
				num++;
				for (int j = 0; j < cellC[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[sift_matches_origional[cellC[tempif].p[j]].trainIdx].pt.x + HALFB_WIDTH) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[sift_matches_origional[cellC[tempif].p[j]].trainIdx].pt.y + HALFB_HEIGHT) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell + CELLB_NUM_COL - 1)
						i7++;
				}
			}

			int i8 = 0;
			int i8num = 0;
			tempif = i + CELLA_NUM_COL;
			if (tempif < CELLA_NUM_ROW * CELLA_NUM_COL && correspondCell + CELLB_NUM_COL >= 0 && correspondCell + CELLB_NUM_COL < CELLB_NUM_ROW * CELLB_NUM_COL)//i4
			{
				i8num = cellC[tempif].p.size();
				num++;
				for (int j = 0; j < cellC[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[sift_matches_origional[cellC[tempif].p[j]].trainIdx].pt.x + HALFB_WIDTH) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[sift_matches_origional[cellC[tempif].p[j]].trainIdx].pt.y + HALFB_HEIGHT) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell + CELLB_NUM_COL)
						i8++;
				}
			}

			int i9 = 0;
			int i9num = 0;
			tempif = i + CELLA_NUM_COL + 1;
			if (tempif < CELLA_NUM_ROW * CELLA_NUM_COL && correspondCell + CELLB_NUM_COL + 1 >= 0 && correspondCell + CELLB_NUM_COL + 1 < CELLB_NUM_ROW * CELLB_NUM_COL)//i4
			{
				i9num = cellC[tempif].p.size();
				num++;
				for (int j = 0; j < cellC[tempif].p.size(); j++)
				{
					int col = int(sift_keypoints_2[sift_matches_origional[cellC[tempif].p[j]].trainIdx].pt.x + HALFB_WIDTH) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[sift_matches_origional[cellC[tempif].p[j]].trainIdx].pt.y + HALFB_HEIGHT) / CELLB_HEIGHT;

					if (row * CELLB_NUM_COL + col == correspondCell + CELLB_NUM_COL + 1)
						i9++;
				}
			}

			int sum = i1 + i2 + i3 + i4 + i5 + i6 + i7 + i8 + i9;
			float tau = alpha * sqrt((float)(i1num + i2num + i3num + i4num + i5num + i6num + i7num + i8num + i9num) / ((float)num));
			if (sum > tau)
			{
				for (auto it = cellC[i].p.begin(); it != cellC[i].p.end(); )
				{
					int col = int(sift_keypoints_2[sift_matches_origional[*it].trainIdx].pt.x + HALFB_WIDTH) / CELLB_WIDTH;
					int row = int(sift_keypoints_2[sift_matches_origional[*it].trainIdx].pt.y + HALFB_HEIGHT) / CELLB_HEIGHT;
					if (row * CELLB_NUM_COL + col == correspondCell && IsAdd[*it] == false)
					{
						IsRobust[*it] = true;
					}
					it++;
					if (it == cellC[i].p.end())
						break;
				}
			}
			else
			{
				vector <int>().swap(cellC[i].p);
			}
		}
	}
}

