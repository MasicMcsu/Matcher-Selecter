#ifndef SEGMENT_IMAGE  
#define SEGMENT_IMAGE  
typedef unsigned char uchar;
#include "opencv2/opencv.hpp"  
#include "misc.h"  
#include "segmentgraph.h"

//typedef struct {  
//  uchar r, g, b;   
//} rgb;  

rgb random_rgb();

//static inline float diff(std::vector<Mat> &rgbs int x1, int y1, int x2, int y2)  
//{  
//  return norm(rgbs)  
//}  
cv::Mat segment_image(cv::Mat &img, float sigma, float c, int min_size, int *num_ccs);
void segment_image1(cv::Mat& img, float sigma, float c, int min_size, int* num_ccs, universe* u, int* num);
void SegImage(cv::Mat& image, universe* u, int* num, float sigma, float k, int min_size);
#endif  