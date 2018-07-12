// 对比度增强图像去雾算法
// 第一步：输入带雾图像
// 第二步：计算并获得大气光值与图像预处理
// 第三步：新型透射率的计算方法
// 第四步：输出处理图像


/*
总结：
算法对于天空部分的处理相当到位，更优于Kaiming He的暗通道先验算法；
对比度过大时，图像很容易发暗，可以后期将图片稍微调亮一些。
算法本身是从对比增强的角度来进行去雾操作的，所以你可以看出结果自动带对比度增强加成，所以这个算法所取得的结果通常更加鲜亮。
*/

#include "stdafx.h"
#include <Windows.h>                
#include<iostream>
#include<time.h>

// opencv
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

// 全局变量
unsigned char* pBmpBuf;
int bmpWidth;
int bmpHeight;
RGBQUAD* pColorTable;
int biBitCount;

//导向滤波  
Mat guildFilter(cv::Mat g, cv::Mat p, int ksize)
{
	const double eps = 1.0e-5;//regularization parameter  
	//类型转换  
	Mat _g;
	g.convertTo(_g, CV_64FC1);
	g = _g;

	Mat _p;
	p.convertTo(_p, CV_64FC1);
	p = _p;

	//[hei, wid] = size(I);  
	int hei = g.rows;
	int wid = g.cols;

	//N = boxfilter(ones(hei, wid), r); % the size of each local patch; N=(2r+1)^2 except for boundary pixels.  
	cv::Mat N;
	cv::boxFilter(cv::Mat::ones(hei, wid, g.type()), N, CV_64FC1, Size(ksize, ksize));

	//[1] --使用均值模糊求取各项系数  
	Mat mean_G;
	boxFilter(g, mean_G, CV_64FC1, Size(ksize, ksize));

	Mat mean_P;
	boxFilter(p, mean_P, CV_64FC1, Size(ksize, ksize));

	Mat GP = g.mul(p);
	Mat mean_GP;
	boxFilter(GP, mean_GP, CV_64FC1, Size(ksize, ksize));

	Mat GG = g.mul(g);
	Mat mean_GG;
	boxFilter(GG, mean_GG, CV_64FC1, Size(ksize, ksize));

	Mat cov_GP;
	cov_GP = mean_GP - mean_G.mul(mean_P);

	Mat var_G;
	var_G = mean_GG - mean_G.mul(mean_G);
	//[1]  

	//求系数a a=(mean(GP)-mean(G)mean(p))/(mean(GG)-mean(G)mean(G)+eps)  
	Mat a = cov_GP / (var_G + eps);

	//求系数b b=mean(P)-mean(G)*a  
	Mat b = mean_P - a.mul(mean_G);

	//求两个系数的均值  
	Mat mean_a;
	boxFilter(a, mean_a, CV_64FC1, Size(ksize, ksize));
	//mean_a=mean_a/N;  

	Mat mean_b;
	boxFilter(b, mean_b, CV_64FC1, Size(ksize, ksize));
	//mean_b=mean_b/N;  

	//输出结果q  
	Mat q = mean_a.mul(g) + mean_b;
	return q;
}

unsigned char* readBmp(char* bmpName)

{
	FILE* fp = fopen(bmpName, "rb"); //以二进制读的方式打开指定的图像文件
	if (fp == 0) return 0;

	fseek(fp, sizeof(BITMAPFILEHEADER), 0);
	BITMAPINFOHEADER infoHead;
	fread(&infoHead, sizeof(BITMAPINFOHEADER), 1, fp);

	bmpWidth = infoHead.biWidth;
	bmpHeight = infoHead.biHeight;
	biBitCount = infoHead.biBitCount;

	//strick
	int lineByte = (bmpWidth * biBitCount / 8 + 3) / 4 * 4;


	if (biBitCount == 8)
	{
		pColorTable = new RGBQUAD[256];
		fread(pColorTable, sizeof(RGBQUAD), 256, fp);

	}
	pBmpBuf = new unsigned char[lineByte * bmpHeight];

	fread(pBmpBuf, 1, lineByte * bmpHeight, fp);
	fclose(fp);
	return pBmpBuf;
}


BYTE *RmwRead8BitBmpFile2Img(const char * filename, int *width, int *height)
{
	FILE *binFile;
	BYTE *pImg = NULL;
	BITMAPFILEHEADER fileHeader;
	BITMAPINFOHEADER bmpHeader;
	BOOL isRead = TRUE;
	int linenum, ex;

	if ((binFile = fopen(filename, "rb")) == NULL) return NULL;

	if (fread((void *)&fileHeader, 1, sizeof(fileHeader), binFile) != sizeof(fileHeader)) isRead = FALSE;
	if (fread((void *)&bmpHeader, 1, sizeof(bmpHeader), binFile) != sizeof(bmpHeader)) isRead = FALSE;

	if (isRead == FALSE || fileHeader.bfOffBits<sizeof(fileHeader) + sizeof(bmpHeader)) {
		fclose(binFile);
		return NULL;
	}

	*width = bmpHeader.biWidth;
	*height = bmpHeader.biHeight;
	linenum = (*width * 1 + 3) / 4 * 4;
	ex = linenum - *width * 1;

	fseek(binFile, fileHeader.bfOffBits, SEEK_SET);
	pImg = new BYTE[(*width)*(*height)];
	if (pImg != NULL) {
		for (int i = 0; i<*height; i++) {
			int r = fread(pImg + (*height - i - 1)*(*width), sizeof(BYTE), *width, binFile);
			if (r != *width) {
				delete pImg;
				fclose(binFile);
				return NULL;
			}
			fseek(binFile, ex, SEEK_CUR);
		}
	}
	fclose(binFile);
	return pImg;
}

void delay_sec(int sec)//  
{
	time_t start_time, cur_time;
	time(&start_time);
	do
	{
		time(&cur_time);
	} while ((cur_time - start_time) < sec);
}
//***************************************************************************************************************************
// 名称：hist_equalization_dlphay()
// 功能：灰度图像的直方图均衡(测试版本)
// 参数： 形参：Mat input_image    输入图像
//      返回值：Mat output_image   输出图像      
//****************************************************************************************************************************
Mat hist_equalization_GRAY_dlphay_test(Mat input_image)
{
	const int grayMax = 255;
	vector<vector<int>> graylevel(grayMax + 1);

	cout << graylevel.size() << endl;
	Mat output_image;
	input_image.copyTo(output_image);

	if (!input_image.data)
	{
		return output_image;
	}
	for (int i = 0; i < input_image.rows - 1; i++)
	{
		uchar* ptr = input_image.ptr<uchar>(i);  // 处理成为一行一行的数据  储存在ptr
		for (int j = 0; j < input_image.cols - 1; j++)
		{
			int x = ptr[j];
			graylevel[x].push_back(0);//这个地方写的不好，引入二维数组只是为了记录每一个灰度值的像素个数  
		}
	}
	for (int i = 0; i < output_image.rows - 1; i++)
	{
		uchar* imgptr = output_image.ptr<uchar>(i);
		uchar* imageptr = input_image.ptr<uchar>(i);
		for (int j = 0; j < output_image.cols - 1; j++)
		{
			int sumpiexl = 0;
			for (int k = 0; k < imageptr[j]; k++)
			{
				sumpiexl = graylevel[k].size() + sumpiexl;
			}
			imgptr[j] = (grayMax*sumpiexl / (input_image.rows*input_image.cols));
		}
	}
	return output_image;
}
//***************************************************************************************************************************
// 名称：hist_equalization_dlphay()
// 功能：灰度图像的直方图均衡(测试版本)
// 参数： 形参：Mat input_image    输入图像
//      返回值：Mat output_image   输出图像      
//****************************************************************************************************************************
Mat hist_equalization_GRAY_dlphay(Mat& input)
{
	int gray_sum = 0;  //像素总数
	int gray[256] = { 0 };  //记录每个灰度级别下的像素个数
	double gray_prob[256] = { 0 };  //记录灰度分布密度
	double gray_distribution[256] = { 0 };  //记录累计密度
	int gray_equal[256] = { 0 };  //均衡化后的灰度值

	Mat output = input.clone();
	gray_sum = input.cols * input.rows;

	//统计每个灰度下的像素个数
	for (int i = 0; i < input.rows; i++)
	{
		uchar* p = input.ptr<uchar>(i);
		for (int j = 0; j < input.cols; j++)
		{
			int vaule = p[j];
			gray[vaule]++;
		}
	}
	//统计灰度频率
	for (int i = 0; i < 256; i++)
	{
		gray_prob[i] = ((double)gray[i] / gray_sum);
	}

	//计算累计密度
	gray_distribution[0] = gray_prob[0];
	for (int i = 1; i < 256; i++)
	{
		gray_distribution[i] = gray_distribution[i - 1] + gray_prob[i];
	}

	//重新计算均衡化后的灰度值，四舍五入。参考公式：(N-1)*T+0.5
	for (int i = 0; i < 256; i++)
	{
		gray_equal[i] = (uchar)(255 * gray_distribution[i] + 0.5);
	}
	//直方图均衡化,更新原图每个点的像素值
	for (int i = 0; i < output.rows; i++)
	{
		uchar* p = output.ptr<uchar>(i);
		for (int j = 0; j < output.cols; j++)
		{
			p[j] = gray_equal[p[j]];
		}
	}
	return output;
}
Mat hist_equalization_BGR_dlphay(Mat input)
{
	Mat output;
	uchar *dataIn = (uchar *)input.ptr<uchar>(0);//input的头指针，指向第0行第0个像素，且为行优先
	uchar *dataOut = (uchar *)output.ptr<uchar>(0);
	const int ncols = input.cols;//表示输入图像有多少行
	const int nrows = input.rows;//表示输入图像有多少列
	int nchannel = input.channels();//通道数，一般是3
	int pixnum = ncols*nrows;
	int vData[765] = { 0 };//由于R+G+B最时是255+255+255，所以为765个亮度级
	double vRate[765] = { 0 };
	for (int i = 0; i < nrows; i++)
	{
		for (int j = 0; j < ncols; j++)
		{
			vData[dataIn[i*ncols*nchannel + j*nchannel + 0]
				+ dataIn[i*ncols*nchannel + j*nchannel + 1]
				+ dataIn[i*ncols*nchannel + j*nchannel + 2]]++;//对应的亮度级统计
		}
	}
	for (int i = 0; i < 764; i++)
	{
		for (int j = 0; j < i; j++)
		{
			vRate[i] += (double)vData[j] / (double)pixnum;//求出
		}
	}
	for (int i = 0; i < 764; i++)
	{
		vData[i] = (int)(vRate[i] * 764 + 0.5);//进行归一化处理
	}
	for (int i = 0; i < nrows; i++)
	{
		for (int j = 0; j < ncols; j++)
		{
			int amplification = vData[dataIn[i*ncols*nchannel + j*nchannel + 0]
				+ dataIn[i*ncols*nchannel + j*nchannel + 1]
				+ dataIn[i*ncols*nchannel + j*nchannel + 2]] -
				(dataIn[i*ncols*nchannel + j*nchannel + 0]
				+ dataIn[i*ncols*nchannel + j*nchannel + 1]
				+ dataIn[i*ncols*nchannel + j*nchannel + 2]);//用变换后的值减去原值的到亮度级的差值，除3后就是每个通道应当变化的值
			int b = dataIn[i*ncols*nchannel + j*nchannel + 0] + amplification / 3 + 0.5;
			int g = dataIn[i*ncols*nchannel + j*nchannel + 1] + amplification / 3 + 0.5;
			int r = dataIn[i*ncols*nchannel + j*nchannel + 2] + amplification / 3 + 0.5;
			if (b > 255) b = 255;//上溢越位判断
			if (g > 255) g = 255;
			if (r > 255) r = 255;
			if (r < 0) r = 0;//下溢越位判断
			if (g < 0) g = 0;
			if (b < 0) b = 0;
			dataOut[i*ncols*nchannel + j*nchannel + 0] = b;
			dataOut[i*ncols*nchannel + j*nchannel + 1] = g;
			dataOut[i*ncols*nchannel + j*nchannel + 2] = r;
		}
	}
	return output;
}


/***********************************************************************************/
/*****************************    MAIN函数    **************************************/
/***********************************************************************************/
int main()
{
	//图像输入部分
	Mat input_1 = imread("test.jpg", CV_LOAD_IMAGE_UNCHANGED);
	Mat input_show_1 = imread("test.jpg", CV_LOAD_IMAGE_UNCHANGED);

	/*
	
	int flags;
    //! the matrix dimensionality, >= 2
    int dims;
    //! the number of rows and columns or (-1, -1) when the matrix has more than 2 dimensions
    int rows, cols;
    //! pointer to the data
    uchar* data;

    //! pointer to the reference counter;
    // when matrix points to user-allocated data, the pointer is NULL
    int* refcount;

    //! helper fields used in locateROI and adjustROI
    uchar* datastart;
    uchar* dataend;
    uchar* datalimit;

    //! custom allocator
    MatAllocator* allocator;
	
	
	
	*/



	Mat output_1 = imread("test.jpg", CV_LOAD_IMAGE_UNCHANGED);
	// get Info
	int image_width_1 = 0;
	int image_height_1 = 0;
	//图像宽度
	image_width_1 = input_1.cols;
	//图像高度
	image_height_1 = input_1.rows;
	//图像flag与dims
	auto image_flags_1 = input_1.flags;
	auto image_dims_1 = input_1.dims;

	//图像输入部分
	Mat input_2 = imread("input2.jpg", CV_LOAD_IMAGE_UNCHANGED);
	Mat input_show_2 = imread("input2.jpg", CV_LOAD_IMAGE_UNCHANGED);

	Mat output_2 = imread("input2.jpg", CV_LOAD_IMAGE_UNCHANGED);
	int image_width_2 = 0;
	int image_height_2 = 0;
	//图像宽度
	image_width_2 = input_2.cols;
	//图像高度
	image_height_2 = input_2.rows;
	//图像flag与dims
	auto image_flags_2 = input_2.flags;
	auto image_dims_2 = input_2.dims;

	//
	Mat input_3 = imread("test.jpg", CV_LOAD_IMAGE_UNCHANGED);
	Mat output_3 = imread("test.jpg", CV_LOAD_IMAGE_UNCHANGED);

	int image_width_3 = 0;
	int image_height_3 = 0;
	image_width_3 = input_3.cols;
	image_height_3 = input_3.rows;
	auto image_flags_3 = input_3.flags;
	auto image_dims_3 = input_3.dims;

	int image_depth = 3; // 默认是3通道的

	// 图像缓存
	unsigned char* src_1 = (unsigned char *)malloc(sizeof(unsigned char) * (image_width_1 * image_height_1 * image_depth));
	unsigned char* src_2 = (unsigned char *)malloc(sizeof(unsigned char) * (image_width_2 * image_height_2 * image_depth));

	unsigned char* src_3 = (unsigned char *)malloc(sizeof(unsigned char) * (image_width_1 * image_height_1 * image_depth));
	unsigned char* src_4 = (unsigned char *)malloc(sizeof(unsigned char) * (image_width_2 * image_height_2 * image_depth));

	unsigned char* src_5 = (unsigned char *)malloc(sizeof(unsigned char) * (image_width_3 * image_height_3 * image_depth));
	unsigned char* src_6 = (unsigned char *)malloc(sizeof(unsigned char) * (image_width_3 * image_height_3 * image_depth));

	unsigned char* MedianBlurHaze = (unsigned char *)malloc(sizeof(unsigned char) * (image_width_1 * image_height_1 * image_depth));

	src_1 = input_1.data;
	MedianBlurHaze = input_1.data;
	Mat src1 = input_1;
	Mat dst1 = input_1;

	// 参数值的设置
	src_2 = input_2.data;
	MedianBlurHaze = input_2.data;
	Mat src2 = input_2;
	Mat dst2 = input_2;

	// 去雾相关变量
	unsigned char *Src_1 = &src_1[0];
	unsigned char *Dest_1 = &src_3[0];

	int Width_1 = image_width_1;
	int Height_1 = image_height_1;

	int Stride_1 = 0;
	if ((Width_1 * Height_1) % 4 == 0)
	{
		Stride_1 = Width_1 * image_depth;
	}
	else
	{
		Stride_1 = Width_1 * image_depth + (4 - Width_1 * image_depth % 4);
	}

	// 去雾相关变量
	unsigned char *Src_2 = &src_2[0];
	unsigned char *Dest_2 = &src_4[0];
	int Width_2 = image_width_2;
	int Height_2 = image_height_2;
	int Stride_2 = 0;
	if ((Width_2 * image_depth) % 4 == 0)
	{
		Stride_2 = Width_2 * image_depth;
	}
	else
	{
		Stride_2 = Width_2 * image_depth + (4 - Width_2 * image_depth % 4);
	}
	// 参数值的设置
	int BlockSize = 4;
	int GuideRadius = 5;
	int MaxAtom = 220;
	float Omega = 0.9f;

	unsigned char *Src3 = &src_5[0];
	unsigned char *Dest3 = &src_6[0];
	int Width_3 = image_width_3;
	int Height_3 = image_height_3;
	int Stride_3 = 0;

	// Stride的计算
	if ((Width_3 * image_depth) % 4 == 0)
	{
		Stride_3 = Width_3 * image_depth;
	}
	else
	{
		Stride_3 = Width_3 * image_depth + (4 - Width_3 * image_depth % 4);
	}
	//先进行直方图均衡处理
	{
		Mat input = input_3;
		int gray_sum = 0;  //像素总数
		int gray[256] = { 0 };  //记录每个灰度级别下的像素个数
		double gray_prob[256] = { 0 };  //记录灰度分布密度
		double gray_distribution[256] = { 0 };  //记录累计密度
		int gray_equal[256] = { 0 };  //均衡化后的灰度值

		Mat output = input.clone();
		gray_sum = input.cols * input.rows;

		//统计每个灰度下的像素个数
		for (int i = 0; i < input.rows; i++)
		{
			uchar* p = input.ptr<uchar>(i);
			for (int j = 0; j < input.cols; j++)
			{
				int vaule = p[j];
				gray[vaule]++;
			}
		}
		//统计灰度频率
		for (int i = 0; i < 256; i++)
		{
			gray_prob[i] = ((double)gray[i] / gray_sum);
		}

		//计算累计密度
		gray_distribution[0] = gray_prob[0];
		for (int i = 1; i < 256; i++)
		{
			gray_distribution[i] = gray_distribution[i - 1] + gray_prob[i];
		}

		//重新计算均衡化后的灰度值，四舍五入。参考公式：(N-1)*T+0.5
		for (int i = 0; i < 256; i++)
		{
			gray_equal[i] = (uchar)(255 * gray_distribution[i] + 0.5);
		}
		//直方图均衡化,更新原图每个点的像素值
		for (int i = 0; i < output.rows; i++)
		{
			uchar* p = output.ptr<uchar>(i);
			for (int j = 0; j < output.cols; j++)
			{
				p[j] = gray_equal[p[j]];
			}
		}
	}
	typedef int(__stdcall *Dehaze)(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride,
		int BlockSize, int GuideRadius, int MaxAtom, float Omega, float T0, float Gamma);
	Dehaze pfFuncInDll = NULL;
	HINSTANCE hinst = LoadLibraryA("ImageMaster.dll");

	// 大气光计算函数代码
	{
		IplImage* imInput = &IplImage(input_3);
		int nWid = imInput->width;
		int nHei = imInput->height;
		int nMinDistance = 65536;
		int nDistance;
		int nX, nY;
		int nMaxIndex;
		double dpScore[3];
		double dpMean[3];
		double dpStds[3];
		float afMean[4] = { 0 };
		float afScore[4] = { 0 };
		float nMaxScore = 0;
		int nStep = imInput->widthStep;

		// 4 sub-block  
		IplImage *iplUpperLeft = cvCreateImage(cvSize(nWid / 2, nHei / 2), IPL_DEPTH_8U, 3);
		IplImage *iplUpperRight = cvCreateImage(cvSize(nWid / 2, nHei / 2), IPL_DEPTH_8U, 3);

		IplImage *iplLowerLeft = cvCreateImage(cvSize(nWid / 2, nHei / 2), IPL_DEPTH_8U, 3);
		IplImage *iplLowerRight = cvCreateImage(cvSize(nWid / 2, nHei / 2), IPL_DEPTH_8U, 3);

		IplImage *iplR = cvCreateImage(cvSize(nWid / 2, nHei / 2), IPL_DEPTH_8U, 1);
		IplImage *iplG = cvCreateImage(cvSize(nWid / 2, nHei / 2), IPL_DEPTH_8U, 1);

		IplImage *iplB = cvCreateImage(cvSize(nWid / 2, nHei / 2), IPL_DEPTH_8U, 1);

		// divide   
		cvSetImageROI(imInput, cvRect(0, 0, nWid / 2, nHei / 2));
		cvCopyImage(imInput, iplUpperLeft);

		cvSetImageROI(imInput, cvRect(nWid / 2 + nWid % 2, 0, nWid, nHei / 2));
		cvCopyImage(imInput, iplUpperRight);

		cvSetImageROI(imInput, cvRect(0, nHei / 2 + nHei % 2, nWid / 2, nHei));
		cvCopyImage(imInput, iplLowerLeft);

		cvSetImageROI(imInput, cvRect(nWid / 2 + nWid % 2, nHei / 2 + nHei % 2, nWid, nHei));
		cvCopyImage(imInput, iplLowerRight);
		if (nHei*nWid > 200)
		{

			// compute the mean and std-dev in the sub-block  
			// upper left sub-block  
			cvCvtPixToPlane(iplUpperLeft, iplR, iplG, iplB, 0);

			cvMean_StdDev(iplR, dpMean, dpStds);
			cvMean_StdDev(iplG, dpMean + 1, dpStds + 1);
			cvMean_StdDev(iplB, dpMean + 2, dpStds + 2);
			// dpScore: mean - std-dev  
			dpScore[0] = dpMean[0] - dpStds[0];
			dpScore[1] = dpMean[1] - dpStds[1];
			dpScore[2] = dpMean[2] - dpStds[2];

			afScore[0] = (float)(dpScore[0] + dpScore[1] + dpScore[2]);

			nMaxScore = afScore[0];
			nMaxIndex = 0;

			// upper right sub-block  
			cvCvtPixToPlane(iplUpperRight, iplR, iplG, iplB, 0);

			cvMean_StdDev(iplR, dpMean, dpStds);
			cvMean_StdDev(iplG, dpMean + 1, dpStds + 1);
			cvMean_StdDev(iplB, dpMean + 2, dpStds + 2);

			dpScore[0] = dpMean[0] - dpStds[0];
			dpScore[1] = dpMean[1] - dpStds[1];
			dpScore[2] = dpMean[2] - dpStds[2];

			afScore[1] = (float)(dpScore[0] + dpScore[1] + dpScore[2]);

			if (afScore[1] > nMaxScore)
			{
				nMaxScore = afScore[1];
				nMaxIndex = 1;
			}

			// lower left sub-block  
			cvCvtPixToPlane(iplLowerLeft, iplR, iplG, iplB, 0);

			cvMean_StdDev(iplR, dpMean, dpStds);
			cvMean_StdDev(iplG, dpMean + 1, dpStds + 1);
			cvMean_StdDev(iplB, dpMean + 2, dpStds + 2);

			dpScore[0] = dpMean[0] - dpStds[0];
			dpScore[1] = dpMean[1] - dpStds[1];
			dpScore[2] = dpMean[2] - dpStds[2];

			afScore[2] = (float)(dpScore[0] + dpScore[1] + dpScore[2]);

			if (afScore[2] > nMaxScore)
			{
				nMaxScore = afScore[2];
				nMaxIndex = 2;
			}

			// lower right sub-block  
			cvCvtPixToPlane(iplLowerRight, iplR, iplG, iplB, 0);

			cvMean_StdDev(iplR, dpMean, dpStds);
			cvMean_StdDev(iplG, dpMean + 1, dpStds + 1);
			cvMean_StdDev(iplB, dpMean + 2, dpStds + 2);

			dpScore[0] = dpMean[0] - dpStds[0];
			dpScore[1] = dpMean[1] - dpStds[1];
			dpScore[2] = dpMean[2] - dpStds[2];

			afScore[3] = (float)(dpScore[0] + dpScore[1] + dpScore[2]);

			if (afScore[3] > nMaxScore)
			{
				nMaxScore = afScore[3];
				nMaxIndex = 3;
			}
		}
		else
		{
			// select the atmospheric light value in the sub-block  
			for (nY = 0; nY<nHei; nY++)
			{
				for (nX = 0; nX<nWid; nX++)
				{
					// 255-r, 255-g, 255-b  
					nDistance = int(sqrt(float(255 - (uchar)imInput->imageData[nY*nStep + nX * 3])*float(255 - (uchar)imInput->imageData[nY*nStep + nX * 3])
						+ float(255 - (uchar)imInput->imageData[nY*nStep + nX * 3 + 1])*float(255 - (uchar)imInput->imageData[nY*nStep + nX * 3 + 1])
						+ float(255 - (uchar)imInput->imageData[nY*nStep + nX * 3 + 2])*float(255 - (uchar)imInput->imageData[nY*nStep + nX * 3 + 2])));
					if (nMinDistance > nDistance)
					{
						nMinDistance = nDistance;
					}
				}
			}
		}
	}
	if (hinst != NULL)
	{
		pfFuncInDll = (Dehaze)GetProcAddress(hinst, "IM_HazeRemovalBasedOnDarkChannelPrior");
		if (pfFuncInDll != NULL)
		{
			// 透射率的计算
			{
				int *pnImageR;
				int *pnImageG;
				int *pnImageB;
				int nStartX = 0;
				int nStartY = 0;
				int nWid = 10;

				int nHei = 10;
				int nCounter;
				int nX, nY;
				int nEndX;
				int nEndY;

				int nOutR = 0, nOutG = 0, nOutB = 0;
				int nSquaredOut;
				int nSumofOuts;
				int nSumofSquaredOuts;

				float fTrans, fOptTrs;
				int nTrans;
				int nSumofSLoss;
				float fCost, fMinCost, fMean;
				int nNumberofPixels, nLossCount;

				nEndX = (nStartX + 10, nWid);
				nEndY = (nStartY + 10, nHei);

				nNumberofPixels = (nEndY - nStartY)*(nEndX - nStartX) * 3;

				fTrans = 0.3f;
				nTrans = 427;

				for (nCounter = 0; nCounter<7; nCounter++)
				{
					nSumofSLoss = 0;
					nLossCount = 0;
					nSumofSquaredOuts = 0;
					nSumofOuts = 0;

					for (nY = nStartY; nY<nEndY; nY++)
					{
						for (nX = nStartX; nX<nEndX; nX++)
						{

							if (nOutR>255)
							{
								nSumofSLoss += (nOutR - 255)*(nOutR - 255);
								nLossCount++;
							}
							else if (nOutR < 0)
							{
								nSumofSLoss += nOutR * nOutR;
								nLossCount++;
							}
							if (nOutG>255)
							{
								nSumofSLoss += (nOutG - 255)*(nOutG - 255);
								nLossCount++;
							}
							else if (nOutG < 0)
							{
								nSumofSLoss += nOutG * nOutG;
								nLossCount++;
							}
							if (nOutB>255)
							{
								nSumofSLoss += (nOutB - 255)*(nOutB - 255);
								nLossCount++;
							}
							else if (nOutB < 0)
							{
								nSumofSLoss += nOutB * nOutB;
								nLossCount++;
							}
							nSumofSquaredOuts += nOutB * nOutB + nOutR * nOutR + nOutG * nOutG;;
							nSumofOuts += nOutR + nOutG + nOutB;
						}
					}
					fMean = (float)(nSumofOuts) / (float)(nNumberofPixels);
					fCost = 10 * (float)nSumofSLoss / (float)(nNumberofPixels)
						-((float)nSumofSquaredOuts / (float)nNumberofPixels - fMean*fMean);

					if (nCounter == 0 || fMinCost > fCost)
					{
						fMinCost = fCost;
						fOptTrs = fTrans;
					}

					fTrans += 0.1f;
					nTrans = (int)(1.0f / fTrans*128.0f);
				}

				Mat g = input_3;
				Mat p = output_3;
				int ksize = 0;
				const double eps = 1.0e-5;//regularization parameter  
				//类型转换  
				Mat _g = input_3;
				g.convertTo(_g, CV_64FC1);
				g = _g;
				Mat _p;
				p.convertTo(_p, CV_64FC1);
				p = _p;
				//[hei, wid] = size(I);  
				int hei = g.rows;
				int wid = g.cols;
				float T0 = hei*wid / 10000;
			}
			float T0 = 0.1f;
			float Gamma = 0.9f;
			int P = pfFuncInDll(Src_1, Src_1, Width_1, Height_1, Stride_1, BlockSize, GuideRadius, MaxAtom, Omega, T0, Gamma);
			output_1.data = Src_1;
		}
		if (pfFuncInDll != NULL)
		{
			// 透射率的计算
			{
				int *pnImageR;
				int *pnImageG;
				int *pnImageB;
				int nStartX = 0;
				int nStartY = 0;


				int nWid = 10;
				int nHei = 10;
				int nCounter;
				int nX, nY;
				int nEndX;
				int nEndY;

				int nOutR = 0, nOutG = 0, nOutB = 0;
				int nSquaredOut;
				int nSumofOuts;
				int nSumofSquaredOuts;
				float fTrans, fOptTrs;

				int nTrans;
				int nSumofSLoss;
				float fCost, fMinCost, fMean;
				int nNumberofPixels, nLossCount;

				nEndX = (nStartX + 10, nWid);
				nEndY = (nStartY + 10, nHei);

				nNumberofPixels = (nEndY - nStartY)*(nEndX - nStartX) * 3;

				fTrans = 0.3f;
				nTrans = 427;

				for (nCounter = 0; nCounter<7; nCounter++)
				{
					nSumofSLoss = 0;
					nLossCount = 0;

					nSumofSquaredOuts = 0;
					nSumofOuts = 0;

					for (nY = nStartY; nY<nEndY; nY++)
					{
						for (nX = nStartX; nX<nEndX; nX++)
						{

							if (nOutR>255)
							{
								nSumofSLoss += (nOutR - 255)*(nOutR - 255);
								nLossCount++;
							}
							else if (nOutR < 0)
							{
								nSumofSLoss += nOutR * nOutR;
								nLossCount++;
							}
							if (nOutG>255)
							{
								nSumofSLoss += (nOutG - 255)*(nOutG - 255);
								nLossCount++;
							}
							else if (nOutG < 0)
							{
								nSumofSLoss += nOutG * nOutG;
								nLossCount++;
							}
							if (nOutB>255)
							{
								nSumofSLoss += (nOutB - 255)*(nOutB - 255);
								nLossCount++;
							}
							else if (nOutB < 0)
							{
								nSumofSLoss += nOutB * nOutB;
								nLossCount++;
							}
							nSumofSquaredOuts += nOutB * nOutB + nOutR * nOutR + nOutG * nOutG;;
							nSumofOuts += nOutR + nOutG + nOutB;
						}
					}
					fMean = (float)(nSumofOuts) / (float)(nNumberofPixels);
					fCost = 10 * (float)nSumofSLoss / (float)(nNumberofPixels)
						-((float)nSumofSquaredOuts / (float)nNumberofPixels - fMean*fMean);

					if (nCounter == 0 || fMinCost > fCost)
					{
						fMinCost = fCost;
						fOptTrs = fTrans;
					}

					fTrans += 0.1f;
					nTrans = (int)(1.0f / fTrans*128.0f);
				}

				Mat g = input_3;
				Mat p = output_3;
				int ksize = 0;
				const double eps = 1.0e-5;//regularization parameter  
				//类型转换  
				Mat _g = input_3;
				g.convertTo(_g, CV_64FC1);
				g = _g;
				Mat _p;

				p.convertTo(_p, CV_64FC1);
				p = _p;
				//[hei, wid] = size(I);  
				int hei = g.rows;
				int wid = g.cols;
				float T0 = hei*wid / 10000;
			}
			float T0 = 0.1f;
			float Gamma = 0.9f;
			int P = pfFuncInDll(Src_2, Src_2, Width_2, Height_2, Stride_2, BlockSize, GuideRadius, MaxAtom, Omega, T0, Gamma);
			output_2.data = Src_2;
		}
		FreeLibrary(hinst);
	}
	//图像显示部分

	// 图像1的显示
	imshow("原始图像1", input_show_1);
	imshow("处理图像1", output_1);

	// 图像2的显示
	imshow("原始图像2", input_show_2);
	imshow("处理图像2", output_2);

	//waiting...
	waitKey(0);

	//暂停
	system("pause");

	return 0;
}



