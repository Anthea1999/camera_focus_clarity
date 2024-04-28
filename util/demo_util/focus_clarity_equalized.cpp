#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <cstdio>
#include <string>
#include <thread>
#include <chrono>

#include <robot_eye.hpp>

using namespace cv;
using namespace std;

// 宣告函式
void my_mouse_callback(int event, int x, int y, int flags, void* param);
double Thenengrad(const cv::Mat&);

// 宣告全域變數
Mat image;
Rect box;
bool drawing_box = false;

string intToString(int number){

	std::stringstream ss;
	ss << number;
	return ss.str();
}

// 滑鼠回覆函數
void my_mouse_callback(int event, int x, int y,
	int flags, void* parm)
{
	switch( event )
	{
		// 移動滑鼠
		case EVENT_MOUSEMOVE:
		{
			if( drawing_box )
			{
				box.width  = x-box.x;
				box.height = y-box.y;

				Mat temp = image.clone();

				// 畫出選取內容
				rectangle(temp, Point(box.x, box.y),
					Point(box.x + box.width, box.y + box.height),
					Scalar(0, 0xff, 0), 3);
				imshow("Box Example", temp);
			}
		}
		break;

		// 按下滑鼠左鍵
		case EVENT_LBUTTONDOWN:
		{
			drawing_box = true;
			box = Rect( x, y, 0, 0 );
		}
		break;

		// 放掉滑鼠左鍵
		case EVENT_LBUTTONUP:
		{
			drawing_box = false;
			if( box.width<0  )
			{
				box.x+=box.width;
				box.width *=-1;
			}

			if( box.height<0 )
			{
				box.y+=box.height;
				box.height*=-1;
			}

			// 擷取圖像
			Mat temp(image, box);
			imshow("Crop", temp);
            resize(temp, temp, Size(600, 400), INTER_LINEAR);

			imwrite("./images/temp.jpg", temp);

			// 畫出選取內容
			rectangle(image, Point(box.x, box.y),
				Point(box.x+box.width, box.y+box.height),
				Scalar(0, 0xff, 0), 3);
			imshow("Box Example", image);
		}
		break;
	}
}

double Thenengrad(const cv::Mat& img)
{
    double Grad_value = 0;
    double Sx, Sy;
    for (int i = 1; i < img.rows-1; i++)
    {
        //定义行指针
        uchar *current_ptr = (uchar*)img.data + i * img.cols;//当前行
        uchar *pre_ptr = (uchar*)img.data + (i - 1)*img.cols;//上一行
        uchar *next_ptr= (uchar*)img.data + (i +1)*img.cols;//下一行
        for (int j = 1; j < img.cols-1; j++)
        {
            Sx = pre_ptr[j - 1] * (-1) + pre_ptr[j + 1] + current_ptr[j - 1] * (-2) + current_ptr[j + 1] * 2 + next_ptr[j - 1] * (-1) + next_ptr[j + 1];//x方向梯度
            Sy = pre_ptr[j - 1] + 2 * pre_ptr[j] + pre_ptr[j + 1] - next_ptr[j - 1] - 2 * next_ptr[j] - next_ptr[j + 1];//y方向梯度
            //求总和
            Grad_value += Sx * Sx + Sy * Sy;
        }
    }
    return Grad_value / (img.cols - 2) / (img.rows - 2);
}

int main(int argc, char* argv[])
{
    Mat img;
    Mat temp;
    Mat result;
    Mat mergedst;
	vector<Mat> channels;
    char kbin = 0;
    bool ret;
    double score;
    double original_score;
    double sharpened_score;
    double equalized_score;

    // 儲存圖檔的檔名
	char filename [50];
	// 檔名序號起始値
	int n = 0;


    RobotEye eye(0);
    ret = eye.set_focus_auto(false);
    if (!ret)
    {
        cout << "Failed to disable device auto focus" << endl;
        return -1;
    }

    int min, max, step;
    ret = eye.get_focus_info(min, max, step);
    if (!ret)
    {
        cout << "Failed to get focus information" << endl;
        return -1;
    }

    int focus = -1;
    ret = eye.get_focus(focus);
    if (!ret)
    {
        cout << "Failed to get focus value" << endl;
        return -1;
    }
    
    do {
        Mat src;
        eye.get_frame(src);
        if (!src.empty()){
            imshow("camera", src);
            kbin = waitKey(10);
            score = Thenengrad(src);
            printf("score:%f\n", score);
            if (score < 40000){
                focus -= step;
            }
            switch (toupper(kbin)){
                case '+':
                    focus += step;
                    break;

                case '-':
                    focus -= step;
                    break;

                case '*':
                sprintf(filename, "./images/VideoFrame%d.jpg", n++);
				imwrite(filename, src);

                image = imread(filename, 1);
                namedWindow( "Box Example");
                imshow("Box Example", image);
                setMouseCallback("Box Example", my_mouse_callback);
                waitKey(0);
				break;
            }
            if(focus < min){
                focus = min;
            }

            if(focus > max){
                focus = max;
            }
            cout << "Focus:" << focus <<endl;
            ret = eye.set_focus(focus);
            //this_thread::sleep_for(chrono::seconds(30));
            }
    }while (kbin != 27);



	img = imread("./images/template.jpg");
	temp = imread("./images/temp.jpg");

	namedWindow("Original Image", cv::WINDOW_AUTOSIZE );
    imshow("Original Image", img);
    original_score = Thenengrad(img);
    printf("Original score:%f\n", original_score);

    score = Thenengrad(temp);
    printf("Temp score:%f\n", score);

    //銳利化
    GaussianBlur(temp, result, Size(0,0), 5);
    addWeighted(temp, 1.5, result, -0.5, 0, result);

    namedWindow("Sharpened Image", cv::WINDOW_AUTOSIZE );
    imshow("Sharpened Image", result);
    sharpened_score = Thenengrad(result);
    printf("Sharpened score:%f\n", sharpened_score);
    printf("\n");
    waitKey(0);

    while(original_score - sharpened_score > 100000){
        Mat kernel = (Mat_<double>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
        filter2D(result, result, -1, kernel, Point(-1, -1), 0, 4);

        imshow("Original Image", img);
        imshow("Sharpened Image", result);
        sharpened_score = Thenengrad(result);
        printf("\n");
        printf("Original score:%f\n", original_score);
        printf("Sharpened score:%f\n", sharpened_score);
        waitKey(0);
    }

    split(result, channels);

	/// 使用色階分佈圖 (Histogram) 等化(Equalization)
	equalizeHist(channels[0], channels[0]);
	equalizeHist(channels[1], channels[1]);
	equalizeHist(channels[2], channels[2]);

	Mat mergesrc[3] = {channels[0], channels[1], channels[2]};
	merge(mergesrc, 3, mergedst);

	imshow("equalized_window", mergedst);
	equalized_score = Thenengrad(mergedst);
	printf("\n");
	printf("Equalized score:%f\n", equalized_score);
	waitKey(0);

	return 0;
}
