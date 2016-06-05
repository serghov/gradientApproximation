#define byte unsigned char

#include "cv.h"
#include "highgui.h"
#include <random>
#include <cmath>
#include <thread>
#include <mutex>
#include <chrono>
using namespace std;
using namespace cv;



Mat img;

void makeGradient(Mat &img, Mat &grad, byte tl, byte tr, byte bl, byte br)
{
    grad = img.clone();
    int i, j;
    for (i = 0; i < img.rows; i++)
    {
        for (j = 0; j < img.cols; j++)
        {
            byte l = tl + (bl - tl) * i / (img.rows + 0.0);
            byte r = tr + (br - tr) * i / (img.rows + 0.0);
            grad.at<byte>(i, j) = l + (r - l) * j / (img.cols + 0.0);

            byte a = tl, b = tr, c = bl, d = br;
            double x = j, y = i;
            double w = img.cols, h = img.rows;
            grad.at<byte>(i, j) = (a * ((h - y)*(w - x)) + c * (y * (w - x)) + b * ((h - y) * x) + d * (x * y)) / h / w;
        }
    }
}

void calcDerivatives(Mat &img, double a, double b, double c, double d,
                     double &da, double &db, double &dc, double &dd)
{
    double h = img.rows;
    double w = img.cols;
    da = 0;
    db = 0;
    dc = 0;
    dd = 0;
    int i, j;
    for (i = 0; i < img.rows; i++)
    {
        for (j = 0; j < img.cols; j++)
        {
            double y = i / h, x = j / w;
            da += -2 * (double) img.at<byte>(i, j)*(1 - y)*(1 - x) + 2 * a * (1 - y)*(1 - y)*(1 - x)*(1 - x) + 2 * b * (1 - y)*(1 - x)*(1 - y) * x +
                    2 * c * (1 - y)*(1 - x) * y * (1 - x) + 2 * d * (1 - y)*(1 - x) * x * y;
            db += -2 * (double) img.at<byte>(i, j)*(1 - y) * x + 2 * b * (1 - y)*(1 - y) * x * x + 2 * a * (1 - y) * x * (1 - y)*(1 - x) +
                    2 * c * (1 - y) * x * y * (1 - x) + 2 * d * (1 - y) * x * y * x;
            dc += -2 * (double) img.at<byte>(i, j) * y * (1 - x) + 2 * c * y * y * (1 - x)*(1 - x) + 2 * a * y * (1 - x)*(1 - y)*(1 - x) +
                    2 * b * y * (1 - x)*(1 - y) * x + 2 * d * y * (1 - x) * y * x;
            dd += -2 * (double) img.at<byte>(i, j) * x * y + 2 * d * x * x * y * y + 2 * a * y * x * (1 - y)*(1 - x) +
                    2 * b * y * x * (1 - y) * x + 2 * c * y * x * y * (1 - x);

        }
    }
    da /= h*w;
    db /= h*w;
    dc /= h*w;
    dd /= h*w;
}

double getSquareError(Mat &img, byte a, byte b, byte c, byte d)
{
    double res = 0;
    int i, j;
    for (i = 0; i < img.rows; i++)
    {
        for (j = 0; j < img.cols; j++)
        {
            byte l = a + (c - a) * i / (img.rows + 0.0);
            byte r = b + (d - b) * i / (img.rows + 0.0);
            byte cur = l + (r - l) * j / (img.cols + 0.0);
            res += ((double) img.at<byte>(i, j) - (double) cur) *((double) img.at<byte>(i, j) - (double) cur);
        }
    }
    return res / (img.rows * img.cols + 0.0);
}

/*
 * 
 */
int main()
{

    img = imread("test.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat grad;


    double da, db, dc, dd;
    double a = 255, b = 255, c = 255, d = 255;
    double alpha = 0.1;
    for (int i = 0; i < 1000; i++)
    {
        calcDerivatives(img, a, b, c, d, da, db, dc, dd);
        //cout << da << " " << db << " " << dc << " " << dd << endl;
        a += -alpha*da;
        b += -alpha*db;
        c += -alpha*dc;
        d += -alpha*dd;

        a < 0 ? a = 0 : 1;
        b < 0 ? b = 0 : 1;
        c < 0 ? c = 0 : 1;
        d < 0 ? d = 0 : 1;

        a > 255 ? a = 255 : 1;
        b > 255 ? b = 255 : 1;
        c > 255 ? c = 255 : 1;
        d > 255 ? d = 255 : 1;


        makeGradient(img, grad, a, b, c, d);
        cout << a << " " << b << " " << c << " " << d << " ";
        cout << getSquareError(img, a, b, c, d) << endl;
        //absdiff(img, grad, grad);
        imshow("img", grad);
        if (waitKey(10) > 0)
            return 0;
    }


    //absdiff(img, grad, img);

    waitKey(0);


    return 0;
}