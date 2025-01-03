// src/resize.h

#ifndef RESIZE_H
#define RESIZE_H

#include <opencv2/opencv.hpp>

// 定义插值方法枚举
enum InterpolationMethod {
    NEAREST_NEIGHBOR,
    BILINEAR
};

// 多线程版本的Resize函数
void resize_custom(const cv::Mat& input, cv::Mat& output, const cv::Size& new_size, InterpolationMethod method);

// 单线程版本的最近邻插值
cv::Mat resizeNN_ST(const cv::Mat& input, int output_width, int output_height);

// 单线程版本的双线性插值
cv::Mat resizeBilinear_ST(const cv::Mat& input, int output_width, int output_height);

// 多线程版本的最近邻插值
cv::Mat resizeNN_MT(const cv::Mat& input, int output_width, int output_height);

// 多线程版本的双线性插值
cv::Mat resizeBilinear_MT(const cv::Mat& input, int output_width, int output_height);


#endif // RESIZE_H
