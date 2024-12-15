// resize.h
#ifndef RESIZE_H
#define RESIZE_H

#include <opencv2/opencv.hpp>

// 定义插值方法枚举类型
enum InterpolationMethod {
    NEAREST_NEIGHBOR,
    BILINEAR
};

// 声明自定义的resize函数
void resize_custom(const cv::Mat& input, cv::Mat& output, const cv::Size& new_size, InterpolationMethod method);

#endif // RESIZE_H
