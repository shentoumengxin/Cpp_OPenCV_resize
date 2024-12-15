// resize.cpp
#include "resize.h"
#include <cmath>
#include <algorithm>
#include <opencv2/core.hpp>

// 假设你已经实现了以下两个函数
cv::Mat resizeNN_MT(const cv::Mat& input, int output_width, int output_height);
cv::Mat resizeBilinear_MT(const cv::Mat& input, int output_width, int output_height);

// 实现自定义的resize函数
void resize_custom(const cv::Mat& input, cv::Mat& output, const cv::Size& new_size, InterpolationMethod method) {
    if(method == NEAREST_NEIGHBOR){
        output = resizeNN_MT(input, new_size.width, new_size.height);
    }
    else if(method == BILINEAR){
        output = resizeBilinear_MT(input, new_size.width, new_size.height);
    }
    else{
        // 默认使用最近邻插值
        output = resizeNN_MT(input, new_size.width, new_size.height);
    }
}

// 下面是示例实现，你需要根据项目需求填充具体逻辑

cv::Mat resizeNN_MT(const cv::Mat& input, int output_width, int output_height) {
    // 实现最近邻插值的多线程优化版本
    // 这里可以使用之前提供的 ResizeNN_Parallel 结构体和 cv::parallel_for_
    cv::Mat output(output_height, output_width, input.type());

    double scale_width = static_cast<double>(output_width) / input.cols;
    double scale_height = static_cast<double>(output_height) / input.rows;

    struct ResizeNN_Parallel : public cv::ParallelLoopBody {
        const cv::Mat& input;
        cv::Mat& output;
        double scale_width;
        double scale_height;

        ResizeNN_Parallel(const cv::Mat& in, cv::Mat& out, double sw, double sh)
            : input(in), output(out), scale_width(sw), scale_height(sh) {}

        virtual void operator()(const cv::Range& range) const CV_OVERRIDE {
            for(int y_dst = range.start; y_dst < range.end; ++y_dst){
                for(int x_dst = 0; x_dst < output.cols; ++x_dst){
                    int x_src = std::round(x_dst / scale_width);
                    int y_src = std::round(y_dst / scale_height);

                    x_src = std::min(x_src, input.cols - 1);
                    y_src = std::min(y_src, input.rows - 1);

                    if(input.channels() == 1){
                        output.at<uchar>(y_dst, x_dst) = input.at<uchar>(y_src, x_src);
                    }
                    else if(input.channels() == 3){
                        output.at<cv::Vec3b>(y_dst, x_dst) = input.at<cv::Vec3b>(y_src, x_src);
                    }
                }
            }
        }
    };

    ResizeNN_Parallel body(input, output, scale_width, scale_height);
    cv::parallel_for_(cv::Range(0, output_height), body);

    return output;
}

cv::Mat resizeBilinear_MT(const cv::Mat& input, int output_width, int output_height) {
    // 实现双线性插值的多线程优化版本
    // 这里可以使用之前提供的 ResizeBilinear_Parallel 结构体和 cv::parallel_for_
    cv::Mat output(output_height, output_width, input.type());

    double scale_width = static_cast<double>(output_width) / input.cols;
    double scale_height = static_cast<double>(output_height) / input.rows;

    struct ResizeBilinear_Parallel : public cv::ParallelLoopBody {
        const cv::Mat& input;
        cv::Mat& output;
        double scale_width;
        double scale_height;

        ResizeBilinear_Parallel(const cv::Mat& in, cv::Mat& out, double sw, double sh)
            : input(in), output(out), scale_width(sw), scale_height(sh) {}

        virtual void operator()(const cv::Range& range) const CV_OVERRIDE {
            for(int y_dst = range.start; y_dst < range.end; ++y_dst){
                for(int x_dst = 0; x_dst < output.cols; ++x_dst){
                    // 双线性插值逻辑
                    double x_src_f = (x_dst + 0.5) / scale_width - 0.5;
                    double y_src_f = (y_dst + 0.5) / scale_height - 0.5;

                    int x1 = static_cast<int>(std::floor(x_src_f));
                    int y1 = static_cast<int>(std::floor(y_src_f));
                    int x2 = x1 + 1;
                    int y2 = y1 + 1;

                    double dx = x_src_f - x1;
                    double dy = y_src_f - y1;

                    // Clamp coordinates
                    x1 = std::max(0, std::min(x1, input.cols - 1));
                    y1 = std::max(0, std::min(y1, input.rows - 1));
                    x2 = std::max(0, std::min(x2, input.cols - 1));
                    y2 = std::max(0, std::min(y2, input.rows - 1));

                    if(input.channels() == 1){
                        double I11 = input.at<uchar>(y1, x1);
                        double I21 = input.at<uchar>(y1, x2);
                        double I12 = input.at<uchar>(y2, x1);
                        double I22 = input.at<uchar>(y2, x2);

                        double val = (1 - dx) * (1 - dy) * I11 +
                                     dx * (1 - dy) * I21 +
                                     (1 - dx) * dy * I12 +
                                     dx * dy * I22;

                        output.at<uchar>(y_dst, x_dst) = static_cast<uchar>(std::round(val));
                    }
                    else if(input.channels() == 3){
                        cv::Vec3d I11 = input.at<cv::Vec3b>(y1, x1);
                        cv::Vec3d I21 = input.at<cv::Vec3b>(y1, x2);
                        cv::Vec3d I12 = input.at<cv::Vec3b>(y2, x1);
                        cv::Vec3d I22 = input.at<cv::Vec3b>(y2, x2);

                        cv::Vec3d val;
                        for(int c = 0; c < 3; ++c){
                            val[c] = (1 - dx) * (1 - dy) * I11[c] +
                                     dx * (1 - dy) * I21[c] +
                                     (1 - dx) * dy * I12[c] +
                                     dx * dy * I22[c];
                            val[c] = std::round(val[c]);
                            val[c] = std::max(0.0, std::min(val[c], 255.0));
                        }

                        output.at<cv::Vec3b>(y_dst, x_dst) = cv::Vec3b(static_cast<uchar>(val[0]),
                                                                        static_cast<uchar>(val[1]),
                                                                        static_cast<uchar>(val[2]));
                    }
                }
            }
        }
    };

    ResizeBilinear_Parallel body(input, output, scale_width, scale_height);
    cv::parallel_for_(cv::Range(0, output_height), body);

    return output;
}
