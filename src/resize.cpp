// src/resize.cpp

#include "resize.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <stdexcept>

// 使用cv命名空间中的Mat
using cv::Mat;

// 辅助模板函数：线性插值并确保结果在有效范围内
template <typename T>
T clamp_and_round(double value);

// 特化模板函数用于不同的数据类型
template <>
uchar clamp_and_round<uchar>(double value) {
    return static_cast<uchar>(std::round(std::min(std::max(value, 0.0), 255.0)));
}

template <>
ushort clamp_and_round<ushort>(double value) {
    return static_cast<ushort>(std::round(std::min(std::max(value, 0.0), 65535.0)));
}

template <>
float clamp_and_round<float>(double value) {
    // 对于浮点数，通常不需要限制范围，但根据需求可以添加限制
    return static_cast<float>(value);
}

// 多线程最近邻插值模板实现
template <typename T>
Mat resizeNN(const Mat& input, int output_width, int output_height, int channels) {
    Mat output(output_height, output_width, input.type());

    double scale_width = static_cast<double>(output_width) / input.cols;
    double scale_height = static_cast<double>(output_height) / input.rows;

    struct ResizeNN_Parallel : public cv::ParallelLoopBody {
        const Mat& input;
        Mat& output;
        double scale_width;
        double scale_height;
        int channels;

        ResizeNN_Parallel(const Mat& in, Mat& out, double sw, double sh, int ch)
            : input(in), output(out), scale_width(sw), scale_height(sh), channels(ch) {}

        virtual void operator()(const cv::Range& range) const CV_OVERRIDE {
            for(int y_dst = range.start; y_dst < range.end; ++y_dst){
                // 反向映射到输入图像的y坐标
                int y_src = static_cast<int>(round(y_dst / scale_height));
                y_src = std::min(y_src, input.rows - 1);

                if (channels == 1) {
                    const T* input_row = input.ptr<T>(y_src);
                    T* output_row = output.ptr<T>(y_dst);
                    for(int x_dst = 0; x_dst < output.cols; ++x_dst){
                        int x_src = static_cast<int>(round(x_dst / scale_width));
                        x_src = std::min(x_src, input.cols - 1);
                        output_row[x_dst] = input_row[x_src];
                    }
                }
                else if (channels == 3) {
                    const cv::Vec<T, 3>* input_row = input.ptr<cv::Vec<T, 3>>(y_src);
                    cv::Vec<T, 3>* output_row = output.ptr<cv::Vec<T, 3>>(y_dst);
                    for(int x_dst = 0; x_dst < output.cols; ++x_dst){
                        int x_src = static_cast<int>(round(x_dst / scale_width));
                        x_src = std::min(x_src, input.cols - 1);
                        output_row[x_dst] = input_row[x_src];
                    }
                }
            }
        }
    };

    ResizeNN_Parallel body(input, output, scale_width, scale_height, channels);
    cv::parallel_for_(cv::Range(0, output_height), body);

    return output;
}

// 多线程双线性插值模板实现
template <typename T>
Mat resizeBilinear(const Mat& input, int output_width, int output_height, int channels) {
    Mat output(output_height, output_width, input.type());

    double scale_width = static_cast<double>(output_width) / input.cols;
    double scale_height = static_cast<double>(output_height) / input.rows;

    struct ResizeBilinear_Parallel : public cv::ParallelLoopBody {
        const Mat& input;
        Mat& output;
        double scale_width;
        double scale_height;
        int channels;

        ResizeBilinear_Parallel(const Mat& in, Mat& out, double sw, double sh, int ch)
            : input(in), output(out), scale_width(sw), scale_height(sh), channels(ch) {}

        virtual void operator()(const cv::Range& range) const CV_OVERRIDE {
            for(int y_dst = range.start; y_dst < range.end; ++y_dst){
                // 反向映射到输入图像的y坐标
                double y_src_f = (y_dst + 0.5) / scale_height - 0.5;
                int y1 = static_cast<int>(floor(y_src_f));
                int y2 = y1 + 1;
                double dy = y_src_f - y1;

                // Clamp y coordinates
                y1 = std::max(0, std::min(y1, input.rows - 1));
                y2 = std::max(0, std::min(y2, input.rows - 1));

                if (channels == 1) {
                    T* output_row = output.ptr<T>(y_dst);
                    for(int x_dst = 0; x_dst < output.cols; ++x_dst){
                        // 反向映射到输入图像的x坐标
                        double x_src_f = (x_dst + 0.5) / scale_width - 0.5;
                        int x1 = static_cast<int>(floor(x_src_f));
                        int x2 = x1 + 1;
                        double dx = x_src_f - x1;

                        // Clamp x coordinates
                        x1 = std::max(0, std::min(x1, input.cols - 1));
                        x2 = std::max(0, std::min(x2, input.cols - 1));

                        // 获取四个邻近像素的值
                        const T* row1 = input.ptr<T>(y1);
                        const T* row2 = input.ptr<T>(y2);
                        double I11 = row1[x1];
                        double I21 = row1[x2];
                        double I12 = row2[x1];
                        double I22 = row2[x2];

                        // 计算双线性插值结果
                        double val = (1 - dx) * (1 - dy) * I11 +
                                     dx * (1 - dy) * I21 +
                                     (1 - dx) * dy * I12 +
                                     dx * dy * I22;

                        // 赋值给输出图像，确保不溢出
                        output_row[x_dst] = clamp_and_round<T>(val);
                    }
                }
                else if (channels == 3) {
                    cv::Vec<T, 3>* output_row = output.ptr<cv::Vec<T, 3>>(y_dst);
                    for(int x_dst = 0; x_dst < output.cols; ++x_dst){
                        // 反向映射到输入图像的x坐标
                        double x_src_f = (x_dst + 0.5) / scale_width - 0.5;
                        int x1 = static_cast<int>(floor(x_src_f));
                        int x2 = x1 + 1;
                        double dx = x_src_f - x1;

                        // Clamp x coordinates
                        x1 = std::max(0, std::min(x1, input.cols - 1));
                        x2 = std::max(0, std::min(x2, input.cols - 1));

                        // 获取四个邻近像素的值
                        const cv::Vec<T, 3>* row1 = input.ptr<cv::Vec<T, 3>>(y1);
                        const cv::Vec<T, 3>* row2 = input.ptr<cv::Vec<T, 3>>(y2);
                        cv::Vec<T, 3> I11 = row1[x1];
                        cv::Vec<T, 3> I21 = row1[x2];
                        cv::Vec<T, 3> I12 = row2[x1];
                        cv::Vec<T, 3> I22 = row2[x2];

                        // 计算双线性插值结果
                        cv::Vec<double, 3> val;
                        for(int c = 0; c < 3; ++c){
                            val[c] = (1 - dx) * (1 - dy) * I11[c] +
                                     dx * (1 - dy) * I21[c] +
                                     (1 - dx) * dy * I12[c] +
                                     dx * dy * I22[c];
                            val[c] = clamp_and_round<T>(val[c]);
                        }

                        // 赋值给输出图像
                        output_row[x_dst] = cv::Vec<T, 3>(
                            static_cast<T>(val[0]),
                            static_cast<T>(val[1]),
                            static_cast<T>(val[2])
                        );
                    }
                }
            }
        }
    };

    ResizeBilinear_Parallel body(input, output, scale_width, scale_height, channels);
    cv::parallel_for_(cv::Range(0, output_height), body);

    return output;
}

// 多线程最近邻插值实现
Mat resizeNN_MT(const Mat& input, int output_width, int output_height) {
    int channels = input.channels();
    int depth = input.depth();

    switch(depth) {
        case CV_8U:
            return resizeNN<uchar>(input, output_width, output_height, channels);
        case CV_16U:
            return resizeNN<ushort>(input, output_width, output_height, channels);
        case CV_32F:
            return resizeNN<float>(input, output_width, output_height, channels);
        default:
            throw std::runtime_error("Unsupported data type for resizeNN_MT");
    }
}

// 多线程双线性插值实现
Mat resizeBilinear_MT(const Mat& input, int output_width, int output_height) {
    int channels = input.channels();
    int depth = input.depth();

    switch(depth) {
        case CV_8U:
            return resizeBilinear<uchar>(input, output_width, output_height, channels);
        case CV_16U:
            return resizeBilinear<ushort>(input, output_width, output_height, channels);
        case CV_32F:
            return resizeBilinear<float>(input, output_width, output_height, channels);
        default:
            throw std::runtime_error("Unsupported data type for resizeBilinear_MT");
    }
}

// 单线程最近邻插值模板实现
template <typename T>
Mat resizeNN_ST_impl(const Mat& input, int output_width, int output_height, int channels) {
    Mat output(output_height, output_width, input.type());

    double scale_width = static_cast<double>(output_width) / input.cols;
    double scale_height = static_cast<double>(output_height) / input.rows;

    for(int y_dst = 0; y_dst < output_height; ++y_dst){
        // 反向映射到输入图像的y坐标
        int y_src = static_cast<int>(round(y_dst / scale_height));
        y_src = std::min(y_src, input.rows - 1);

        if (channels == 1) {
            const T* input_row = input.ptr<T>(y_src);
            T* output_row = output.ptr<T>(y_dst);
            for(int x_dst = 0; x_dst < output_width; ++x_dst){
                int x_src = static_cast<int>(round(x_dst / scale_width));
                x_src = std::min(x_src, input.cols - 1);
                output_row[x_dst] = input_row[x_src];
            }
        }
        else if (channels == 3) {
            const cv::Vec<T, 3>* input_row = input.ptr<cv::Vec<T, 3>>(y_src);
            cv::Vec<T, 3>* output_row = output.ptr<cv::Vec<T, 3>>(y_dst);
            for(int x_dst = 0; x_dst < output_width; ++x_dst){
                int x_src = static_cast<int>(round(x_dst / scale_width));
                x_src = std::min(x_src, input.cols - 1);
                output_row[x_dst] = input_row[x_src];
            }
        }
    }

    return output;
}

// 单线程双线性插值模板实现
template <typename T>
Mat resizeBilinear_ST_impl(const Mat& input, int output_width, int output_height, int channels) {
    Mat output(output_height, output_width, input.type());

    double scale_width = static_cast<double>(output_width) / input.cols;
    double scale_height = static_cast<double>(output_height) / input.rows;

    for(int y_dst = 0; y_dst < output_height; ++y_dst){
        // 反向映射到输入图像的y坐标
        double y_src_f = (y_dst + 0.5) / scale_height - 0.5;
        int y1 = static_cast<int>(floor(y_src_f));
        int y2 = y1 + 1;
        double dy = y_src_f - y1;

        // Clamp y coordinates
        y1 = std::max(0, std::min(y1, input.rows - 1));
        y2 = std::max(0, std::min(y2, input.rows - 1));

        if (channels == 1) {
            T* output_row = output.ptr<T>(y_dst);
            for(int x_dst = 0; x_dst < output_width; ++x_dst){
                // 反向映射到输入图像的x坐标
                double x_src_f = (x_dst + 0.5) / scale_width - 0.5;
                int x1 = static_cast<int>(floor(x_src_f));
                int x2 = x1 + 1;
                double dx = x_src_f - x1;

                // Clamp x coordinates
                x1 = std::max(0, std::min(x1, input.cols - 1));
                x2 = std::max(0, std::min(x2, input.cols - 1));

                // 获取四个邻近像素的值
                const T* row1 = input.ptr<T>(y1);
                const T* row2 = input.ptr<T>(y2);
                double I11 = row1[x1];
                double I21 = row1[x2];
                double I12 = row2[x1];
                double I22 = row2[x2];

                // 计算双线性插值结果
                double val = (1 - dx) * (1 - dy) * I11 +
                             dx * (1 - dy) * I21 +
                             (1 - dx) * dy * I12 +
                             dx * dy * I22;

                // 赋值给输出图像，确保不溢出
                output_row[x_dst] = clamp_and_round<T>(val);
            }
        }
        else if (channels == 3) {
            cv::Vec<T, 3>* output_row = output.ptr<cv::Vec<T, 3>>(y_dst);
            for(int x_dst = 0; x_dst < output_width; ++x_dst){
                // 反向映射到输入图像的x坐标
                double x_src_f = (x_dst + 0.5) / scale_width - 0.5;
                int x1 = static_cast<int>(floor(x_src_f));
                int x2 = x1 + 1;
                double dx = x_src_f - x1;

                // Clamp x coordinates
                x1 = std::max(0, std::min(x1, input.cols - 1));
                x2 = std::max(0, std::min(x2, input.cols - 1));

                // 获取四个邻近像素的值
                const cv::Vec<T, 3>* row1 = input.ptr<cv::Vec<T, 3>>(y1);
                const cv::Vec<T, 3>* row2 = input.ptr<cv::Vec<T, 3>>(y2);
                cv::Vec<T, 3> I11 = row1[x1];
                cv::Vec<T, 3> I21 = row1[x2];
                cv::Vec<T, 3> I12 = row2[x1];
                cv::Vec<T, 3> I22 = row2[x2];

                // 计算双线性插值结果
                cv::Vec<double, 3> val;
                for(int c = 0; c < 3; ++c){
                    val[c] = (1 - dx) * (1 - dy) * I11[c] +
                             dx * (1 - dy) * I21[c] +
                             (1 - dx) * dy * I12[c] +
                             dx * dy * I22[c];
                    val[c] = clamp_and_round<T>(val[c]);
                }

                // 赋值给输出图像
                output_row[x_dst] = cv::Vec<T, 3>(
                    static_cast<T>(val[0]),
                    static_cast<T>(val[1]),
                    static_cast<T>(val[2])
                );
            }
        }
    }

    return output;
}

// 单线程最近邻插值实现
Mat resizeNN_ST(const Mat& input, int output_width, int output_height) {
    int channels = input.channels();
    int depth = input.depth();

    switch(depth) {
        case CV_8U:
            return resizeNN_ST_impl<uchar>(input, output_width, output_height, channels);
        case CV_16U:
            return resizeNN_ST_impl<ushort>(input, output_width, output_height, channels);
        case CV_32F:
            return resizeNN_ST_impl<float>(input, output_width, output_height, channels);
        default:
            throw std::runtime_error("Unsupported data type for resizeNN_ST");
    }
}

// 单线程双线性插值实现
Mat resizeBilinear_ST(const Mat& input, int output_width, int output_height) {
    int channels = input.channels();
    int depth = input.depth();

    switch(depth) {
        case CV_8U:
            return resizeBilinear_ST_impl<uchar>(input, output_width, output_height, channels);
        case CV_16U:
            return resizeBilinear_ST_impl<ushort>(input, output_width, output_height, channels);
        case CV_32F:
            return resizeBilinear_ST_impl<float>(input, output_width, output_height, channels);
        default:
            throw std::runtime_error("Unsupported data type for resizeBilinear_ST");
    }
}

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
