// src/resize.cpp

#include "resize.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <stdexcept>
#include <immintrin.h> // 如果需要AVX向量化
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


// 多线程最近邻插值模板实现，使用AVX
template <typename T>
Mat resizeNN(const Mat& input, int output_width, int output_height, int channels) {
    Mat output(output_height, output_width, input.type());

    double scale_width = static_cast<double>(output_width) / input.cols;
    double scale_height = static_cast<double>(output_height) / input.rows;

    cv::parallel_for_(cv::Range(0, output_height), [&](const cv::Range& range) {
        for(int y_dst = range.start; y_dst < range.end; ++y_dst){
            int y_src = std::min(static_cast<int>(round(y_dst / scale_height)), input.rows - 1);

            if (channels == 1) {
                const T* input_row = input.ptr<T>(y_src);
                T* output_row = output.ptr<T>(y_dst);

                int x = 0;
                for(; x <= output_width - 8; x += 8){
                    // 设置8个x_dst索引
                    __m256i x_dst_v = _mm256_set_epi32(
                        x+7, x+6, x+5, x+4, x+3, x+2, x+1, x
                    );

                    // 缩放并四舍五入
                    __m256 scale_v = _mm256_set1_ps(static_cast<float>(scale_width));
                    __m256 x_dst_f = _mm256_cvtepi32_ps(x_dst_v);
                    __m256 x_src_f = _mm256_div_ps(x_dst_f, scale_v);
                    __m256 x_src_rounded = _mm256_round_ps(x_src_f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    __m256i x_src_i = _mm256_cvtps_epi32(x_src_rounded);

                    // Clamp x_src_i to [0, input.cols - 1]
                    __m256i cols_minus_1 = _mm256_set1_epi32(input.cols - 1);
                    x_src_i = _mm256_min_epi32(x_src_i, cols_minus_1);
                    x_src_i = _mm256_max_epi32(x_src_i, _mm256_setzero_si256());

                    // 存储x_src_i到数组
                    alignas(32) int x_src_array[8];
                    _mm256_store_si256((__m256i*)x_src_array, x_src_i);

                    // 加载像素值
                    for(int i = 0; i < 8; ++i){
                        output_row[x + i] = input_row[x_src_array[i]];
                    }
                }

                // 处理剩余像素
                for(; x < output_width; ++x){
                    int x_src = std::min(static_cast<int>(round(x / scale_width)), input.cols - 1);
                    output_row[x] = input_row[x_src];
                }
            }
            else if (channels == 3) {
                const cv::Vec<T, 3>* input_row = input.ptr<cv::Vec<T, 3>>(y_src);
                cv::Vec<T, 3>* output_row = output.ptr<cv::Vec<T, 3>>(y_dst);

                int x = 0;
                for(; x <= output_width - 8; x += 8){
                    __m256i x_dst_v = _mm256_set_epi32(
                        x+7, x+6, x+5, x+4, x+3, x+2, x+1, x
                    );

                    __m256 scale_v = _mm256_set1_ps(static_cast<float>(scale_width));
                    __m256 x_dst_f = _mm256_cvtepi32_ps(x_dst_v);
                    __m256 x_src_f = _mm256_div_ps(x_dst_f, scale_v);
                    __m256 x_src_rounded = _mm256_round_ps(x_src_f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    __m256i x_src_i = _mm256_cvtps_epi32(x_src_rounded);

                    __m256i cols_minus_1 = _mm256_set1_epi32(input.cols - 1);
                    x_src_i = _mm256_min_epi32(x_src_i, cols_minus_1);
                    x_src_i = _mm256_max_epi32(x_src_i, _mm256_setzero_si256());

                    alignas(32) int x_src_array[8];
                    _mm256_store_si256((__m256i*)x_src_array, x_src_i);

                    for(int i = 0; i < 8; ++i){
                        output_row[x + i] = input_row[x_src_array[i]];
                    }
                }

                // 处理剩余像素
                for(; x < output_width; ++x){
                    int x_src = std::min(static_cast<int>(round(x / scale_width)), input.cols - 1);
                    output_row[x] = input_row[x_src];
                }
            }
        }
    });

    return output;
}

// 多线程双线性插值模板实现，使用AVX
template <typename T>
Mat resizeBilinear(const Mat& input, int output_width, int output_height, int channels) {
    Mat output(output_height, output_width, input.type());

    double scale_width = static_cast<double>(output_width) / input.cols;
    double scale_height = static_cast<double>(output_height) / input.rows;

    cv::parallel_for_(cv::Range(0, output_height), [&](const cv::Range& range) {
        for(int y_dst = range.start; y_dst < range.end; ++y_dst){
            double y_src_f = (y_dst + 0.5) / scale_height - 0.5;
            int y1 = std::max(0, std::min(static_cast<int>(floor(y_src_f)), input.rows - 1));
            int y2 = std::max(0, std::min(y1 + 1, input.rows - 1));
            double dy = y_src_f - y1;

            if (channels == 1) {
                T* output_row = output.ptr<T>(y_dst);
                int x = 0;

                for(; x <= output_width - 8; x += 8){
                    // 设置8个x_dst索引
                    __m256i x_dst_v = _mm256_set_epi32(
                        x+7, x+6, x+5, x+4, x+3, x+2, x+1, x
                    );

                    // 缩放
                    __m256 scale_v = _mm256_set1_ps(static_cast<float>(scale_width));
                    __m256 x_dst_f = _mm256_cvtepi32_ps(x_dst_v);
                    __m256 x_src_f = _mm256_div_ps(x_dst_f, scale_v);
                    __m256 x_src_floor = _mm256_floor_ps(x_src_f);
                    __m256 dx_v = _mm256_sub_ps(x_src_f, x_src_floor);
                    __m256i x1_i = _mm256_cvtps_epi32(x_src_floor);
                    __m256i x2_i = _mm256_add_epi32(x1_i, _mm256_set1_epi32(1));

                    // Clamp
                    __m256i cols_minus_1 = _mm256_set1_epi32(input.cols - 1);
                    x1_i = _mm256_min_epi32(x1_i, cols_minus_1);
                    x1_i = _mm256_max_epi32(x1_i, _mm256_setzero_si256());
                    x2_i = _mm256_min_epi32(x2_i, cols_minus_1);
                    x2_i = _mm256_max_epi32(x2_i, _mm256_setzero_si256());

                    // 存储索引
                    alignas(32) int indices1[8];
                    alignas(32) int indices2[8];
                    _mm256_store_si256((__m256i*)indices1, x1_i);
                    _mm256_store_si256((__m256i*)indices2, x2_i);

                    // 加载像素值
                    double I11[8], I21[8], I12[8], I22[8];
                    for(int i = 0; i < 8; ++i){
                        I11[i] = input.ptr<T>(y1)[indices1[i]];
                        I21[i] = input.ptr<T>(y1)[indices2[i]];
                        I12[i] = input.ptr<T>(y2)[indices1[i]];
                        I22[i] = input.ptr<T>(y2)[indices2[i]];
                    }

                    // 计算插值
                    for(int i = 0; i < 8; ++i){
                        double val = (1 - dx_v[i]) * (1 - dy) * I11[i] +
                                     dx_v[i] * (1 - dy) * I21[i] +
                                     (1 - dx_v[i]) * dy * I12[i] +
                                     dx_v[i] * dy * I22[i];
                        output_row[x + i] = clamp_and_round<T>(val);
                    }
                }

                // 处理剩余像素
                for(; x < output_width; ++x){
                    double x_src_f = (x + 0.5) / scale_width - 0.5;
                    int x1 = std::max(0, std::min(static_cast<int>(floor(x_src_f)), input.cols - 1));
                    int x2 = std::max(0, std::min(x1 + 1, input.cols - 1));
                    double dx = x_src_f - x1;

                    double I11 = input.ptr<T>(y1)[x1];
                    double I21 = input.ptr<T>(y1)[x2];
                    double I12 = input.ptr<T>(y2)[x1];
                    double I22 = input.ptr<T>(y2)[x2];

                    double val = (1 - dx) * (1 - dy) * I11 +
                                 dx * (1 - dy) * I21 +
                                 (1 - dx) * dy * I12 +
                                 dx * dy * I22;

                    output_row[x] = clamp_and_round<T>(val);
                }
            }
            else if (channels == 3) {
                cv::Vec<T, 3>* output_row = output.ptr<cv::Vec<T, 3>>(y_dst);
                int x = 0;

                for(; x <= output_width - 8; x += 8){
                    __m256i x_dst_v = _mm256_set_epi32(
                        x+7, x+6, x+5, x+4, x+3, x+2, x+1, x
                    );

                    __m256 scale_v = _mm256_set1_ps(static_cast<float>(scale_width));
                    __m256 x_dst_f = _mm256_cvtepi32_ps(x_dst_v);
                    __m256 x_src_f = _mm256_div_ps(x_dst_f, scale_v);
                    __m256 x_src_floor = _mm256_floor_ps(x_src_f);
                    __m256 dx_v = _mm256_sub_ps(x_src_f, x_src_floor);
                    __m256i x1_i = _mm256_cvtps_epi32(x_src_floor);
                    __m256i x2_i = _mm256_add_epi32(x1_i, _mm256_set1_epi32(1));

                    __m256i cols_minus_1 = _mm256_set1_epi32(input.cols - 1);
                    x1_i = _mm256_min_epi32(x1_i, cols_minus_1);
                    x1_i = _mm256_max_epi32(x1_i, _mm256_setzero_si256());
                    x2_i = _mm256_min_epi32(x2_i, cols_minus_1);
                    x2_i = _mm256_max_epi32(x2_i, _mm256_setzero_si256());

                    alignas(32) int indices1[8];
                    alignas(32) int indices2[8];
                    _mm256_store_si256((__m256i*)indices1, x1_i);
                    _mm256_store_si256((__m256i*)indices2, x2_i);

                    // 加载像素值并计算插值
                    for(int i = 0; i < 8; ++i){
                        double I11[3] = { 
                            static_cast<double>(input.ptr<cv::Vec<T, 3>>(y1)[indices1[i]][0]),
                            static_cast<double>(input.ptr<cv::Vec<T, 3>>(y1)[indices1[i]][1]),
                            static_cast<double>(input.ptr<cv::Vec<T, 3>>(y1)[indices1[i]][2])
                        };
                        double I21[3] = { 
                            static_cast<double>(input.ptr<cv::Vec<T, 3>>(y1)[indices2[i]][0]),
                            static_cast<double>(input.ptr<cv::Vec<T, 3>>(y1)[indices2[i]][1]),
                            static_cast<double>(input.ptr<cv::Vec<T, 3>>(y1)[indices2[i]][2])
                        };
                        double I12[3] = { 
                            static_cast<double>(input.ptr<cv::Vec<T, 3>>(y2)[indices1[i]][0]),
                            static_cast<double>(input.ptr<cv::Vec<T, 3>>(y2)[indices1[i]][1]),
                            static_cast<double>(input.ptr<cv::Vec<T, 3>>(y2)[indices1[i]][2])
                        };
                        double I22[3] = { 
                            static_cast<double>(input.ptr<cv::Vec<T, 3>>(y2)[indices2[i]][0]),
                            static_cast<double>(input.ptr<cv::Vec<T, 3>>(y2)[indices2[i]][1]),
                            static_cast<double>(input.ptr<cv::Vec<T, 3>>(y2)[indices2[i]][2])
                        };

                        cv::Vec<double, 3> val;
                        for(int c = 0; c < 3; ++c){
                            val[c] = (1 - dx_v[i]) * (1 - dy) * I11[c] +
                                     dx_v[i] * (1 - dy) * I21[c] +
                                     (1 - dx_v[i]) * dy * I12[c] +
                                     dx_v[i] * dy * I22[c];
                        }

                        // 赋值并确保每个通道正确
                        output_row[x + i][0] = clamp_and_round<T>(val[0]);
                        output_row[x + i][1] = clamp_and_round<T>(val[1]);
                        output_row[x + i][2] = clamp_and_round<T>(val[2]);
                    }
                }

                // 处理剩余像素
                for(; x < output_width; ++x){
                    double x_src_f = (x + 0.5) / scale_width - 0.5;
                    int x1 = std::max(0, std::min(static_cast<int>(floor(x_src_f)), input.cols - 1));
                    int x2 = std::max(0, std::min(x1 + 1, input.cols - 1));
                    double dx = x_src_f - x1;

                    cv::Vec<double, 3> I11 = input.ptr<cv::Vec<T, 3>>(y1)[x1];
                    cv::Vec<double, 3> I21 = input.ptr<cv::Vec<T, 3>>(y1)[x2];
                    cv::Vec<double, 3> I12 = input.ptr<cv::Vec<T, 3>>(y2)[x1];
                    cv::Vec<double, 3> I22 = input.ptr<cv::Vec<T, 3>>(y2)[x2];

                    cv::Vec<double, 3> val;
                    for(int c = 0; c < 3; ++c){
                        val[c] = (1 - dx) * (1 - dy) * I11[c] +
                                 dx * (1 - dy) * I21[c] +
                                 (1 - dx) * dy * I12[c] +
                                 dx * dy * I22[c];
                    }

                    output_row[x][0] = clamp_and_round<T>(val[0]);
                    output_row[x][1] = clamp_and_round<T>(val[1]);
                    output_row[x][2] = clamp_and_round<T>(val[2]);
                }
            }
        }
    });

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
