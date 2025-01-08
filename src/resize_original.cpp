// src/resize.cpp

#include "resize.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <stdexcept>

using namespace cv;

// 辅助模板函数：限制并四舍五入
template <typename T>
T clamp_and_round(double value);

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
    return static_cast<float>(value);
}

// 多线程最近邻插值，模板化实现不同深度数据类型。
template <typename T>
Mat resize_nn(const Mat& input, int out_w, int out_h, int channels) {
    Mat output(out_h, out_w, input.type());
    double scale_w = static_cast<double>(out_w) / input.cols;
    double scale_h = static_cast<double>(out_h) / input.rows;

    struct ResizeNNParallel : public ParallelLoopBody {
        const Mat& input;
        Mat& output;
        double scale_w, scale_h;
        int channels;

        ResizeNNParallel(const Mat& in, Mat& out, double sw, double sh, int ch)   //重写parallel类
            : input(in), output(out), scale_w(sw), scale_h(sh), channels(ch) {}

        void operator()(const Range& range) const override {
            for(int y = range.start; y < range.end; ++y){
                int y_src = std::min(static_cast<int>(round(y / scale_h)), input.rows - 1);
                if (channels == 1) {   //单通道处理
                    const T* in_row = input.ptr<T>(y_src);
                    T* out_row = output.ptr<T>(y);
                    for(int x = 0; x < output.cols; ++x){
                        int x_src = std::min(static_cast<int>(round(x / scale_w)), input.cols - 1);
                        out_row[x] = in_row[x_src];
                    }
                }
                else if (channels == 3) {   //多通道处理
                    const Vec<T, 3>* in_row = input.ptr<Vec<T, 3>>(y_src);
                    Vec<T, 3>* out_row = output.ptr<Vec<T, 3>>(y);
                    for(int x = 0; x < output.cols; ++x){
                        int x_src = std::min(static_cast<int>(round(x / scale_w)), input.cols - 1);
                        out_row[x] = in_row[x_src];
                    }
                }
            }
        }
    };

    ResizeNNParallel body(input, output, scale_w, scale_h, channels);
    parallel_for_(Range(0, out_h), body);   //调用多线程
    return output;
}

// 多线程双线性插值
template <typename T>
Mat resize_bilinear(const Mat& input, int out_w, int out_h, int channels) {
    Mat output(out_h, out_w, input.type());
    double scale_w = static_cast<double>(out_w) / input.cols;
    double scale_h = static_cast<double>(out_h) / input.rows;

    struct ResizeBilinearParallel : public ParallelLoopBody {
        const Mat& input;
        Mat& output;
        double scale_w, scale_h;
        int channels;

        ResizeBilinearParallel(const Mat& in, Mat& out, double sw, double sh, int ch)
            : input(in), output(out), scale_w(sw), scale_h(sh), channels(ch) {}

        void operator()(const Range& range) const override {
            for(int y = range.start; y < range.end; ++y){
                double y_src_f = (y + 0.5) / scale_h - 0.5;
                int y1 = std::max(0, std::min(static_cast<int>(floor(y_src_f)), input.rows - 1));
                int y2 = std::max(0, std::min(y1 + 1, input.rows - 1));
                double dy = y_src_f - y1;

                if (channels == 1) {
                    T* out_row = output.ptr<T>(y);
                    for(int x = 0; x < output.cols; ++x){
                        double x_src_f = (x + 0.5) / scale_w - 0.5;
                        int x1 = std::max(0, std::min(static_cast<int>(floor(x_src_f)), input.cols - 1));
                        int x2 = std::max(0, std::min(x1 + 1, input.cols - 1));
                        double dx = x_src_f - x1;

                        const T* row1 = input.ptr<T>(y1);   //用指针取出，比@要快很多
                        const T* row2 = input.ptr<T>(y2);
                        double I11 = row1[x1];
                        double I21 = row1[x2];
                        double I12 = row2[x1];
                        double I22 = row2[x2];

                        double val = (1 - dx) * (1 - dy) * I11 +
                                     dx * (1 - dy) * I21 +
                                     (1 - dx) * dy * I12 +
                                     dx * dy * I22;
                        out_row[x] = clamp_and_round<T>(val);
                    }
                }
                else if (channels == 3) {
                    Vec<T, 3>* out_row = output.ptr<Vec<T, 3>>(y);
                    for(int x = 0; x < output.cols; ++x){
                        double x_src_f = (x + 0.5) / scale_w - 0.5;
                        int x1 = std::max(0, std::min(static_cast<int>(floor(x_src_f)), input.cols - 1));
                        int x2 = std::max(0, std::min(x1 + 1, input.cols - 1));
                        double dx = x_src_f - x1;

                        const Vec<T, 3>* row1 = input.ptr<Vec<T, 3>>(y1);
                        const Vec<T, 3>* row2 = input.ptr<Vec<T, 3>>(y2);
                        Vec<T, 3> I11 = row1[x1];
                        Vec<T, 3> I21 = row1[x2];
                        Vec<T, 3> I12 = row2[x1];
                        Vec<T, 3> I22 = row2[x2];

                        Vec<double, 3> val;
                        for(int c = 0; c < 3; ++c){
                            val[c] = (1 - dx) * (1 - dy) * I11[c] +
                                     dx * (1 - dy) * I21[c] +
                                     (1 - dx) * dy * I12[c] +
                                     dx * dy * I22[c];
                            val[c] = clamp_and_round<T>(val[c]);
                        }
                        out_row[x] = Vec<T, 3>(
                            static_cast<T>(val[0]),
                            static_cast<T>(val[1]),
                            static_cast<T>(val[2])
                        );
                    }
                }
            }
        }
    };

    ResizeBilinearParallel body(input, output, scale_w, scale_h, channels);
    parallel_for_(Range(0, out_h), body);
    return output;
}

// 多线程最近邻
Mat resize_nn_mt(const Mat& input, int out_w, int out_h) {
    int channels = input.channels();
    int depth = input.depth();

    switch(depth) {   //不同depth实例化不同resize
        case CV_8U:
            return resize_nn<uchar>(input, out_w, out_h, channels);
        case CV_16U:
            return resize_nn<ushort>(input, out_w, out_h, channels);
        case CV_32F:
            return resize_nn<float>(input, out_w, out_h, channels);
        default:
            throw std::runtime_error("不支持的数据类型用于 resize_nn_mt");
    }
}

// 多线程双线性
Mat resize_bilinear_mt(const Mat& input, int out_w, int out_h) {
    int channels = input.channels();
    int depth = input.depth();

    switch(depth) {
        case CV_8U:
            return resize_bilinear<uchar>(input, out_w, out_h, channels);
        case CV_16U:
            return resize_bilinear<ushort>(input, out_w, out_h, channels);
        case CV_32F:
            return resize_bilinear<float>(input, out_w, out_h, channels);
        default:
            throw std::runtime_error("不支持的数据类型用于 resize_bilinear_mt");
    }
}

// 单线程最近邻实现
template <typename T>
Mat resize_nn_st_impl(const Mat& input, int out_w, int out_h, int channels) {
    Mat output(out_h, out_w, input.type());
    double scale_w = static_cast<double>(out_w) / input.cols;
    double scale_h = static_cast<double>(out_h) / input.rows;

    for(int y = 0; y < out_h; ++y){
        int y_src = std::min(static_cast<int>(round(y / scale_h)), input.rows - 1);
        if (channels == 1) {
            const T* in_row = input.ptr<T>(y_src);
            T* out_row = output.ptr<T>(y);
            for(int x = 0; x < out_w; ++x){
                int x_src = std::min(static_cast<int>(round(x / scale_w)), input.cols - 1);
                out_row[x] = in_row[x_src];
            }
        }
        else if (channels == 3) {
            const Vec<T, 3>* in_row = input.ptr<Vec<T, 3>>(y_src);
            Vec<T, 3>* out_row = output.ptr<Vec<T, 3>>(y);
            for(int x = 0; x < out_w; ++x){
                int x_src = std::min(static_cast<int>(round(x / scale_w)), input.cols - 1);
                out_row[x] = in_row[x_src];
            }
        }
    }
    return output;
}

// 单线程双线性实现
template <typename T>
Mat resize_bilinear_st_impl(const Mat& input, int out_w, int out_h, int channels) {
    Mat output(out_h, out_w, input.type());
    double scale_w = static_cast<double>(out_w) / input.cols;
    double scale_h = static_cast<double>(out_h) / input.rows;

    for(int y = 0; y < out_h; ++y){
        double y_src_f = (y + 0.5) / scale_h - 0.5;
        int y1 = std::max(0, std::min(static_cast<int>(floor(y_src_f)), input.rows - 1));
        int y2 = std::max(0, std::min(y1 + 1, input.rows - 1));
        double dy = y_src_f - y1;

        if (channels == 1) {
            T* out_row = output.ptr<T>(y);
            for(int x = 0; x < out_w; ++x){
                double x_src_f = (x + 0.5) / scale_w - 0.5;
                int x1 = std::max(0, std::min(static_cast<int>(floor(x_src_f)), input.cols - 1));
                int x2 = std::max(0, std::min(x1 + 1, input.cols - 1));
                double dx = x_src_f - x1;

                const T* row1 = input.ptr<T>(y1);
                const T* row2 = input.ptr<T>(y2);
                double I11 = row1[x1];
                double I21 = row1[x2];
                double I12 = row2[x1];
                double I22 = row2[x2];

                double val = (1 - dx) * (1 - dy) * I11 +
                             dx * (1 - dy) * I21 +
                             (1 - dx) * dy * I12 +
                             dx * dy * I22;
                out_row[x] = clamp_and_round<T>(val);
            }
        }
        else if (channels == 3) {
            Vec<T, 3>* out_row = output.ptr<Vec<T, 3>>(y);
            for(int x = 0; x < out_w; ++x){
                double x_src_f = (x + 0.5) / scale_w - 0.5;
                int x1 = std::max(0, std::min(static_cast<int>(floor(x_src_f)), input.cols - 1));
                int x2 = std::max(0, std::min(x1 + 1, input.cols - 1));
                double dx = x_src_f - x1;

                const Vec<T, 3>* row1 = input.ptr<Vec<T, 3>>(y1);
                const Vec<T, 3>* row2 = input.ptr<Vec<T, 3>>(y2);
                Vec<T, 3> I11 = row1[x1];
                Vec<T, 3> I21 = row1[x2];
                Vec<T, 3> I12 = row2[x1];
                Vec<T, 3> I22 = row2[x2];

                Vec<double, 3> val;
                for(int c = 0; c < 3; ++c){
                    val[c] = (1 - dx) * (1 - dy) * I11[c] +
                             dx * (1 - dy) * I21[c] +
                             (1 - dx) * dy * I12[c] +
                             dx * dy * I22[c];
                    val[c] = clamp_and_round<T>(val[c]);
                }
                out_row[x] = Vec<T, 3>(
                    static_cast<T>(val[0]),
                    static_cast<T>(val[1]),
                    static_cast<T>(val[2])
                );
            }
        }
    }
    return output;
}

// 单线程最近邻
Mat resize_nn_st(const Mat& input, int out_w, int out_h) {
    int channels = input.channels();
    int depth = input.depth();

    switch(depth) {
        case CV_8U:
            return resize_nn_st_impl<uchar>(input, out_w, out_h, channels);
        case CV_16U:
            return resize_nn_st_impl<ushort>(input, out_w, out_h, channels);
        case CV_32F:
            return resize_nn_st_impl<float>(input, out_w, out_h, channels);
        default:
            throw std::runtime_error("不支持的数据类型用于 resize_nn_st");
    }
}

// 单线程双线性
Mat resize_bilinear_st(const Mat& input, int out_w, int out_h) {
    int channels = input.channels();
    int depth = input.depth();

    switch(depth) {
        case CV_8U:
            return resize_bilinear_st_impl<uchar>(input, out_w, out_h, channels);
        case CV_16U:
            return resize_bilinear_st_impl<ushort>(input, out_w, out_h, channels);
        case CV_32F:
            return resize_bilinear_st_impl<float>(input, out_w, out_h, channels);
        default:
            throw std::runtime_error("不支持的数据类型用于 resize_bilinear_st");
    }
}

// 自定义resize函数
void resize_custom(const Mat& input, Mat& output, const Size& new_size, InterpolationMethod method) {
    if(method == NEAREST_NEIGHBOR){
        output = resize_nn_mt(input, new_size.width, new_size.height);
    }
    else if(method == BILINEAR){
        output = resize_bilinear_mt(input, new_size.width, new_size.height);
    }
    else{
        output = resize_nn_mt(input, new_size.width, new_size.height);
    }
}
