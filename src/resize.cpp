// src/resize.cpp

#include "resize.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <stdexcept>
#include <immintrin.h> // For AVX intrinsics
#include <omp.h>       // For OpenMP
namespace customfinal
{ // src/resize.cpp
    // src/resize.cpp

    using cv::Mat;
    using namespace std;
    // 这里就是相当于取整的函数
    template <typename T>
    T clamp_and_round(double value);

    // 相当于对不同类型的取整，uchar,ushort,udoube这里我们也使用了一个模板去构建它们
    template <>
    uchar clamp_and_round<uchar>(double value)
    {
        return static_cast<uchar>(std::round(std::min(std::max(value, 0.0), 255.0)));
    }

    template <>
    ushort clamp_and_round<ushort>(double value)
    {
        return static_cast<ushort>(std::round(std::min(std::max(value, 0.0), 65535.0)));
    }

    template <>
    float clamp_and_round<float>(double value)
    {
        // For floating points, clamping can be adjusted based on requirements
        return static_cast<float>(value);
    }

    vector<int> cal_vector(int limit, double scale, const Mat &input_continuous)
    {
        std::vector<int> y(limit);
#pragma omp parallel for schedule(dynamic)
        for (int y_dst = 0; y_dst < limit; ++y_dst)
        {
            int y_src = static_cast<int>(round(y_dst * scale));
            y_src = std::min(y_src, input_continuous.rows - 1);
            y[y_dst] = y_src;
        }
        return y;
    }
    // Multi-threaded Nearest Neighbor Interpolation with AVX Optimization
    // 多线性临近
    template <typename T>
    Mat resizeNN_AVX(const Mat &input, int output_width, int output_height, int channels)
    {
        // Ensure the input matrix is continuous for optimized memory access
        Mat input_continuous = input.isContinuous() ? input : input.clone();
        Mat output(output_height, output_width, input_continuous.type());

        // Calculate inverse scaling factors to avoid division in loops
        double inv_scale_width = static_cast<double>(input_continuous.cols) / output_width;
        double inv_scale_height = static_cast<double>(input_continuous.rows) / output_height;
        // Precompute y_src and x_src mappings
        std::vector<int> y_mapping(output_height), x_mapping(output_width);
#pragma omp parallel for schedule(dynamic)
        for (int y_dst = 0; y_dst < output_height; ++y_dst)
        {
            int y_src = static_cast<int>(round(y_dst * inv_scale_height));
            y_src = std::min(y_src, input_continuous.rows - 1);
            y_mapping[y_dst] = y_src;
        }
#pragma omp parallel for schedule(dynamic)
        for (int x_dst = 0; x_dst < output_width; ++x_dst)
        {
            int x_src = static_cast<int>(round(x_dst * inv_scale_width));
            x_src = std::min(x_src, input_continuous.cols - 1);
            x_mapping[x_dst] = x_src;
        }
        // Process based on the number of channels
        if (channels == 1)
        {
            // Single-channel image (Grayscale)
#pragma omp parallel for schedule(dynamic)
            for (int y_dst = 0; y_dst < output_height; ++y_dst)
            {
                int y_src = y_mapping[y_dst];
                const T *input_row = input_continuous.ptr<T>(y_src);
                T *output_row = output.ptr<T>(y_dst);
                int x = 0;

                // Process 8 pixels at a time using AVX2
                for (; x <= output_width - 8; x += 8)
                {
                    // Load 8 x_src indices
                    __m256i x_src_v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&x_mapping[x]));

                    // Clamp indices to [0, cols - 1]
                    __m256i cols_minus_1 = _mm256_set1_epi32(input_continuous.cols - 1);
                    x_src_v = _mm256_min_epi32(x_src_v, cols_minus_1);
                    x_src_v = _mm256_max_epi32(x_src_v, _mm256_setzero_si256());

                    // Extract indices into an array
                    alignas(32) int x_src_array[8];
                    _mm256_store_si256(reinterpret_cast<__m256i *>(x_src_array), x_src_v);

                    // Gather pixel values
                    alignas(32) T pixels[8];
                    for (int i = 0; i < 8; ++i)
                    {
                        pixels[i] = input_row[x_src_array[i]];
                    }

                    // Load pixels into AVX registers
                    __m256 pixels_v = _mm256_loadu_ps(reinterpret_cast<float *>(pixels));

                    // Convert pixels back to integers
                    __m256i pixels_i = _mm256_cvtps_epi32(pixels_v);

                    // Pack and store the results
                    alignas(32) int packed_pixels[8];
                    _mm256_store_si256(reinterpret_cast<__m256i *>(packed_pixels), pixels_i);
                    for (int i = 0; i < 8; ++i)
                    {
                        output_row[x + i] = static_cast<T>(packed_pixels[i]);
                    }
                }

                // Handle remaining pixels
                for (; x < output_width; ++x)
                {
                    int x_src = x_mapping[x];
                    output_row[x] = input_row[x_src];
                }
            }
        }
        else if (channels == 3)
        {
            // Three-channel image (Color)
#pragma omp parallel for schedule(dynamic)
            for (int y_dst = 0; y_dst < output_height; ++y_dst)
            {
                int y_src = y_mapping[y_dst];
                const cv::Vec<T, 3> *input_row = input_continuous.ptr<cv::Vec<T, 3>>(y_src);
                cv::Vec<T, 3> *output_row = output.ptr<cv::Vec<T, 3>>(y_dst);
                int x = 0;

                // Process 8 pixels at a time using AVX2
                for (; x <= output_width - 8; x += 8)
                {
                    // Load 8 x_src indices
                    __m256i x_src_v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&x_mapping[x]));

                    // Clamp indices to [0, cols - 1]
                    __m256i cols_minus_1 = _mm256_set1_epi32(input_continuous.cols - 1);
                    x_src_v = _mm256_min_epi32(x_src_v, cols_minus_1);
                    x_src_v = _mm256_max_epi32(x_src_v, _mm256_setzero_si256());

                    // Extract indices into an array
                    alignas(32) int x_src_array[8];
                    _mm256_store_si256(reinterpret_cast<__m256i *>(x_src_array), x_src_v);

                    // Gather pixel values
                    alignas(32) cv::Vec<T, 3> pixels[8];
                    for (int i = 0; i < 8; ++i)
                    {
                        pixels[i] = input_row[x_src_array[i]];
                    }

                    // Store the pixel values
                    for (int i = 0; i < 8; ++i)
                    {
                        output_row[x + i] = pixels[i];
                    }
                }

                // Handle remaining pixels
                for (; x < output_width; ++x)
                {
                    int x_src = x_mapping[x];
                    output_row[x] = input_row[x_src];
                }
            }
        }
        else
        {
            // Handle other channel numbers (generic)
#pragma omp parallel for schedule(dynamic)
            for (int y_dst = 0; y_dst < output_height; ++y_dst)
            {
                int y_src = y_mapping[y_dst];
                const T *input_row = input_continuous.ptr<T>(y_src);
                T *output_row = output.ptr<T>(y_dst);
                int x = 0;

                for (; x < output_width; ++x)
                {
                    int x_src = x_mapping[x];
                    for (int c = 0; c < channels; ++c)
                    {
                        output_row[x * channels + c] = input_row[x_src * channels + c];
                    }
                }
            }
        }

        return output;
    }

    // Explicit template instantiation
    template Mat resizeNN_AVX<uchar>(const Mat &, int, int, int);
    template Mat resizeNN_AVX<float>(const Mat &, int, int, int);

    // Multi-threaded Bilinear Interpolation with AVX Optimization
    template <typename T>
    Mat resizeBilinear_AVX(const Mat &input, int output_width, int output_height, int channels)
    {
        // Ensure the input matrix is continuous for optimized memory access
        Mat input_continuous = input.isContinuous() ? input : input.clone();
        Mat output(output_height, output_width, input_continuous.type());

        // Calculate scaling factors
        double scale_width = static_cast<double>(output_width) / input_continuous.cols;
        double scale_height = static_cast<double>(output_height) / input_continuous.rows;

        // Precompute y_src mappings
        std::vector<int> y1_mapping(output_height);
        std::vector<int> y2_mapping(output_height);
        std::vector<double> dy_mapping(output_height);

#pragma omp parallel for schedule(dynamic)
        for (int y_dst = 0; y_dst < output_height; ++y_dst)
        {
            double y_src_f = (y_dst + 0.5) / scale_height - 0.5;
            int y1 = static_cast<int>(floor(y_src_f));
            int y2 = y1 + 1;
            double dy = y_src_f - y1;
            // Clamp y coordinates
            y1 = std::max(0, std::min(y1, input_continuous.rows - 1));
            y2 = std::max(0, std::min(y2, input_continuous.rows - 1));
            dy_mapping[y_dst] = dy;
            y1_mapping[y_dst] = y1;
            y2_mapping[y_dst] = y2;
        }

        // Precompute x_src mappings
        std::vector<int> x1_mapping(output_width);
        std::vector<int> x2_mapping(output_width);
        std::vector<double> dx_mapping(output_width);

#pragma omp parallel for schedule(dynamic)
        for (int x_dst = 0; x_dst < output_width; ++x_dst)
        {
            double x_src_f = (x_dst + 0.5) / scale_width - 0.5;
            int x1 = static_cast<int>(floor(x_src_f));
            int x2 = x1 + 1;
            double dx = x_src_f - x1;

            // Clamp x coordinates
            x1 = std::max(0, std::min(x1, input_continuous.cols - 1));
            x2 = std::max(0, std::min(x2, input_continuous.cols - 1));
            dx_mapping[x_dst] = dx;
            x1_mapping[x_dst] = x1;
            x2_mapping[x_dst] = x2;
        }

        // Process based on the number of channels
        if (channels == 1)
        {
            // Single-channel image (Grayscale)
#pragma omp parallel for schedule(dynamic)
            for (int y_dst = 0; y_dst < output_height; ++y_dst)
            {
                int y1 = y1_mapping[y_dst];
                int y2 = y2_mapping[y_dst];
                double dy = dy_mapping[y_dst];
                const T *row1 = input_continuous.ptr<T>(y1);
                const T *row2 = input_continuous.ptr<T>(y2);
                T *output_row = output.ptr<T>(y_dst);
                int x = 0;

                // Process 8 pixels at a time using AVX2
                for (; x <= output_width - 8; x += 8)
                {
                    // Load 8 x1 and x2 indices
                    __m256i x1_v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&x1_mapping[x]));
                    __m256i x2_v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&x2_mapping[x]));

                    // Clamp indices to [0, cols - 1]
                    __m256i cols_minus_1 = _mm256_set1_epi32(input_continuous.cols - 1);
                    x1_v = _mm256_min_epi32(x1_v, cols_minus_1);
                    x1_v = _mm256_max_epi32(x1_v, _mm256_setzero_si256());
                    x2_v = _mm256_min_epi32(x2_v, cols_minus_1);
                    x2_v = _mm256_max_epi32(x2_v, _mm256_setzero_si256());

                    // Extract indices into arrays
                    alignas(32) int x1_array[8];
                    alignas(32) int x2_array[8];
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(x1_array), x1_v);
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(x2_array), x2_v);

                    // Gather pixel values from row1 and row2
                    alignas(32) T pixels1[8];
                    alignas(32) T pixels2[8];
                    for (int i = 0; i < 8; ++i)
                    {
                        pixels1[i] = row1[x1_array[i]];
                        pixels2[i] = row2[x2_array[i]];
                    }

                    // Perform bilinear interpolation
                    alignas(32) double interp_vals[8];
                    for (int i = 0; i < 8; ++i)
                    {
                        double I11 = pixels1[i];
                        double I21 = pixels2[i];
                        double I12 = row2[x1_array[i]];
                        double I22 = row2[x2_array[i]];

                        interp_vals[i] = (1 - dx_mapping[x + i]) * (1 - dy) * I11 +
                                         dx_mapping[x + i] * (1 - dy) * I21 +
                                         (1 - dx_mapping[x + i]) * dy * I12 +
                                         dx_mapping[x + i] * dy * I22;
                    }

                    // Store the interpolated values
                    for (int i = 0; i < 8; ++i)
                    {
                        output_row[x + i] = clamp_and_round<T>(interp_vals[i]);
                    }
                }

                // Handle remaining pixels
                for (; x < output_width; ++x)
                {
                    int x1 = x1_mapping[x];
                    int x2 = x2_mapping[x];
                    double dx = dx_mapping[x];
                    double interpolated = (1 - dx) * (1 - dy) * row1[x1] +
                                          dx * (1 - dy) * row1[x2] +
                                          (1 - dx) * dy * row2[x1] +
                                          dx * dy * row2[x2];
                    output_row[x] = clamp_and_round<T>(interpolated);
                }
            }
        }
        else if (channels == 3)
        {
            // Three-channel image (Color)
#pragma omp parallel for schedule(dynamic)
            for (int y_dst = 0; y_dst < output_height; ++y_dst)
            {
                int y1 = y1_mapping[y_dst];
                int y2 = y2_mapping[y_dst];
                double dy = dy_mapping[y_dst];
                const cv::Vec<T, 3> *row1 = input_continuous.ptr<cv::Vec<T, 3>>(y1);
                const cv::Vec<T, 3> *row2 = input_continuous.ptr<cv::Vec<T, 3>>(y2);
                cv::Vec<T, 3> *output_row = output.ptr<cv::Vec<T, 3>>(y_dst);
                int x = 0;

                // Process 8 pixels at a time using AVX2
                for (; x <= output_width - 8; x += 8)
                {
                    // Load 8 x1 and x2 indices
                    __m256i x1_v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&x1_mapping[x]));
                    __m256i x2_v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&x2_mapping[x]));

                    // Clamp indices to [0, cols - 1]
                    __m256i cols_minus_1 = _mm256_set1_epi32(input_continuous.cols - 1);
                    x1_v = _mm256_min_epi32(x1_v, cols_minus_1);
                    x1_v = _mm256_max_epi32(x1_v, _mm256_setzero_si256());
                    x2_v = _mm256_min_epi32(x2_v, cols_minus_1);
                    x2_v = _mm256_max_epi32(x2_v, _mm256_setzero_si256());

                    // Extract indices into arrays
                    alignas(32) int x1_array[8];
                    alignas(32) int x2_array[8];
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(x1_array), x1_v);
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(x2_array), x2_v);

                    // Gather pixel values from row1 and row2
                    alignas(32) cv::Vec<T, 3> pixels1[8];
                    alignas(32) cv::Vec<T, 3> pixels2[8];
                    for (int i = 0; i < 8; ++i)
                    {
                        pixels1[i] = row1[x1_array[i]];
                        pixels2[i] = row2[x2_array[i]];
                    }

                    // Perform bilinear interpolation for each channel
                    alignas(32) cv::Vec<double, 3> interp_vals[8];
                    for (int i = 0; i < 8; ++i)
                    {
                        for (int c = 0; c < 3; ++c)
                        {
                            double I11 = pixels1[i][c];
                            double I21 = pixels2[i][c];
                            double I12 = row2[x1_array[i]][c];
                            double I22 = row2[x2_array[i]][c];

                            interp_vals[i][c] = (1 - dx_mapping[x + i]) * (1 - dy) * I11 +
                                                dx_mapping[x + i] * (1 - dy) * I21 +
                                                (1 - dx_mapping[x + i]) * dy * I12 +
                                                dx_mapping[x + i] * dy * I22;
                        }
                    }

                    // Store the interpolated values
                    for (int i = 0; i < 8; ++i)
                    {
                        output_row[x + i] = cv::Vec<T, 3>(
                            clamp_and_round<T>(interp_vals[i][0]),
                            clamp_and_round<T>(interp_vals[i][1]),
                            clamp_and_round<T>(interp_vals[i][2]));
                    }
                }

                // Handle remaining pixels
                for (; x < output_width; ++x)
                {
                    int x1 = x1_mapping[x];
                    int x2 = x2_mapping[x];
                    double dx = dx_mapping[x];
                    cv::Vec<double, 3> interpolated;
                    for (int c = 0; c < 3; ++c)
                    {
                        double I11 = row1[x1][c];
                        double I21 = row1[x2][c];
                        double I12 = row2[x1][c];
                        double I22 = row2[x2][c];

                        interpolated[c] = (1 - dx) * (1 - dy) * I11 +
                                          dx * (1 - dy) * I21 +
                                          (1 - dx) * dy * I12 +
                                          dx * dy * I22;
                    }
                    output_row[x] = cv::Vec<T, 3>(
                        clamp_and_round<T>(interpolated[0]),
                        clamp_and_round<T>(interpolated[1]),
                        clamp_and_round<T>(interpolated[2]));
                }
            }
        }
        else
        {
            // Handle other channel numbers (generic)
#pragma omp parallel for schedule(dynamic)
            for (int y_dst = 0; y_dst < output_height; ++y_dst)
            {
                int y_src = dy_mapping[y_dst];
                const T *input_row = input_continuous.ptr<T>(y_src);
                T *output_row = output.ptr<T>(y_dst);
                int x = 0;

                for (; x < output_width; ++x)
                {
                    int x_src = dx_mapping[x];
                    for (int c = 0; c < channels; ++c)
                    {
                        output_row[x * channels + c] = input_row[x_src * channels + c];
                    }
                }
            }
        }

        return output;
    }

    // Explicit template instantiation
    template Mat resizeBilinear_AVX<uchar>(const Mat &, int, int, int);
    template Mat resizeBilinear_AVX<float>(const Mat &, int, int, int);

    // Multi-threaded Nearest Neighbor Interpolation
    Mat resizeNN_MT(const Mat &input, int output_width, int output_height)
    {
        int channels = input.channels();
        int depth = input.depth();

        switch (depth)
        {
        case CV_8U:
            return resizeNN_AVX<uchar>(input, output_width, output_height, channels);
        case CV_16U:
            return resizeNN_AVX<ushort>(input, output_width, output_height, channels);
        case CV_32F:
            return resizeNN_AVX<float>(input, output_width, output_height, channels);
        default:
            throw std::runtime_error("Unsupported data type for resizeNN_MT");
        }
    }

    // Multi-threaded Bilinear Interpolation
    Mat resizeBilinear_MT(const Mat &input, int output_width, int output_height)
    {
        int channels = input.channels();
        int depth = input.depth();

        switch (depth)
        {
        case CV_8U:
            return resizeBilinear_AVX<uchar>(input, output_width, output_height, channels);
        case CV_16U:
            return resizeBilinear_AVX<ushort>(input, output_width, output_height, channels);
        case CV_32F:
            return resizeBilinear_AVX<float>(input, output_width, output_height, channels);
        default:
            throw std::runtime_error("Unsupported data type for resizeBilinear_MT");
        }
    }

    // Single-threaded Nearest Neighbor Interpolation Implementation
    template <typename T>
    Mat resizeNN_ST_impl(const Mat &input, int output_width, int output_height, int channels)
    {
        Mat output(output_height, output_width, input.type());

        double scale_width = static_cast<double>(output_width) / input.cols;
        double scale_height = static_cast<double>(output_height) / input.rows;

        for (int y_dst = 0; y_dst < output_height; ++y_dst)
        {
            // Map output y to input y
            int y_src = static_cast<int>(round(y_dst * (input.rows / static_cast<double>(output_height))));
            y_src = std::min(y_src, input.rows - 1);

            if (channels == 1)
            {
                const T *input_row = input.ptr<T>(y_src);
                T *output_row = output.ptr<T>(y_dst);
                for (int x_dst = 0; x_dst < output_width; ++x_dst)
                {
                    int x_src = static_cast<int>(round(x_dst * (input.cols / static_cast<double>(output_width))));
                    x_src = std::min(x_src, input.cols - 1);
                    output_row[x_dst] = input_row[x_src];
                }
            }
            else if (channels == 3)
            {
                const cv::Vec<T, 3> *input_row = input.ptr<cv::Vec<T, 3>>(y_src);
                cv::Vec<T, 3> *output_row = output.ptr<cv::Vec<T, 3>>(y_dst);
                for (int x_dst = 0; x_dst < output_width; ++x_dst)
                {
                    int x_src = static_cast<int>(round(x_dst * (input.cols / static_cast<double>(output_width))));
                    x_src = std::min(x_src, input.cols - 1);
                    output_row[x_dst] = input_row[x_src];
                }
            }
        }

        return output;
    }

    // Single-threaded Bilinear Interpolation Implementation
    template <typename T>
    Mat resizeBilinear_ST_impl(const Mat &input, int output_width, int output_height, int channels)
    {
        Mat output(output_height, output_width, input.type());

        double scale_width = static_cast<double>(output_width) / input.cols;
        double scale_height = static_cast<double>(output_height) / input.rows;

        for (int y_dst = 0; y_dst < output_height; ++y_dst)
        {
            // Map output y to input y
            double y_src_f = (y_dst + 0.5) / scale_height - 0.5;
            int y1 = static_cast<int>(floor(y_src_f));
            int y2 = y1 + 1;
            double dy = y_src_f - y1;

            // Clamp y coordinates
            y1 = std::max(0, std::min(y1, input.rows - 1));
            y2 = std::max(0, std::min(y2, input.rows - 1));

            for (int x_dst = 0; x_dst < output_width; ++x_dst)
            {
                // Map output x to input x
                double x_src_f = (x_dst + 0.5) / scale_width - 0.5;
                int x1 = static_cast<int>(floor(x_src_f));
                int x2 = x1 + 1;
                double dx = x_src_f - x1;

                // Clamp x coordinates
                x1 = std::max(0, std::min(x1, input.cols - 1));
                x2 = std::max(0, std::min(x2, input.cols - 1));

                if (channels == 1)
                {
                    const T *row1 = input.ptr<T>(y1);
                    const T *row2 = input.ptr<T>(y2);
                    double I11 = row1[x1];
                    double I21 = row1[x2];
                    double I12 = row2[x1];
                    double I22 = row2[x2];

                    // Bilinear interpolation
                    double val = (1 - dx) * (1 - dy) * I11 +
                                 dx * (1 - dy) * I21 +
                                 (1 - dx) * dy * I12 +
                                 dx * dy * I22;

                    output.at<T>(y_dst, x_dst) = clamp_and_round<T>(val);
                }
                else if (channels == 3)
                {
                    const cv::Vec<T, 3> *row1 = input.ptr<cv::Vec<T, 3>>(y1);
                    const cv::Vec<T, 3> *row2 = input.ptr<cv::Vec<T, 3>>(y2);
                    cv::Vec<T, 3> I11 = row1[x1];
                    cv::Vec<T, 3> I21 = row1[x2];
                    cv::Vec<T, 3> I12 = row2[x1];
                    cv::Vec<T, 3> I22 = row2[x2];

                    // Bilinear interpolation for each channel
                    cv::Vec<double, 3> val;
                    for (int c = 0; c < 3; ++c)
                    {
                        val[c] = (1 - dx) * (1 - dy) * I11[c] +
                                 dx * (1 - dy) * I21[c] +
                                 (1 - dx) * dy * I12[c] +
                                 dx * dy * I22[c];
                        val[c] = clamp_and_round<T>(val[c]);
                    }

                    output.at<cv::Vec<T, 3>>(y_dst, x_dst) = cv::Vec<T, 3>(
                        static_cast<T>(val[0]),
                        static_cast<T>(val[1]),
                        static_cast<T>(val[2]));
                }
            }
        }

        return output;
    }

    // Single-threaded Nearest Neighbor Implementation
    Mat resizeNN_ST(const Mat &input, int output_width, int output_height)
    {
        int channels = input.channels();
        int depth = input.depth();

        switch (depth)
        {
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

    // Single-threaded Bilinear Interpolation Implementation
    Mat resizeBilinear_ST(const Mat &input, int output_width, int output_height)
    {
        int channels = input.channels();
        int depth = input.depth();

        switch (depth)
        {
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

    // Implementation of custom resize function
    void resize_customfinal(const cv::Mat &input, cv::Mat &output, const cv::Size &new_size, InterpolationMethod method, bool MT)
    {
        if (MT)
        {
            if (method == NEAREST_NEIGHBOR)
            {
                output = resizeNN_MT(input, new_size.width, new_size.height);
            }
            else if (method == BILINEAR)
            {
                output = resizeBilinear_MT(input, new_size.width, new_size.height);
            }
            else
            {
                // Default to nearest neighbor
                output = resizeNN_MT(input, new_size.width, new_size.height);
            }
        }
        else
        {
            if (method == NEAREST_NEIGHBOR)
            {
                output = resizeNN_ST(input, new_size.width, new_size.height);
            }
            else if (method == BILINEAR)
            {
                output = resizeBilinear_ST(input, new_size.width, new_size.height);
            }
            else
            {
                // Default to nearest neighbor
                output = resizeNN_ST(input, new_size.width, new_size.height);
            }
        }
    }

}