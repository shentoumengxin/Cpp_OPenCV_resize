// src/main.cpp

#include "resize.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <functional>

// 仅使用 std 命名空间
using namespace std;

// 不使用 cv 命名空间，显式使用 cv:: 前缀

// 实现 PSNR 计算函数
double PSNR(const cv::Mat& I1, const cv::Mat& I2) {
    cv::Mat s1;
    cv::absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);      // 转换为 float
    s1 = s1.mul(s1);               // |I1 - I2|^2

    cv::Scalar s = cv::sum(s1);    // 计算所有元素的和

    double sse = s.val[0] + s.val[1] + s.val[2];
    if( sse <= 1e-10) // 如果两张图像完全相同
        return 0;
    else {
        double mse = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}

// 函数用于测量执行时间（毫秒）
double measure_time(function<void()> func) {
    auto start = chrono::high_resolution_clock::now();
    func(); // 执行待测函数
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double, milli>(end - start).count();
}

int main(int argc, char** argv) {
    // 检查输入参数
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <image_path>" << endl;
        return -1;
    }

    // 读取输入图像
    cv::Mat input = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (input.empty()) {
        cerr << "Failed to load image: " << argv[1] << endl;
        return -1;
    }

    // 定义测试参数
    vector<cv::Size> test_sizes = {cv::Size(800, 600), cv::Size(400, 300), cv::Size(1600, 1200)};
    vector<double> scale_factors = {0.5, 1.5};
    vector<InterpolationMethod> methods = {NEAREST_NEIGHBOR, BILINEAR};

    // 遍历测试参数
    for (const auto& size : test_sizes) {
        for (double scale : scale_factors) {
            cv::Size new_size(static_cast<int>(size.width * scale), static_cast<int>(size.height * scale));

            for (const auto& method : methods) {
                // 自定义Resize
                cv::Mat output_custom;
                double custom_time = measure_time([&]() {
                    resize_custom(input, output_custom, new_size, method);
                });

                // OpenCV的Resize
                cv::Mat output_opencv;
                double opencv_time = measure_time([&]() {
                    int interp_flag = (method == NEAREST_NEIGHBOR) ? cv::INTER_NEAREST : cv::INTER_LINEAR;
                    cv::resize(input, output_opencv, new_size, 0, 0, interp_flag);
                });

                // 输出结果
                cout << "New Size: [" << new_size.width << "x" << new_size.height << "], Scale: " << scale << ", Method: "
                     << ((method == NEAREST_NEIGHBOR) ? "Nearest Neighbor" : "Bilinear") << endl;
                cout << "Custom Resize Time: " << custom_time << " ms" << endl;
                cout << "OpenCV Resize Time: " << opencv_time << " ms" << endl;

                // 计算PSNR以评估图像质量（仅对最近邻和双线性插值有效）
                double psnr = PSNR(output_custom, output_opencv);
                cout << "PSNR between Custom and OpenCV: " << psnr << " dB" << endl;
                cout << "----------------------------------------" << endl;

                // 动态生成文件名并保存自定义缩放后的图像
                string method_str = (method == NEAREST_NEIGHBOR) ? "nearest_neighbor" : "bilinear";
                string filename_custom = "resized_custom_" + method_str + "_" + to_string(new_size.width) + "x" + to_string(new_size.height) + ".jpg";
                bool success_custom = cv::imwrite("../output/"+filename_custom, output_custom);
                if (!success_custom) {
                    cerr << "Failed to save resized image: " << filename_custom << endl;
                } else {
                    cout << "Resized image saved as " << filename_custom << endl;
                }

                // 动态生成文件名并保存 OpenCV 缩放后的图像
                string filename_opencv = "resized_opencv_" + method_str + "_" + to_string(new_size.width) + "x" + to_string(new_size.height) + ".jpg";
                bool success_opencv = cv::imwrite("../output/"+filename_opencv, output_opencv);
                if (!success_opencv) {
                    cerr << "Failed to save OpenCV resized image: " << filename_opencv << endl;
                } else {
                    cout << "OpenCV resized image saved as " << filename_opencv << endl;
                }
            }
        }
    }

    // 保存原始图像和一个调整大小后的图像作为示例
    cv::Mat resized_example;
    resize_custom(input, resized_example, cv::Size(static_cast<int>(input.cols * 1.5), static_cast<int>(input.rows * 1.5)), NEAREST_NEIGHBOR);

    // 保存原始图像
    bool success_original = cv::imwrite("original_image.jpg", input);
    if (!success_original) {
        cerr << "Failed to save original image." << endl;
    } else {
        cout << "Original image saved as original_image.jpg" << endl;
    }

    // 保存调整大小后的图像
    bool success_resized = cv::imwrite("resized_custom_nearest_neighbor.jpg", resized_example);
    if (!success_resized) {
        cerr << "Failed to save resized image." << endl;
    } else {
        cout << "Resized image saved as resized_custom_nearest_neighbor.jpg" << endl;
    }

    return 0;
}
