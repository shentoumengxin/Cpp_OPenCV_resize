// src/main.cpp

// resize.cpp
#include "resize.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <algorithm>



// 函数用于测量执行时间（毫秒）
double measure_time(std::function<void()> func) {
    auto start = std::chrono::high_resolution_clock::now();
    func(); // 执行待测函数
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main(int argc, char** argv) {
    // 检查输入参数
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    // 读取输入图像
    cv::Mat input = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (input.empty()) {
        std::cerr << "Failed to load image: " << argv[1] << std::endl;
        return -1;
    }

    // 定义测试参数
    std::vector<cv::Size> test_sizes = {cv::Size(800, 600), cv::Size(400, 300), cv::Size(1600, 1200)};
    std::vector<double> scale_factors = {0.5, 1.5};
    std::vector<InterpolationMethod> methods = {NEAREST_NEIGHBOR, BILINEAR};

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
                std::cout << "New Size: " << new_size << ", Scale: " << scale << ", Method: "
                          << ((method == NEAREST_NEIGHBOR) ? "Nearest Neighbor" : "Bilinear") << std::endl;
                std::cout << "Custom Resize Time: " << custom_time << " ms" << std::endl;
                std::cout << "OpenCV Resize Time: " << opencv_time << " ms" << std::endl;

                // 计算PSNR以评估图像质量（仅对最近邻和双线性插值有效）
                double psnr = cv::PSNR(output_custom, output_opencv);
                std::cout << "PSNR between Custom and OpenCV: " << psnr << " dB" << std::endl;
                std::cout << "----------------------------------------" << std::endl;
            }
        }
    }

    // 显示原始图像和一个调整大小后的图像作为示例
    cv::Mat resized_example;
    resize_custom(input, resized_example, cv::Size(static_cast<int>(input.cols * 1.5), static_cast<int>(input.rows * 1.5)), NEAREST_NEIGHBOR);
    cv::imshow("Original Image", input);
    cv::imshow("Resized Image (Custom Nearest Neighbor)", resized_example);
    cv::waitKey(0);

    return 0;
}
