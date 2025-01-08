# OpenCV-based Resize Function Implementation and Optimization

**Team Members:** 12311124 Zhang Zihan 12311624 Lin Pinhao 12312210 Liu Ruihan 
**Email:** [12311124@mail.sustech.edu.cn](mailto:12332469@mail.sustech.edu.cn)  

---

## Project Introduction

Image resizing is a fundamental operation in computer vision, essential for tasks such as data preprocessing, thumbnail generation, and multi-resolution analysis. While OpenCV provides highly optimized built-in resize functions, developing a custom resize implementation offers valuable insights into the underlying interpolation methods and performance optimization techniques. This project focuses on creating a high-performance resize function supporting both nearest neighbor and bilinear interpolation methods, enhanced through multi-threading and SIMD (Single Instruction, Multiple Data) optimizations.

---

## 1. Implemented Functionalities

### 1.1. Nearest Neighbor and Bilinear Interpolation Methods

- **Nearest Neighbor Interpolation:**  
  Assigns each pixel in the output image the value of the closest corresponding pixel in the input image. This method is straightforward and fast but may produce blocky images when upscaling.

- **Bilinear Interpolation:**  
  Calculates the output pixel value as a weighted average of the four nearest pixels in the input image, resulting in smoother and higher-quality resized images compared to nearest neighbor interpolation.

   Both interpolation methods are implemented in single-threaded and multi-threaded versions to balance performance and simplicity. And each method supports scaling at **any arbitrary ratio**.

   To prevent out-of-bounds access during resizing, the implementation ensures that all calculated source coordinates (`x_src`, `y_src`) are clamped within the valid range of the input image dimensions. This guarantees that every pixel mapping remains valid, maintaining image integrity.

### 1.3. Multi-channel Support

The resize functions support both single-channel (grayscale) and three-channel (color) images. This versatility allows the functions to handle a wide range of image types commonly used in computer vision applications.

### 1.4. Support for Multiple Data Types

Beyond the standard unsigned 8-bit integers (`CV_8U`), the implementation extends support to:
- **16-bit Unsigned Integers (`CV_16U`)**
- **32-bit Floating Points (`CV_32F`)**

This is achieved through C++ template programming, enabling seamless handling of different data types without code duplication. The program will **automatically** check the depth of this image. 

### 1.5. Usage Instructions

  This CMake configuration sets up a C++17 project with OpenCV dependencies, enables performance optimizations through AVX2 and FMA instruction sets (for supported compilers), and ensures that the necessary OpenCV libraries are linked. It also defines the source files and compiles them into an executable. Please prepare your [OpenCV]((https://opencv.org/)) environment at first.

```cmd
~/opencv_resize_custom/build:cmake ..
~/opencv_resize_custom/build:make
~/opencv_resize_custom/build:./custom_resize
```



To utilize the custom resize function in your project, follow these steps:

1. **Include the Resize Module:**  
   Ensure that the `resize.h` header file is included in your project and that the corresponding `resize.cpp` is compiled alongside your application.

2. **Prepare the Input Image:**  
   Load the input image using OpenCV's `imread` function. The image can be single-channel or multi-channel and of supported data types.

3. **Define the Output Size and Interpolation Method:**
   Specify the desired output dimensions and choose the interpolation method (`NEAREST_NEIGHBOR` or `BILINEAR`).**Save or Display the Resized Image:**
   Use OpenCV's `imwrite` or `imshow` functions to save or display the resized image.

   ```cpp
   cv::Mat input = cv::imread("path_to_image.jpg", cv::IMREAD_UNCHANGED);
   cv::Size new_size(800, 600);                   //output size
   InterpolationMethod method = NEAREST_NEIGHBOR; // or BILINEAR
   cv::Mat output;                             //output mat
   bool use_multithreading = true; //for single-threaded set false, default:true
   resize_custom(input, output, new_size, method, use_multithreading);
   cv::imwrite("resized_image.jpg", output);
   // or
   cv::imshow("Resized Image", output);



------

## 2. Comparison Between Single-threaded and Multi-threaded Implementations

### 2.1. Performance Enhancements

- **Single-threaded Implementation:**
  Processes each pixel sequentially, resulting in straightforward but potentially slower execution times, especially for large images.
- **Multi-threaded Implementation:**
  Utilizes OpenCV's `parallel_for_` in combination with AVX SIMD instructions to process multiple pixels concurrently across different CPU cores. This parallelism significantly reduces execution time, making the resize operations more efficient.

### 2.2. Accuracy Assessment

- **Nearest Neighbor Interpolation:**
  Both single-threaded and multi-threaded implementations achieve exact pixel value matches with OpenCV's native `resize` function, resulting in perfect PSNR values (`∞ dB`).
- **Bilinear Interpolation:**
  The single-threaded implementation closely matches OpenCV's results with high PSNR values. The multi-threaded version maintains high accuracy, with minor discrepancies potentially arising from floating-point precision differences inherent in parallel processing.

### 2.3. Resource Utilization and Scalability

- **Single-threaded:**
  Limited by the processing speed of a single CPU core.
- **Multi-threaded:**
  Scales effectively with the number of available CPU cores, utilizing system resources more efficiently and handling larger images with improved performance.

------

## 3. Optimization Strategies

### 3.1. SIMD Optimization with AVX

Leveraging AVX SIMD instructions allows the processing of multiple pixels simultaneously within a single CPU instruction. This data-level parallelism accelerates computationally intensive tasks like interpolation by handling eight pixels at a time.

### 3.2. Advanced Multi-threading with OpenCV's parallel_for_

By integrating OpenCV's `parallel_for_` with SIMD optimizations, the implementation achieves thread-level parallelism alongside data-level parallelism. This dual approach maximizes performance gains by fully utilizing multi-core CPU architectures.

### 3.3. Precomputation of Pixel Mappings

Precomputing the mappings from output pixels to input pixels (`x_mapping` and `y_mapping`) reduces redundant calculations within the resize loops. This optimization minimizes computational overhead and enhances cache performance by accessing precomputed values stored in contiguous memory.

### 3.4. Ensuring Input Matrix Continuity

Ensuring that the input matrix is stored contiguously in memory (`input.isContinuous()`) optimizes memory access patterns, leading to faster data retrieval and processing during resizing operations.

------

### 4. Compare with original resize

#### 1. Our weakness

- **Lacks optimizations**: The custom `resize` implementation is likely to be slower due to the absence of advanced optimizations like multi-threading or SIMD instructions.
- **No hardware acceleration**: The current solution does not leverage GPU acceleration or other hardware-specific features, leading to potentially higher processing times for large images.
- **Scalability issues**: As the image size increases, the custom method may face performance bottlenecks, especially when scaling up or down by large factors.

- **Limited interpolation methods**: The custom implementation might support only basic interpolation techniques, such as linear or nearest-neighbor interpolation, limiting its flexibility.

- **Basic performance**: More advanced algorithms (like bicubic or Lanczos) are not present in the custom function, which may lead to lower-quality resized images, especially when upscaling or downscaling by significant amounts.

- **Platform-dependent**: The custom function might not be as robust or portable across different platforms, and it could require additional adjustments or optimizations for different operating systems.

- **Potential for bugs**: As the custom solution is not as widely tested as OpenCV, it might be more prone to bugs or inconsistencies when used in diverse environments.

- 

#### 2. **Optimize for Performance**

- **Multi-threading**: Implementing multi-threading can significantly improve the performance of the resizing operation, particularly when processing large images.
- **SIMD Support**: Adding SIMD (Single Instruction, Multiple Data) instructions, such as AVX2 or SSE, would optimize the resizing process by processing multiple pixels in parallel, resulting in faster execution.
- **GPU Acceleration**: Leveraging GPU libraries like CUDA or OpenCL could drastically speed up the resizing operation, especially when dealing with large images or videos.

- To ensure broader compatibility, we can refactor the code to make it more cross-platform. This may involve using platform-agnostic libraries or adding conditional code to handle different environments (e.g., Windows, Linux, macOS).

- Improving how edge cases (e.g., very small images, non-square aspect ratios) are handled would enhance the robustness of the custom function.
- Ensuring that the resized images maintain good visual quality without unexpected artifacts (such as pixelation or blurring) in edge regions would improve the overall user experience.

## 4. Conclusion

This project successfully developed a custom image resize function inspired by OpenCV's native `resize` function. By implementing both nearest neighbor and bilinear interpolation methods and enhancing them with multi-threading and SIMD optimizations, the custom resize function achieves high performance and accuracy. The support for multiple data types and multi-channel images broadens its applicability across various computer vision tasks. Comparative analysis demonstrates that while the custom implementation performs admirably, OpenCV's native functions still hold advantages in terms of extensive optimizations and broader feature support. Nonetheless, the custom resize function serves as an excellent educational tool for understanding image processing fundamentals and optimization techniques.

------

## References

- **OpenCV Documentation:** https://docs.opencv.org/
- **Bilinear Interpolation:** https://en.wikipedia.org/wiki/Bilinear_interpolation
- **SIMD Optimization:** https://en.wikipedia.org/wiki/SIMD
- [OpenCV图像缩放resize各种插值方式的比较实现 / 张生荣](https://www.zhangshengrong.com/p/8AaYmRQGa2/)



**Explanation:**

1. **Loading the Image:**
   The input image is loaded using OpenCV's `imread` function. Ensure that the image path is correct and that the image is successfully loaded.
2. **Defining Output Size:**
   Specify the desired dimensions for the resized image. In this example, the output size is set to 800x600 pixels.
3. **Choosing Interpolation Method:**
   Select between `NEAREST_NEIGHBOR` and `BILINEAR` interpolation methods based on the quality and performance requirements.
4. **Performing Resize Operation:**
   Call the `resize_custom` function, passing in the input image, an output image container,