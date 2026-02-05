#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"
#include "common.h"
#include "buffers.h"
#include "utils/preprocess.h"
#include "utils/postprocess_cuda.h"
#include "utils/types.h"

// 加载模型文件
std::vector<unsigned char> load_engine_file(const std::string &file_name)
{
    std::vector<unsigned char> engine_data;
    std::ifstream engine_file(file_name, std::ios::binary);
    assert(engine_file.is_open() && "Unable to load engine file.");
    engine_file.seekg(0, engine_file.end);  // 将文件指针移动到文件末尾
    int length = engine_file.tellg();
    engine_data.resize(length);
    engine_file.seekg(0, engine_file.beg);  // 将文件指针移动到文件开头
    engine_file.read(reinterpret_cast<char *>(engine_data.data()), length);
    return engine_data;
}

// 获取binding的形状信息
std::vector<int> get_binding_dims(nvinfer1::IExecutionContext* context, int binding_idx)
{
    auto dims = context->getBindingDimensions(binding_idx);
    std::vector<int> shape;
    for (int i = 0; i < dims.nbDims; i++) {
        shape.push_back(dims.d[i]);
    }
    return shape;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "用法: " << argv[0] << " <engine_file> <image_path> [output_path]" << std::endl;
        std::cerr << "  engine_file: TensorRT引擎文件路径" << std::endl;
        std::cerr << "  image_path:  输入图片路径" << std::endl;
        std::cerr << "  output_path: 输出路径 (可选，默认./output/result.jpg)" << std::endl;
        return -1;
    }

    auto engine_file = argv[1];                                                // 模型文件
    auto image_path = argv[2];                                                 // 输入图片路径
    std::string output_path = (argc >= 4) ? argv[3] : "./output/result.jpg";  // 输出路径

    // ========= 1. 创建推理运行时runtime =========
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!runtime)
    {
        std::cout << "runtime create failed" << std::endl;
        return -1;
    }
    // ======== 2. 反序列化生成engine =========
    // 加载模型文件
    auto plan = load_engine_file(engine_file);
    // 反序列化生成engine
    auto mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan.data(), plan.size()));
    if (!mEngine)
    {
        return -1;
    }

    // ======== 3. 创建执行上下文context =========
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        std::cout << "context create failed" << std::endl;
        return -1;
    }

    // ========== 4. 创建输入输出缓冲区 =========
    samplesCommon::BufferManager buffers(mEngine);
    
    // 打印binding信息
    int n_io = mEngine->getNbBindings();
    std::cout << "Engine has " << n_io << " bindings:" << std::endl;
    for (int i = 0; i < n_io; i++) {
        auto dims = context->getBindingDimensions(i);
        std::string dims_str = "[";
        for (int j = 0; j < dims.nbDims; j++) {
            dims_str += std::to_string(dims.d[j]);
            if (j < dims.nbDims - 1) dims_str += ", ";
        }
        dims_str += "]";
        std::cout << "  Binding " << i << ": " << mEngine->getBindingName(i) 
                  << " shape=" << dims_str
                  << (mEngine->bindingIsInput(i) ? " (input)" : " (output)") << std::endl;
    }

    // ========== 5. 读取输入图片 =========
    std::cout << "\n读取图片: " << image_path << std::endl;
    cv::Mat image = cv::imread(image_path);
    if (image.empty())
    {
        std::cerr << "错误: 无法读取图片 " << image_path << std::endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    std::cout << "图片尺寸: " << width << "x" << height << std::endl;

    // 申请CUDA内存用于预处理
    int img_size = width * height;
    cuda_preprocess_init(img_size);

    // ========== 6. 连续推理10次 =========
    const int num_iterations = 10;
    std::vector<float> preprocess_times;
    std::vector<float> inference_times;
    std::vector<float> postprocess_times;
    std::vector<SegmentationResult> seg_results;
    
    std::cout << "\n开始连续推理 " << num_iterations << " 次...\n" << std::endl;
    
    for (int iter = 0; iter < num_iterations; iter++) {
        std::cout << "========== 第 " << (iter + 1) << " 次推理 ==========" << std::endl;
        
        // GPU 预处理
        auto preprocess_start = std::chrono::high_resolution_clock::now();
        process_input(image, (float *)buffers.getDeviceBuffer(kInputTensorName));
        auto preprocess_end = std::chrono::high_resolution_clock::now();
        float preprocess_time = std::chrono::duration_cast<std::chrono::microseconds>(preprocess_end - preprocess_start).count() / 1000.f;
        preprocess_times.push_back(preprocess_time);
        
        // 执行推理
        auto inference_start = std::chrono::high_resolution_clock::now();
        context->executeV2(buffers.getDeviceBindings().data());
        buffers.copyOutputToHost();
        auto inference_end = std::chrono::high_resolution_clock::now();
        float inference_time = std::chrono::duration_cast<std::chrono::microseconds>(inference_end - inference_start).count() / 1000.f;
        inference_times.push_back(inference_time);
        
        // GPU 后处理 - YOLOv26 End2End
        auto postprocess_start = std::chrono::high_resolution_clock::now();
        float *output0 = (float *)buffers.getHostBuffer(kOutput0TensorName);
        float *output1 = (float *)buffers.getHostBuffer(kOutput1TensorName);
        
        // 获取输出shape
        auto output0_shape = get_binding_dims(context.get(), mEngine->getBindingIndex(kOutput0TensorName));
        auto output1_shape = get_binding_dims(context.get(), mEngine->getBindingIndex(kOutput1TensorName));
        
        // 计算dim0和dim1 (假设输出是2D或去除batch后的2D)
        int output0_dim0 = 1, output0_dim1 = 1;
        int output1_dim0 = 1, output1_dim1 = 1;
        
        if (output0_shape.size() >= 2) {
            // 可能是 [batch, N, 38] 或 [batch, 32, 160, 160] 或 [N, 38] 或 [32, 160, 160]
            if (output0_shape.size() == 2) {
                output0_dim0 = output0_shape[0];
                output0_dim1 = output0_shape[1];
            } else if (output0_shape.size() == 3) {
                output0_dim0 = output0_shape[1];  // 跳过batch
                output0_dim1 = output0_shape[2];
            } else if (output0_shape.size() == 4) {
                // [batch, 32, 160, 160] -> dim0=32, dim1=160*160
                output0_dim0 = output0_shape[1];
                output0_dim1 = output0_shape[2] * output0_shape[3];
            }
        }
        
        if (output1_shape.size() >= 2) {
            if (output1_shape.size() == 2) {
                output1_dim0 = output1_shape[0];
                output1_dim1 = output1_shape[1];
            } else if (output1_shape.size() == 3) {
                output1_dim0 = output1_shape[1];
                output1_dim1 = output1_shape[2];
            } else if (output1_shape.size() == 4) {
                output1_dim0 = output1_shape[1];
                output1_dim1 = output1_shape[2] * output1_shape[3];
            }
        }
        
        seg_results.clear();
        process_yolov26_e2e_segmentation(seg_results, output0, output1,
                                         output0_dim0, output0_dim1,
                                         output1_dim0, output1_dim1,
                                         height, width, kConfThresh);
        auto postprocess_end = std::chrono::high_resolution_clock::now();
        float postprocess_time = std::chrono::duration_cast<std::chrono::microseconds>(postprocess_end - postprocess_start).count() / 1000.f;
        postprocess_times.push_back(postprocess_time);
        
        // 显示本次耗时
        float total = preprocess_time + inference_time + postprocess_time;
        std::cout << "预处理: " << preprocess_time << " ms, "
                  << "推理: " << inference_time << " ms, "
                  << "后处理: " << postprocess_time << " ms, "
                  << "总计: " << total << " ms" << std::endl;
    }
    
    // 计算统计数据
    auto calc_avg = [](const std::vector<float>& times) -> float {
        float sum = 0;
        for (float t : times) sum += t;
        return sum / times.size();
    };
    
    auto calc_min = [](const std::vector<float>& times) -> float {
        float min_val = times[0];
        for (float t : times) if (t < min_val) min_val = t;
        return min_val;
    };
    
    auto calc_max = [](const std::vector<float>& times) -> float {
        float max_val = times[0];
        for (float t : times) if (t > max_val) max_val = t;
        return max_val;
    };
    
    auto calc_avg_without_first = [](const std::vector<float>& times) -> float {
        float sum = 0;
        for (size_t i = 1; i < times.size(); i++) sum += times[i];
        return sum / (times.size() - 1);
    };
    
    float preprocess_avg = calc_avg(preprocess_times);
    float preprocess_min = calc_min(preprocess_times);
    float preprocess_max = calc_max(preprocess_times);
    float preprocess_first = preprocess_times[0];
    float preprocess_avg_rest = calc_avg_without_first(preprocess_times);
    
    float inference_avg = calc_avg(inference_times);
    float inference_min = calc_min(inference_times);
    float inference_max = calc_max(inference_times);
    float inference_first = inference_times[0];
    float inference_avg_rest = calc_avg_without_first(inference_times);
    
    float postprocess_avg = calc_avg(postprocess_times);
    float postprocess_min = calc_min(postprocess_times);
    float postprocess_max = calc_max(postprocess_times);
    float postprocess_first = postprocess_times[0];
    float postprocess_avg_rest = calc_avg_without_first(postprocess_times);
    
    std::cout << "\n========== 性能统计汇总 ==========" << std::endl;
    std::cout << "检测到对象数: " << seg_results.size() << std::endl;
    std::cout << "\n预处理时间:" << std::endl;
    std::cout << "  第1次:   " << preprocess_first << " ms" << std::endl;
    std::cout << "  平均值:  " << preprocess_avg << " ms" << std::endl;
    std::cout << "  2-10次:  " << preprocess_avg_rest << " ms" << std::endl;
    std::cout << "  最小值:  " << preprocess_min << " ms" << std::endl;
    std::cout << "  最大值:  " << preprocess_max << " ms" << std::endl;
    
    std::cout << "\n推理时间:" << std::endl;
    std::cout << "  第1次:   " << inference_first << " ms" << std::endl;
    std::cout << "  平均值:  " << inference_avg << " ms" << std::endl;
    std::cout << "  2-10次:  " << inference_avg_rest << " ms" << std::endl;
    std::cout << "  最小值:  " << inference_min << " ms" << std::endl;
    std::cout << "  最大值:  " << inference_max << " ms" << std::endl;
    
    std::cout << "\n后处理时间:" << std::endl;
    std::cout << "  第1次:   " << postprocess_first << " ms" << std::endl;
    std::cout << "  平均值:  " << postprocess_avg << " ms" << std::endl;
    std::cout << "  2-10次:  " << postprocess_avg_rest << " ms" << std::endl;
    std::cout << "  最小值:  " << postprocess_min << " ms" << std::endl;
    std::cout << "  最大值:  " << postprocess_max << " ms" << std::endl;
    
    float total_avg = preprocess_avg + inference_avg + postprocess_avg;
    float total_avg_rest = preprocess_avg_rest + inference_avg_rest + postprocess_avg_rest;
    std::cout << "\n总时间 (不含绘图):" << std::endl;
    std::cout << "  平均值:  " << total_avg << " ms" << std::endl;
    std::cout << "  2-10次:  " << total_avg_rest << " ms" << std::endl;
    std::cout << "  稳定FPS: " << 1000.0f / total_avg_rest << std::endl;
    std::cout << "==============================\n" << std::endl;
    
    // 使用最后一次的结果进行绘图
    float preprocess_time = preprocess_times.back();
    float inference_time = inference_times.back();
    float postprocess_time = postprocess_times.back();
    
    // ========== 9. 绘制结果 =========
    auto draw_start = std::chrono::high_resolution_clock::now();
    draw_segmentation_results(image, seg_results);
    auto draw_end = std::chrono::high_resolution_clock::now();
    float draw_time = std::chrono::duration_cast<std::chrono::microseconds>(draw_end - draw_start).count() / 1000.f;
    
    std::cout << "\n绘图时间: " << draw_time << " ms" << std::endl;

    // ========== 10. 添加统计信息到图片 =========
    std::string time_str = "Avg: " + std::to_string((int)total_avg_rest) + "ms";
    std::string infer_str = "Infer: " + std::to_string((int)inference_avg_rest) + "ms";
    std::string det_count = "Objects: " + std::to_string(seg_results.size());
    cv::putText(image, time_str, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    cv::putText(image, infer_str, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    cv::putText(image, det_count, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

    // ========== 11. 保存结果 =========
    if (cv::imwrite(output_path, image))
    {
        std::cout << "结果已保存到: " << output_path << std::endl;
    }
    else
    {
        std::cerr << "错误: 无法保存结果到 " << output_path << std::endl;
        return -1;
    }

    // 释放CUDA内存
    cuda_preprocess_destroy();
    
    // ========== 12. 释放资源 =========
    // 因为使用了unique_ptr，所以不需要手动释放
    
    std::cout << "推理完成!" << std::endl;
    return 0;
}
