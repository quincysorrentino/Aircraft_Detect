#include "detector.h"

Detector::Detector(const std::wstring &modelPath)
{
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "AeroLog");
    Ort::SessionOptions options;

    // 1. Multi-threading: Use all available logical cores
    options.SetIntraOpNumThreads(std::thread::hardware_concurrency());

    // 2. Optimization Level: Enable all hardware-specific math optimizations
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 3. Execution Mode: Sequential is usually faster for single-model CPU inference
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    session = Ort::Session(env, modelPath.c_str(), options);
}

std::vector<Detection> Detector::run(cv::Mat &frame)
{
    // Pre-process
    // add padding to images for 640x640 input size

    float scale = std::min(640.0f / frame.cols, 640.0f / frame.rows);
    int nw = frame.cols * scale;
    int nh = frame.rows * scale;
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(nw, nh));

    // create 640x640 canvas to paste image onto
    cv::Mat canvas(640, 640, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(0, 0, nw, nh)));

    // convert bgr (cv) to rgb (yolo)
    cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);

    // convert hwc to chw - OPTIMIZED with pointer access
    std::vector<float> inputTensorValues(3 * 640 * 640);

    // Use pointer access for 3-5x speedup over .at<>()
    const int stride = 640;
    for (int h = 0; h < 640; ++h)
    {
        const uchar *row_ptr = canvas.ptr<uchar>(h);
        for (int w = 0; w < 640; ++w)
        {
            int pixel_idx = w * 3;
            // BGR -> RGB and normalize in one pass
            inputTensorValues[0 * 640 * 640 + h * 640 + w] = row_ptr[pixel_idx + 0] / 255.0f; // R
            inputTensorValues[1 * 640 * 640 + h * 640 + w] = row_ptr[pixel_idx + 1] / 255.0f; // G
            inputTensorValues[2 * 640 * 640 + h * 640 + w] = row_ptr[pixel_idx + 2] / 255.0f; // B
        }
    }

    // Inference
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 4> inputShape = {1, 3, 640, 640};

    // convert vector into tensor
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memory_info, inputTensorValues.data(), inputTensorValues.size(),
        inputShape.data(), inputShape.size());

    const char *inputNames[] = {"images"};
    const char *outputNames[] = {"output0"};

    // outputs 8,400 boxes
    auto outputTensors = session.Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);
    float *rawOutput = outputTensors[0].GetTensorMutableData<float>();

    // Post-process
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    for (int i = 0; i < 8400; ++i)
    {
        float *classes_scores = rawOutput + 4 * 8400 + i;
        float max_score = 0;
        int class_id = 0;

        // Find which plane type has the highest score for this box.
        for (int cls = 0; cls < 60; cls++)
        {
            float score = *(classes_scores + cls * 8400);
            if (score > max_score)
            {
                max_score = score;
                class_id = cls;
            }
        }

        // Only keep detections with high confidence (e.g., > 50%).
        if (max_score > 0.50f)
        {
            float cx = rawOutput[0 * 8400 + i]; // Center X
            float cy = rawOutput[1 * 8400 + i]; // Center Y
            float w = rawOutput[2 * 8400 + i];  // Width
            float h = rawOutput[3 * 8400 + i];  // Height

            // Convert YOLO coordinates back to original pixel coordinates.
            int left = (cx - w / 2) / scale;
            int top = (cy - h / 2) / scale;
            boxes.push_back(cv::Rect(left, top, w / scale, h / scale));
            confs.push_back(max_score);
            classIds.push_back(class_id);
        }
    }

    // NMS: If AI detects the same plane twice, only keep the best box.
    //
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, 0.4f, 0.5f, indices);

    std::vector<Detection> final_results;
    for (int idx : indices)
    {
        final_results.push_back({boxes[idx], confs[idx], classIds[idx]});
    }

    return final_results;
}