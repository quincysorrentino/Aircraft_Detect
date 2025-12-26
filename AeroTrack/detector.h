#pragma once
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>

// holds what model found
struct Detection
{
    cv::Rect box; // The [x, y, width, height] of the plane
    float confidence;
    int classID;
};

class Detector
{
public:
    Detector(const std::wstring &modelPath);
    std::vector<Detection> run(cv::Mat &frame);

private:
    Ort::Env env;
    Ort::Session session{nullptr};
    const cv::Size inputSize = {640, 640};
};