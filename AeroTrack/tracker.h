#pragma once
#include <opencv2/video/tracking.hpp>

class GhostTracker
{
public:
    GhostTracker();
    void initialize(cv::Rect firstDetection);
    cv::Rect predict();
    void update(cv::Rect detectedBox, float confidence = 1.0f);
    cv::Rect getPrediction() const;
    void incrementLostFrames();
    void resetLostFrames();
    int getLostFrames() const;
    bool isInitialized() const;

private:
    cv::KalmanFilter kf;
    int framesLost;
    bool initialized;
};