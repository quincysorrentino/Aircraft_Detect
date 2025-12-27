#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <map>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <deque>
#include <opencv2/video.hpp>
#include <opencv2/imgproc.hpp>
#include "detector.h"
#include "tracker.h"

cv::Rect clampToFrame(const cv::Rect &box, int maxWidth, int maxHeight);
cv::Mat computeHSVHist(const cv::Mat &frame, const cv::Rect &box);

enum class TrackState
{
    Tracked,
    Lost,
    Removed
};

// Lightweight track record with velocity + Kalman motion
struct Track
{
    int id;
    cv::Rect lastBox;
    cv::Point2f velocity;
    int lastSeenFrame;
    int classID;
    std::string className;
    float lastConfidence;
    float bestConfidence;
    int bestClassID;
    int totalHits;
    cv::Mat appearanceHist;
    std::vector<cv::Point2f> keypoints;
    std::deque<int> classHistory;
    std::deque<float> confHistory;
    std::map<int, int> classTally; // cumulative class duration to reduce label flipping
    std::deque<cv::Point2f> trail; // breadcrumb of centers
    int lostFrames = 0;
    TrackState state = TrackState::Tracked;
    cv::Rect lastPredBox;
    GhostTracker motion;

    Track(int trackId, cv::Rect box, int frame, int cls, const std::string &name, float conf)
        : id(trackId), lastBox(box), velocity(0.0f, 0.0f), lastSeenFrame(frame), classID(cls),
          className(name), lastConfidence(conf), bestConfidence(conf), bestClassID(cls), totalHits(1)
    {
        classHistory.push_back(cls);
        confHistory.push_back(conf);
        classTally[cls] = 1;
        trail.push_back(center());
        lastPredBox = box;
    }

    cv::Point2f center() const
    {
        return {lastBox.x + lastBox.width / 2.0f, lastBox.y + lastBox.height / 2.0f};
    }

    cv::Rect predictedBox(float damping = 0.9f) const
    {
        cv::Point2f newCenter = center() + velocity * damping;
        cv::Rect predicted(static_cast<int>(newCenter.x - lastBox.width / 2.0f),
                           static_cast<int>(newCenter.y - lastBox.height / 2.0f),
                           lastBox.width, lastBox.height);
        return predicted;
    }

    cv::Rect predictMotion(float damping, int maxWidth, int maxHeight)
    {
        if (motion.isInitialized())
        {
            return clampToFrame(motion.getPrediction(), maxWidth, maxHeight);
        }
        return clampToFrame(predictedBox(damping), maxWidth, maxHeight);
    }

    void applyPrediction(float damping, int maxWidth, int maxHeight)
    {
        if (motion.isInitialized())
        {
            lastBox = clampToFrame(motion.getPrediction(), maxWidth, maxHeight);
        }
        else
        {
            cv::Rect predicted = clampToFrame(predictedBox(damping), maxWidth, maxHeight);
            lastBox = predicted;
        }
        velocity *= damping;
    }

    void update(const cv::Rect &box, int frame, float conf, int cls, const std::string &name)
    {
        cv::Point2f currentCenter = center();
        cv::Point2f newCenter(box.x + box.width / 2.0f, box.y + box.height / 2.0f);
        cv::Point2f measuredVelocity = newCenter - currentCenter;

        // Blend measured velocity with running estimate to avoid abrupt jumps
        velocity = 0.6f * measuredVelocity + 0.4f * velocity;

        lastBox = box;
        lastSeenFrame = frame;
        state = TrackState::Tracked;
        lostFrames = 0;
        lastConfidence = conf;
        totalHits++;

        // Maintain rolling class/conf history (keep last 20 entries)
        const size_t MAX_HISTORY = 20;
        classHistory.push_back(cls);
        confHistory.push_back(conf);
        if (classHistory.size() > MAX_HISTORY)
            classHistory.pop_front();
        if (confHistory.size() > MAX_HISTORY)
            confHistory.pop_front();

        classTally[cls]++;

        if (conf >= bestConfidence)
        {
            bestConfidence = conf;
            bestClassID = cls;
            className = name;
            classID = cls;
        }

        if (!motion.isInitialized())
        {
            motion.initialize(box);
        }
        else
        {
            motion.update(box, conf);
        }
    }

    int getFramesSinceLastSeen(int currentFrame) const
    {
        return currentFrame - lastSeenFrame;
    }

    void refreshKeypoints(int step = 10)
    {
        keypoints.clear();
        for (int y = lastBox.y + step; y < lastBox.y + lastBox.height - step; y += step)
        {
            for (int x = lastBox.x + step; x < lastBox.x + lastBox.width - step; x += step)
            {
                keypoints.emplace_back(static_cast<float>(x), static_cast<float>(y));
            }
        }
        if (keypoints.empty())
        {
            keypoints.push_back({static_cast<float>(lastBox.x + lastBox.width / 2.0f),
                                 static_cast<float>(lastBox.y + lastBox.height / 2.0f)});
        }
    }

    void pushTrail()
    {
        trail.push_back(center());
        if (trail.size() > 20)
            trail.pop_front();
    }

    void updateWithFlow(const cv::Mat &prevGray, const cv::Mat &gray, int maxWidth, int maxHeight)
    {
        if (keypoints.empty())
            return;

        std::vector<cv::Point2f> nextPts;
        std::vector<uchar> status;
        std::vector<float> err;

        cv::calcOpticalFlowPyrLK(prevGray, gray, keypoints, nextPts, status, err, cv::Size(15, 15), 3,
                                 cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03));

        cv::Point2f sumDelta(0.0f, 0.0f);
        int good = 0;
        for (size_t i = 0; i < nextPts.size(); ++i)
        {
            if (status[i])
            {
                sumDelta += (nextPts[i] - keypoints[i]);
                good++;
            }
        }

        if (good > 0)
        {
            cv::Point2f avgDelta = sumDelta * (1.0f / good);
            velocity = 0.7f * avgDelta + 0.3f * velocity;

            cv::Rect shifted(lastBox.x + static_cast<int>(std::round(avgDelta.x)),
                             lastBox.y + static_cast<int>(std::round(avgDelta.y)),
                             lastBox.width, lastBox.height);
            lastBox = clampToFrame(shifted, maxWidth, maxHeight);
            keypoints.clear();
            for (size_t i = 0; i < nextPts.size(); ++i)
            {
                if (status[i])
                {
                    keypoints.push_back(nextPts[i]);
                }
            }
            if (keypoints.empty())
            {
                refreshKeypoints();
            }
        }
    }

    void updateAppearance(const cv::Mat &frame, const cv::Rect &box)
    {
        if (!frame.empty())
        {
            appearanceHist = computeHSVHist(frame, box);
        }
    }

    std::pair<int, float> majorityClass() const
    {
        // Prefer the class seen for the longest cumulative time to avoid flip-flopping
        if (classTally.empty())
            return {classID, lastConfidence};

        int bestCls = classID;
        int bestCount = 0;
        for (const auto &kv : classTally)
        {
            if (kv.second > bestCount)
            {
                bestCount = kv.second;
                bestCls = kv.first;
            }
        }

        float avgConf = 0.0f;
        for (float c : confHistory)
            avgConf += c;
        if (!confHistory.empty())
            avgConf /= static_cast<float>(confHistory.size());

        return {bestCls, avgConf};
    }

    bool markStaleIfOld(int currentFrame, int staleFrames)
    {
        if (getFramesSinceLastSeen(currentFrame) > staleFrames)
        {
            classID = -1;
            className = "Unknown";
            bestConfidence = 0.0f;
            lastConfidence = 0.0f;
            classHistory.clear();
            confHistory.clear();
            classTally.clear();
            return true;
        }
        return false;
    }
};

// map the ID numbers to real names
std::vector<std::string> getClassNames()
{
    return {
        "A10", "A400M", "AG600", "AH64", "AV8B", "B1", "B2", "B52", "Be200", "C130",
        "C17", "C2", "C5", "CH47", "CL415", "E2", "EF2000", "EMB314", "F117", "F14",
        "F15", "F16", "F18", "F22", "F35", "F4", "H6", "Il76", "J10", "J20",
        "JAS39", "JF17", "JH7", "KC135", "Ka52", "MQ9", "Mi24", "Mi8", "Mig29", "Mig31",
        "Mirage2000", "P3", "RQ4", "Rafale", "SR71", "Su24", "Su25", "Su34", "Su57", "TB2",
        "Tornado", "Tu160", "Tu22M", "Tu95", "U2", "UH60", "US2", "V22", "Vulcan", "Y20"};
}

// Helper function to calculate IOU
float calculateIOU(const cv::Rect &a, const cv::Rect &b)
{
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);

    int intersect = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int unionArea = a.area() + b.area() - intersect;

    return unionArea > 0 ? (float)intersect / unionArea : 0.0f;
}

// Helper function to calculate center-to-center distance
float calculateCenterDistance(const cv::Rect &a, const cv::Rect &b)
{
    float cx1 = a.x + a.width / 2.0f;
    float cy1 = a.y + a.height / 2.0f;
    float cx2 = b.x + b.width / 2.0f;
    float cy2 = b.y + b.height / 2.0f;

    return std::sqrt((cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2));
}

cv::Rect clampToFrame(const cv::Rect &box, int maxWidth, int maxHeight)
{
    int clampedX = std::max(0, std::min(box.x, maxWidth - box.width));
    int clampedY = std::max(0, std::min(box.y, maxHeight - box.height));
    int clampedW = std::max(1, std::min(box.width, maxWidth - clampedX));
    int clampedH = std::max(1, std::min(box.height, maxHeight - clampedY));
    return {clampedX, clampedY, clampedW, clampedH};
}

cv::Mat computeHSVHist(const cv::Mat &frame, const cv::Rect &box)
{
    cv::Rect safeBox = clampToFrame(box, frame.cols, frame.rows);
    cv::Mat roi = frame(safeBox);
    cv::Mat hsv;
    cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

    int hBins = 16, sBins = 16;
    int histSize[] = {hBins, sBins};
    float hRanges[] = {0, 180};
    float sRanges[] = {0, 256};
    const float *ranges[] = {hRanges, sRanges};
    int channels[] = {0, 1};

    cv::Mat hist;
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
    if (!hist.empty())
    {
        cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);
    }
    return hist;
}

// Non-Maximum Suppression to filter duplicate/overlapping detections
std::vector<Detection> applyNMS(const std::vector<Detection> &detections, float nmsThreshold = 0.5f)
{
    if (detections.empty())
        return detections;

    // Sort by confidence descending
    std::vector<Detection> sorted = detections;
    std::sort(sorted.begin(), sorted.end(),
              [](const Detection &a, const Detection &b)
              { return a.confidence > b.confidence; });

    std::vector<bool> suppressed(sorted.size(), false);
    std::vector<Detection> result;

    for (size_t i = 0; i < sorted.size(); i++)
    {
        if (suppressed[i])
            continue;

        result.push_back(sorted[i]);

        // Suppress overlapping boxes
        for (size_t j = i + 1; j < sorted.size(); j++)
        {
            if (suppressed[j])
                continue;

            float iou = calculateIOU(sorted[i].box, sorted[j].box);
            if (iou > nmsThreshold)
            {
                suppressed[j] = true;
            }
        }
    }

    return result;
}

// Basic Hungarian/Munkres solver for dense cost matrices (min-cost assignment)
std::vector<std::pair<int, int>> hungarianAssign(const std::vector<std::vector<float>> &costMatrix, float maxCost)
{
    if (costMatrix.empty())
        return {};

    const float INF = 1e6f;
    int n = static_cast<int>(costMatrix.size());
    int m = static_cast<int>(costMatrix[0].size());
    int dim = std::max(n, m);

    std::vector<std::vector<float>> cost(dim, std::vector<float>(dim, INF));
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            cost[i][j] = costMatrix[i][j];
        }
    }

    std::vector<float> u(dim + 1), v(dim + 1);
    std::vector<int> p(dim + 1), way(dim + 1);

    for (int i = 1; i <= dim; ++i)
    {
        p[0] = i;
        int j0 = 0;
        std::vector<float> minv(dim + 1, INF);
        std::vector<char> used(dim + 1, false);
        do
        {
            used[j0] = true;
            int i0 = p[j0], j1 = 0;
            float delta = INF;
            for (int j = 1; j <= dim; ++j)
            {
                if (used[j])
                    continue;
                float cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                if (cur < minv[j])
                {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta)
                {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= dim; ++j)
            {
                if (used[j])
                {
                    u[p[j]] += delta;
                    v[j] -= delta;
                }
                else
                {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do
        {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    std::vector<std::pair<int, int>> assignment;
    assignment.reserve(dim);
    for (int j = 1; j <= dim; ++j)
    {
        if (p[j] == 0 || p[j] > n || j > m)
            continue;
        float c = cost[p[j] - 1][j - 1];
        if (c < maxCost)
            assignment.emplace_back(p[j] - 1, j - 1);
    }
    return assignment;
}

int main(int argc, char **argv)
{
    try
    {
        // CLI: --kalman-test enables a Kalman-only window; optional "--kalman-test=start,duration" overrides timings (seconds)
        bool kalmanTest = false;
        double kalmanHoldStart = 3.0;
        double kalmanHoldDuration = 3.0;
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            const std::string flag = "--kalman-test";
            if (arg == flag)
            {
                kalmanTest = true;
            }
            else if (arg.rfind(flag + "=", 0) == 0)
            {
                kalmanTest = true;
                std::string payload = arg.substr(flag.size() + 1);
                size_t comma = payload.find(',');
                if (comma != std::string::npos)
                {
                    try
                    {
                        kalmanHoldStart = std::stod(payload.substr(0, comma));
                        kalmanHoldDuration = std::stod(payload.substr(comma + 1));
                    }
                    catch (...)
                    {
                        std::cerr << "WARN: Failed to parse kalman-test timings, using defaults 3s/3s." << std::endl;
                    }
                }
            }
        }

        // initialize detector
        std::cout << "Loading model..." << std::endl;
        Detector detector(L"last.onnx");
        std::cout << "Model loaded successfully!" << std::endl;

        // --- VIDEO INITIALIZATION ---
        cv::VideoCapture cap("test.mp4");

        if (!cap.isOpened())
        {
            std::cerr << "ERROR: Could not open video source." << std::endl;
            return -1;
        }

        // Get video properties for output
        int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps_input = cap.get(cv::CAP_PROP_FPS);
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

        // Calculate output dimensions - maintain aspect ratio, fit to 1920x1080
        int output_width = 1920;
        int output_height = 1080;
        float scale = std::min((float)output_width / frame_width, (float)output_height / frame_height);
        int scaled_width = static_cast<int>(frame_width * scale);
        int scaled_height = static_cast<int>(frame_height * scale);
        int offset_x = (output_width - scaled_width) / 2;
        int offset_y = (output_height - scaled_height) / 2;

        // Initialize video writer with standard landscape dimensions
        cv::VideoWriter writer("output.mp4", cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
                               fps_input, cv::Size(output_width, output_height));

        if (!writer.isOpened())
        {
            std::cerr << "ERROR: Could not open video writer." << std::endl;
            return -1;
        }

        std::cout << "Input video: " << frame_width << "x" << frame_height
                  << " @ " << fps_input << " FPS, " << total_frames << " frames" << std::endl;
        std::cout << "Output video: " << output_width << "x" << output_height << std::endl;
        std::cout << "Writing to output.mp4..." << std::endl;

        std::vector<std::string> classNames = getClassNames();
        cv::Mat frame, resized_frame, output_frame;
        cv::Mat prev_gray, gray;
        int frame_count = 0;
        double smooth_fps = 0.0;

        // Simplified tracking parameters
        std::vector<Track> activeTracks;
        int nextTrackID = 1;
        const int MAX_FRAMES_LOST = 200;     // frames to keep as lost before removal
        const float DIST_GATE_BASE = 450.0f; // Base gate
        const float IOU_GATE = 0.05f;        // Minimum IoU gate
        const float MERGE_DISTANCE = 80.0f;  // Conservative merge radius to avoid churn
        const float MERGE_IOU = 0.65f;
        const float HIGH_CONF = 0.35f;          // Primary association threshold
        const float LOW_CONF = 0.12f;           // Recovery threshold
        const float PREDICTION_DAMPING = 0.90f; // Slightly stronger damping
        const int CLASS_STALE_FRAMES = 90;      // After this many frames unseen, clear class label
        const float MAX_ASSIGN_COST = 2.5f;     // Gate for Hungarian results
        const float COSINE_PENALTY = 0.25f;     // Penalty weight for opposite motion
        const float APPEARANCE_GATE = 0.05f;    // If appearance similarity below and IoU small, gate out

        std::cout << "Processing video (no preview window)..." << std::endl;
        if (kalmanTest)
        {
            std::cout << "Kalman-only test enabled: disabling detections from "
                      << kalmanHoldStart << "s to " << (kalmanHoldStart + kalmanHoldDuration) << "s" << std::endl;
        }

        // --- THE VIDEO LOOP ---
        bool suppressDetections = false;
        bool suppressDetectionsPrev = false;
        while (true)
        {
            auto start = std::chrono::high_resolution_clock::now();

            cap >> frame;
            if (frame.empty())
                break;

            frame_count++;
            std::cout << "Processing frame " << frame_count << "/" << total_frames << "\r" << std::flush;

            // Resize frame to fit output dimensions while maintaining aspect ratio
            cv::resize(frame, resized_frame, cv::Size(scaled_width, scaled_height));

            // Prepare grayscale for optical flow
            cv::cvtColor(resized_frame, gray, cv::COLOR_BGR2GRAY);

            // Create black canvas and place resized frame in center
            output_frame = cv::Mat::zeros(output_height, output_width, frame.type());
            resized_frame.copyTo(output_frame(cv::Rect(offset_x, offset_y, scaled_width, scaled_height)));

            // === STEP 0: Optical flow keep-alive ===
            if (!prev_gray.empty())
            {
                for (auto &track : activeTracks)
                {
                    track.updateWithFlow(prev_gray, gray, scaled_width, scaled_height);
                }
            }

            // === STEP 1: Run YOLO detection (optionally suppressed for Kalman-only test window) ===
            double elapsedSeconds = (fps_input > 1e-6) ? (frame_count / fps_input) : 0.0;
            suppressDetections = kalmanTest && elapsedSeconds >= kalmanHoldStart && elapsedSeconds < (kalmanHoldStart + kalmanHoldDuration);
            if (suppressDetections != suppressDetectionsPrev)
            {
                if (suppressDetections)
                    std::cout << "\n[Kalman test] Detections OFF (" << elapsedSeconds << "s)" << std::endl;
                else
                    std::cout << "\n[Kalman test] Detections ON (" << elapsedSeconds << "s)" << std::endl;
                suppressDetectionsPrev = suppressDetections;
            }

            std::vector<Detection> results;
            if (!suppressDetections)
            {
                results = detector.run(resized_frame);
            }
            std::vector<Detection> highDetections;
            std::vector<Detection> lowDetections;
            for (const auto &det : results)
            {
                if (det.confidence >= HIGH_CONF)
                {
                    highDetections.push_back(det);
                }
                else if (det.confidence >= LOW_CONF)
                {
                    lowDetections.push_back(det);
                }
            }

            auto buildHists = [&](const std::vector<Detection> &dets)
            {
                std::vector<cv::Mat> hists;
                hists.reserve(dets.size());
                for (const auto &d : dets)
                    hists.push_back(computeHSVHist(resized_frame, d.box));
                return hists;
            };

            std::vector<cv::Mat> highHists = buildHists(highDetections);
            std::vector<cv::Mat> lowHists = buildHists(lowDetections);

            // === STEP 3: Match detections to existing tracks via Hungarian ===
            std::vector<int> trackMatchedDet(activeTracks.size(), -1);
            std::vector<bool> trackMatchIsLow(activeTracks.size(), false);
            std::vector<bool> highMatched(highDetections.size(), false);
            std::vector<bool> lowMatched(lowDetections.size(), false);

            std::vector<cv::Rect> predictedBoxes;
            predictedBoxes.reserve(activeTracks.size());
            for (auto &track : activeTracks)
            {
                cv::Rect pred;
                if (track.motion.isInitialized())
                {
                    pred = track.motion.predict();
                }
                else
                {
                    pred = track.predictedBox(PREDICTION_DAMPING);
                }
                pred = clampToFrame(pred, scaled_width, scaled_height);
                track.lastPredBox = pred;
                predictedBoxes.push_back(pred);
            }

            auto buildCostMatrix = [&](const std::vector<Detection> &dets, const std::vector<cv::Mat> &hists,
                                       const std::vector<int> &trackIndices)
            {
                std::vector<std::vector<float>> cost(trackIndices.size(), std::vector<float>(dets.size(), 1e6f));
                for (size_t ti = 0; ti < trackIndices.size(); ++ti)
                {
                    int t = trackIndices[ti];
                    float velMag = std::sqrt(activeTracks[t].velocity.x * activeTracks[t].velocity.x +
                                             activeTracks[t].velocity.y * activeTracks[t].velocity.y);
                    float dynamicGate = DIST_GATE_BASE + std::min(200.0f, velMag * 5.0f);
                    bool isLost = activeTracks[t].state == TrackState::Lost;
                    if (isLost)
                        dynamicGate *= 1.3f;

                    for (size_t d = 0; d < dets.size(); ++d)
                    {
                        float distance = calculateCenterDistance(predictedBoxes[t], dets[d].box);
                        float iou = calculateIOU(predictedBoxes[t], dets[d].box);

                        if (!(distance < dynamicGate || iou > IOU_GATE))
                            continue;

                        double appearanceSim = 0.0;
                        if (!activeTracks[t].appearanceHist.empty() && !hists[d].empty())
                            appearanceSim = cv::compareHist(activeTracks[t].appearanceHist, hists[d], cv::HISTCMP_CORREL);

                        // Gate very dissimilar appearance when geometry is weak
                        if (!isLost && appearanceSim < APPEARANCE_GATE && iou < 0.01f)
                            continue;

                        float classBonus = (activeTracks[t].bestClassID == dets[d].classID) ? 0.1f : 0.0f;
                        float confidenceBoost = dets[d].confidence * 0.05f;

                        // Velocity direction penalty
                        cv::Point2f predCenter(predictedBoxes[t].x + predictedBoxes[t].width / 2.0f,
                                               predictedBoxes[t].y + predictedBoxes[t].height / 2.0f);
                        cv::Point2f detCenter(dets[d].box.x + dets[d].box.width / 2.0f,
                                              dets[d].box.y + dets[d].box.height / 2.0f);
                        cv::Point2f delta = detCenter - predCenter;
                        float deltaNorm = std::sqrt(delta.x * delta.x + delta.y * delta.y);
                        float velNorm = std::sqrt(activeTracks[t].velocity.x * activeTracks[t].velocity.x + activeTracks[t].velocity.y * activeTracks[t].velocity.y);
                        float cosPenalty = 0.0f;
                        if (deltaNorm > 1e-3f && velNorm > 1e-3f)
                        {
                            float cosSim = (delta.x * activeTracks[t].velocity.x + delta.y * activeTracks[t].velocity.y) / (deltaNorm * velNorm + 1e-6f);
                            if (cosSim < 0.0f)
                                cosPenalty = COSINE_PENALTY * (-cosSim);
                        }

                        float normDist = distance / (dynamicGate + 1e-3f);
                        float baseCost = (1.0f - iou) + 0.35f * normDist + cosPenalty - static_cast<float>(0.25f * appearanceSim) - classBonus - confidenceBoost;
                        if (isLost)
                            baseCost -= 0.1f; // slightly prefer re-association for lost tracks
                        cost[ti][d] = std::max(0.0f, baseCost);
                    }
                }
                return cost;
            };

            // Primary association with high-confidence dets
            std::vector<int> allTrackIdx(activeTracks.size());
            std::iota(allTrackIdx.begin(), allTrackIdx.end(), 0);
            if (!activeTracks.empty() && !highDetections.empty())
            {
                auto highCost = buildCostMatrix(highDetections, highHists, allTrackIdx);
                auto primaryAssign = hungarianAssign(highCost, MAX_ASSIGN_COST);
                for (const auto &match : primaryAssign)
                {
                    int t = match.first;
                    int d = match.second;
                    trackMatchedDet[t] = d;
                    trackMatchIsLow[t] = false;
                    highMatched[d] = true;
                }
            }

            // Recovery association with low-confidence dets for still-unmatched tracks
            std::vector<int> unmatchedTracks;
            for (size_t t = 0; t < activeTracks.size(); ++t)
            {
                if (trackMatchedDet[t] == -1)
                    unmatchedTracks.push_back(static_cast<int>(t));
            }

            if (!unmatchedTracks.empty() && !lowDetections.empty())
            {
                auto lowCost = buildCostMatrix(lowDetections, lowHists, unmatchedTracks);
                auto recoveryAssign = hungarianAssign(lowCost, MAX_ASSIGN_COST * 0.9f);
                for (const auto &match : recoveryAssign)
                {
                    int t = unmatchedTracks[match.first];
                    int d = match.second;
                    if (trackMatchedDet[t] == -1)
                    {
                        trackMatchedDet[t] = d;
                        trackMatchIsLow[t] = true;
                        lowMatched[d] = true;
                    }
                }
            }

            // === STEP 4: Update matched tracks ===
            for (size_t t = 0; t < activeTracks.size(); t++)
            {
                if (trackMatchedDet[t] >= 0)
                {
                    bool useLow = trackMatchIsLow[t];
                    const Detection &det = useLow ? lowDetections[trackMatchedDet[t]] : highDetections[trackMatchedDet[t]];
                    const auto &histVec = useLow ? lowHists : highHists;

                    std::string aircraftName = (det.classID >= 0 && det.classID < classNames.size())
                                                   ? classNames[det.classID]
                                                   : "Unknown";
                    activeTracks[t].update(det.box, frame_count, det.confidence, det.classID, aircraftName);
                    activeTracks[t].updateAppearance(resized_frame, det.box);
                    if (trackMatchIsLow[t] && !histVec.empty())
                        activeTracks[t].appearanceHist = histVec[trackMatchedDet[t]];
                    activeTracks[t].refreshKeypoints();
                    activeTracks[t].pushTrail();
                }
            }

            // === STEP 4b: Propagate unmatched tracks forward ===
            for (size_t t = 0; t < activeTracks.size(); t++)
            {
                if (trackMatchedDet[t] == -1)
                {
                    // Use last predicted box (already from Kalman) to advance
                    activeTracks[t].lastBox = clampToFrame(activeTracks[t].lastPredBox, scaled_width, scaled_height);
                    activeTracks[t].velocity *= PREDICTION_DAMPING;
                    activeTracks[t].markStaleIfOld(frame_count, CLASS_STALE_FRAMES);
                    activeTracks[t].state = TrackState::Lost;
                    activeTracks[t].lostFrames++;
                    activeTracks[t].pushTrail();
                }
            }

            // === STEP 5: Create new tracks for unmatched high-confidence detections ===
            for (size_t d = 0; d < highDetections.size(); d++)
            {
                if (!highMatched[d])
                {
                    const Detection &det = highDetections[d];
                    std::string aircraftName = (det.classID >= 0 && det.classID < classNames.size())
                                                   ? classNames[det.classID]
                                                   : "Unknown";

                    activeTracks.emplace_back(nextTrackID++, det.box, frame_count,
                                              det.classID, aircraftName, det.confidence);
                    if (highHists.size() > d)
                        activeTracks.back().appearanceHist = highHists[d];
                    activeTracks.back().refreshKeypoints();
                    activeTracks.back().motion.initialize(det.box);
                    activeTracks.back().pushTrail();

                    std::cout << "\n[NEW TRACK] ID" << activeTracks.back().id << ": "
                              << aircraftName << " @ " << (int)(det.confidence * 100) << "%" << std::endl;
                }
            }

            // === STEP 5: Merge overlapping tracks (more conservative) ===
            for (size_t i = 0; i < activeTracks.size(); i++)
            {
                for (size_t j = i + 1; j < activeTracks.size(); j++)
                {
                    float distance = calculateCenterDistance(activeTracks[i].lastBox, activeTracks[j].lastBox);
                    float iou = calculateIOU(activeTracks[i].lastBox, activeTracks[j].lastBox);

                    if (distance < MERGE_DISTANCE && iou > MERGE_IOU)
                    {
                        size_t keep = (activeTracks[i].bestConfidence >= activeTracks[j].bestConfidence) ? i : j;
                        size_t drop = (keep == i) ? j : i;
                        std::cout << "\n[MERGE] Removing ID" << activeTracks[drop].id
                                  << " (duplicate of ID" << activeTracks[keep].id << ")" << std::endl;
                        activeTracks.erase(activeTracks.begin() + drop);
                        if (drop < i)
                            i--;
                        j = i + 1;
                    }
                }
            }

            // === STEP 6: Remove old tracks ===
            activeTracks.erase(
                std::remove_if(activeTracks.begin(), activeTracks.end(),
                               [MAX_FRAMES_LOST](const Track &t)
                               {
                                   bool shouldRemove = t.lostFrames > MAX_FRAMES_LOST || t.state == TrackState::Removed;
                                   return shouldRemove;
                               }),
                activeTracks.end());

            // === STEP 7: Draw all active tracks ===
            int trackedCount = 0;
            int lostCount = 0;
            for (const auto &track : activeTracks)
            {
                cv::Rect adjusted_box(track.lastBox.x + offset_x, track.lastBox.y + offset_y,
                                      track.lastBox.width, track.lastBox.height);

                if (track.state == TrackState::Tracked)
                    trackedCount++;
                else if (track.state == TrackState::Lost)
                    lostCount++;

                // Draw trail (breadcrumb)
                if (track.trail.size() >= 2)
                {
                    std::vector<cv::Point> pts;
                    pts.reserve(track.trail.size());
                    for (const auto &p : track.trail)
                        pts.emplace_back(static_cast<int>(p.x) + offset_x, static_cast<int>(p.y) + offset_y);
                    cv::polylines(output_frame, pts, false, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
                }

                // Draw box
                cv::Scalar boxColor = (track.state == TrackState::Tracked) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
                cv::rectangle(output_frame, adjusted_box, boxColor, (track.state == TrackState::Tracked) ? 2 : 2, cv::LINE_AA);

                // If lost, also draw the Kalman-predicted box (ahead) to show where we think it goes
                if (track.state == TrackState::Lost && track.motion.isInitialized())
                {
                    cv::Rect predDraw = clampToFrame(track.lastPredBox, scaled_width, scaled_height);
                    predDraw.x += offset_x;
                    predDraw.y += offset_y;
                    cv::rectangle(output_frame, predDraw, cv::Scalar(0, 0, 200), 1, cv::LINE_4);
                }

                // Predicted next point marker (use last prediction when available)
                cv::Point predictedCenter;
                if (track.motion.isInitialized())
                {
                    cv::Rect p = track.lastPredBox;
                    predictedCenter = {p.x + p.width / 2 + offset_x, p.y + p.height / 2 + offset_y};
                }
                else
                {
                    predictedCenter = {adjusted_box.x + adjusted_box.width / 2, adjusted_box.y + adjusted_box.height / 2};
                }
                cv::circle(output_frame, predictedCenter, 3, cv::Scalar(0, 165, 255), cv::FILLED);

                // Draw label based on majority class and averaged confidence
                auto maj = track.majorityClass();
                int labelCls = (maj.first >= 0 && maj.first < (int)classNames.size()) ? maj.first : track.classID;
                std::string labelName = (labelCls >= 0 && labelCls < (int)classNames.size()) ? classNames[labelCls] : track.className;
                float displayConf = std::max({track.lastConfidence, track.bestConfidence, maj.second});
                std::string label = "ID" + std::to_string(track.id) + ": " + labelName +
                                    " " + std::to_string((int)(displayConf * 100)) + "%";

                int baseLine;
                cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseLine);
                cv::rectangle(output_frame,
                              cv::Point(adjusted_box.x, adjusted_box.y - textSize.height - 8),
                              cv::Point(adjusted_box.x + textSize.width, adjusted_box.y),
                              cv::Scalar(0, 255, 0), -1);

                cv::putText(output_frame, label, {adjusted_box.x, adjusted_box.y - 5},
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
            }

            // === Display active track count ===
            auto loopEnd = std::chrono::high_resolution_clock::now();
            double elapsedMs = std::chrono::duration<double, std::milli>(loopEnd - start).count();
            double instFps = elapsedMs > 0.0 ? 1000.0 / elapsedMs : 0.0;
            smooth_fps = (smooth_fps == 0.0) ? instFps : (0.9 * smooth_fps + 0.1 * instFps);

            std::string trackInfo = "Tracked: " + std::to_string(trackedCount) +
                                    " | Lost: " + std::to_string(lostCount) +
                                    " | FPS: " + std::to_string(static_cast<int>(smooth_fps));
            if (kalmanTest && suppressDetections)
                trackInfo += " | Kalman-only";
            cv::putText(output_frame, trackInfo, {10, output_height - 20},
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);

            // Write frame to output video
            writer.write(output_frame);

            // carry gray frame forward for optical flow
            prev_gray = gray.clone();
        }

        std::cout << "\nProcessing complete! Output saved to output.mp4" << std::endl;
        std::cout << "Total frames processed: " << frame_count << std::endl;

        cap.release();
        writer.release();
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return -1;
    }
}
