#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "detector.h"

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

int main()
{
    try
    {
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
        int frame_count = 0;

        std::cout << "Processing video (no preview window)..." << std::endl;

        // --- THE VIDEO LOOP ---
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

            // Create black canvas and place resized frame in center
            output_frame = cv::Mat::zeros(output_height, output_width, frame.type());
            resized_frame.copyTo(output_frame(cv::Rect(offset_x, offset_y, scaled_width, scaled_height)));

            // Run detection on the resized frame
            std::vector<Detection> results = detector.run(resized_frame);

            // Draw detections (adjust coordinates for the offset)
            for (const auto &det : results)
            {
                // Adjust bounding box coordinates to account for letterboxing
                cv::Rect adjusted_box(det.box.x + offset_x, det.box.y + offset_y,
                                      det.box.width, det.box.height);
                cv::rectangle(output_frame, adjusted_box, cv::Scalar(0, 255, 0), 2);

                std::string aircraftName = (det.classID >= 0 && det.classID < classNames.size())
                                               ? classNames[det.classID]
                                               : "Unknown";

                std::string label = aircraftName + " " + std::to_string((int)(det.confidence * 100)) + "%";

                cv::putText(output_frame, label, {adjusted_box.x, adjusted_box.y - 10},
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            }

            // --- FPS CALCULATION ---
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            double fps = 1.0 / diff.count();
            std::string fpsLabel = "FPS: " + std::to_string((int)fps);
            cv::putText(output_frame, fpsLabel, {10, 40}, cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 255), 3);

            // Write frame to output video
            writer.write(output_frame);
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