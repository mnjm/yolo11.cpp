#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include "yolov11.hpp"

int main(int argc, char **argv)
{
    assert(argc == 3 && "Usage: <program> <model_path> <video_path>");

    std::string model_path = argv[1];
    std::string inp_path = argv[2];
    std::string out_path = inp_path.substr(0, inp_path.find_last_of('.')) + "_out.mp4";

    cv::VideoCapture cap(inp_path);
    assert(cap.isOpened() && "Error: Cannot open video file");

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter writer(out_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
    assert(writer.isOpened() && "Error: Cannout open output video file");

    YOLOv11 model(
        model_path,
        0.45f,
        0.45f,
        [](int lbl_id, const std::string lbl)
        { return lbl_id >= 0 && lbl_id <= 8; } /* Only vehicles */
    );

    cv::Mat frame;
    while (cap.read(frame))
    {
        std::vector<ObjectBBox> bbox_l = model.detect(frame);
        for (auto& bbox: bbox_l) {
            std::cout << "Label:" << bbox.label << " Conf: " << bbox.conf;
            std::cout << "(" << bbox.x1 << ", " << bbox.y1 << ") ";
            std::cout << "(" << bbox.x2 << ", " << bbox.y2 << ")" << std::endl;
            bbox.draw(frame);
        }
        writer.write(frame);
        cv::imshow("Output", frame);
        char key = cv::waitKey(1);
        if (key == 27 || key == 'q') // Exit on q or Esc
            break;
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();

    std::cout << "Video saved as: " << out_path << std::endl;
    return 0;
}