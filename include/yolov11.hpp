#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>
#include <functional>
#include <cassert>
#include <map>
#include <fstream>

class ObjectBBox {
public:
    std::string label;
    float conf;
    cv::Rect rect;
    float x1, x2, y1, y2;

    ObjectBBox(const std::string &lbl, float conf, float cx, float cy, float w, float h, float scale_x, float scale_y);
    cv::Mat draw(cv::Mat &img) const;
};

float calculateIoU(const ObjectBBox &box1, const ObjectBBox &box2);

class YOLOv11 {
    
public:
    using ClassChecker = std::function<bool(int, const std::string &)>;

private:
    cv::dnn::Net net_;
    cv::Size input_size_;
    std::map<int, std::string> class_names_;
    float min_conf_;
    float iou_thresh_;
    ClassChecker valid_class_checker_;

    cv::Mat preprocess(const cv::Mat& image);
    std::vector<ObjectBBox> postprocess(const cv::Mat &output, const cv::Size &original_size);
    void loadClassNames(const std::string &names_file);

public:
    YOLOv11(
        const std::string &model_path,
        float min_conf = 0.45f,
        float iou_thresh = 0.45f,
        ClassChecker valid_class_checker = nullptr,
        const std::string &names_file = ""
    );
    std::map<int, std::string> getClassIdNamePairs() const;
    std::vector<ObjectBBox> detect(const cv::Mat &image);
};