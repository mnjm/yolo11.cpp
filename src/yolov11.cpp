#include "yolov11.hpp"
#include "debug.hpp"

ObjectBBox::ObjectBBox(const std::string &lbl, float conf_, float cx, float cy, float w, float h, float scale_x, float scale_y)
    : label(lbl), conf(conf_)
{
    // Convert from center coordinates to top-left coordinates
    x1 = (cx - w / 2) * scale_x;
    y1 = (cy - h / 2) * scale_y;
    x2 = (cx + w / 2) * scale_x;
    y2 = (cy + h / 2) * scale_y;
    rect = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
}

cv::Mat ObjectBBox::draw(cv::Mat &img) const
{
    cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 2);
    cv::putText(img, label + " " + std::to_string(conf).substr(0, 4),
                cv::Point(rect.x, rect.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    return img;
}

float calculateIoU(const ObjectBBox &box1, const ObjectBBox &box2)
{
    auto intersection = box1.rect & box2.rect;
    if (intersection.empty())
        return 0.0f;

    float intersection_area = intersection.area();
    float union_area = box1.rect.area() + box2.rect.area() - intersection_area;
    return intersection_area / union_area;
}

void YOLOv11::loadClassNames(const std::string &names_file)
{
    std::ifstream file(names_file);
    assert(file.is_open() && "Failed to open class names file `coco.names`");

    std::string line;
    int class_id = 0;

    while (std::getline(file, line))
    {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
        line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);

        // Skip empty lines
        if (!line.empty())
        {
            class_names_[class_id++] = line;
        }
    }

    assert(!class_names_.empty() && "No class names loaded from file `coco.names`");
    DEBUG_PRINT("Loaded " << class_names_.size() << " class names");
}

YOLOv11::YOLOv11(
    const std::string &model_path,
    float min_conf,
    float iou_thresh,
    ClassChecker valid_class_checker,
    const std::string &names_file
) : min_conf_(min_conf), iou_thresh_(iou_thresh)
{
    // Load the network
    net_ = cv::dnn::readNetFromONNX(model_path);
#if defined(CUDA_ACC)
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    std::cout << "Using CUDA" << std::endl;
#elif defined(OPENCL_ACC)
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
    std::cout << "Using OPENCL" << std::endl;
#else
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    std::cout << "Using CPU" << std::endl;
#endif
    assert(!net_.empty() && "Failed to load ONNX model");
    // If names_file is empty, use "coco.names" from the location of the model_path
    std::string resolved_names_file = names_file;
    if (resolved_names_file.empty())
    {
        size_t last_slash_idx = model_path.find_last_of("\\/");
        if (std::string::npos != last_slash_idx)
        {
            resolved_names_file = model_path.substr(0, last_slash_idx + 1) + "coco.names";
        }
        else
        {
            resolved_names_file = "coco.names";
        }
    }
    input_size_.width = 640;
    input_size_.height = 640;
    // Load class names coco.names file
    loadClassNames(resolved_names_file);

    // Set class checker
    valid_class_checker_ = valid_class_checker ? valid_class_checker : [](int, const std::string &)
    { return true; };
}

cv::Mat YOLOv11::preprocess(const cv::Mat &image)
{
    cv::Mat resized;
    cv::resize(image, resized, input_size_);
    DEBUG_PRINT_MAT_SHAPE(resized);

    cv::Mat blob;
    cv::dnn::blobFromImage(resized, blob, 1.0 / 255.0, input_size_,
                           cv::Scalar(), true, false, CV_32F);
    DEBUG_PRINT_MAT_SHAPE(blob);
    return blob;
}

std::vector<ObjectBBox> YOLOv11::postprocess(const cv::Mat &output, const cv::Size &original_size)
{
    DEBUG_PRINT_MAT_SHAPE(output);
    assert(output.dims == 2 && output.cols > 0 &&
           output.rows == (4 + class_names_.size()) &&
           "Invalid output shape");

    std::vector<ObjectBBox> valid_boxes;
    std::vector<bool> suppressed(output.cols, false);
    cv::Point2f scale(
        static_cast<float>(original_size.width) / input_size_.width,
        static_cast<float>(original_size.height) / input_size_.height);

    // Find max confidence for each detection
    std::vector<std::pair<float, int>> conf_idx_pairs;
    for (int i = 0; i < output.cols; ++i)
    {
        cv::Mat scores = output.rowRange(4, output.rows).col(i);
        double max_conf;
        cv::Point max_loc;
        cv::minMaxLoc(scores, nullptr, &max_conf, nullptr, &max_loc);
        conf_idx_pairs.push_back({max_conf, i});
    }

    // Sort by confidence
    std::sort(conf_idx_pairs.begin(), conf_idx_pairs.end(),
              std::greater<std::pair<float, int>>());

    // NMS
    for (size_t i = 0; i < conf_idx_pairs.size(); ++i)
    {
        int idx1 = conf_idx_pairs[i].second;
        if (suppressed[idx1])
            continue;

        // Get class info
        cv::Mat scores = output.rowRange(4, output.rows).col(idx1);
        double max_conf;
        cv::Point max_loc;
        cv::minMaxLoc(scores, nullptr, &max_conf, nullptr, &max_loc);
        int class_id = max_loc.y;

        if (!valid_class_checker_(class_id, class_names_[class_id]) ||
            max_conf < min_conf_)
            continue;

        // Create bbox
        ObjectBBox bbox1(
            class_names_[class_id],
            max_conf,
            output.at<float>(0, idx1),
            output.at<float>(1, idx1),
            output.at<float>(2, idx1),
            output.at<float>(3, idx1),
            scale.x,
            scale.y);
        valid_boxes.push_back(bbox1);

        // Suppress overlapping boxes
        for (size_t j = i + 1; j < conf_idx_pairs.size(); ++j)
        {
            int idx2 = conf_idx_pairs[j].second;
            if (suppressed[idx2])
                continue;

            ObjectBBox bbox2(
                class_names_[class_id],
                output.at<float>(4 + class_id, idx2),
                output.at<float>(0, idx2),
                output.at<float>(1, idx2),
                output.at<float>(2, idx2),
                output.at<float>(3, idx2),
                scale.x,
                scale.y);

            if (calculateIoU(bbox1, bbox2) > iou_thresh_)
            {
                suppressed[idx2] = true;
            }
        }
    }

    return valid_boxes;
}

std::vector<ObjectBBox> YOLOv11::detect(const cv::Mat &image)
{
    assert(!image.empty() && image.type() == CV_8UC3 && "Invalid input image");
    cv::Size original_size = image.size();

    // Preprocess
    cv::Mat blob = preprocess(image);

    // Forward pass
    net_.setInput(blob);
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());
    assert(outputs.size() == 1 && "Unexpected number of outputs");
    cv::Mat rawOutput = outputs[0].reshape(0, 4 + class_names_.size());
    // Postprocess
    return postprocess(rawOutput, original_size);
}


std::map<int, std::string> YOLOv11::getClassIdNamePairs() const
{
    return class_names_;
}