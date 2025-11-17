# YOLO11.cpp

Lite-weight Optimized C++ wrapper for running YOLO11 object detection using an ONNX model with OpenCV's DNN.

To use it in your project, just include the `./include/yolov11.hpp` and `./src/yolov11.cpp`.

Note: for this to work, place the `coco.names` file in the same directory as the model

CUDA acceleration can be enabled using [`CUDA_ACC`](https://github.com/mnjm/yolo11.cpp/blob/ea0701b79efdde78523e15c5ef5dc021e161c94a/include/yolov11.hpp#L3C12-L3C20) macro. Same with OpenCL using [`OPENCL_ACC`](https://github.com/mnjm/yolo11.cpp/blob/ea0701b79efdde78523e15c5ef5dc021e161c94a/include/yolov11.hpp#L4)

## Example

Running on an Image

```cpp
#include <opencv2/opencv.hpp>
#include "yolov11.hpp"
#include <iostream>

int main() {
    YOLOv11 model("yolo11s.onnx");
    cv::Mat img = cv::imread("sample.jpg");

    std::vector<ObjectBBox> bbox_list = model.detect(img);
    for (auto& bbox : bbox_list) {
        std::cout << "Label:" << bbox.label << " Conf: " << bbox.conf;
        std::cout << "(" << bbox.x1 << ", " << bbox.y1 << ") ";
        std::cout << "(" << bbox.x2 << ", " << bbox.y2 << ")" << std::endl;
        bbox.draw(img);
    }

    cv::imwrite("sample_out.jpg", img);
    return 0;
}
```

## Class Targeted NMS

You can pass a function or callable to filter valid classes, making NMS slightly more efficient.

```cpp
YOLOv11 model("yolo11s.onnx", 0.45f, 0.45f,
    [](int lbl_id, const std::string& lbl) {
        return lbl_id >= 0 && lbl_id <= 8;/* Only vehicles */
    }
);
```

(or)

```cpp
std::map<int, std::string> valid_class_d = {
    {1, "bicycle"},
    {2, "car"},
    {3, "motorcycle"},
    {4, "airplane"},
    {5, "bus"},
    {6, "train"},
    {7, "truck"},
    {8, "boat"},
};

YOLOv11 model("yolo11s.onnx", 0.45f, 0.45f,
    [](int lbl_id, const std::string& lbl) {
        return valid_class_d.find(lbl_id) != valid_class_d.end();
    }
);
```

To get all `{class_id, name}` pairs:
```cpp
for (const auto& [id, name] : model.getClassIdNamePairs()) {
    std::cout << id << ": " << name << std::endl;
}
```
