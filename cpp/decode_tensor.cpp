#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::array<char, 8> kMagic = {'T', 'D', 'E', 'T', 'E', 'N', 'S', '1'};
constexpr float kXScale = 1.0f;
constexpr float kYScale = 1.0f;
constexpr float kWScale = 1.0f;
constexpr float kHScale = 1.0f;
constexpr float kAnchorScale = 3.0f;
constexpr float kNmsIouThreshold = 0.3f;

const std::vector<std::string> kCocoLabels = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "???", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "???", "backpack", "umbrella",
    "???", "???", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "???",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "???", "dining table",
    "???", "???", "toilet", "???", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "???", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
};

struct TensorInput {
    std::string image_name;
    std::uint32_t image_width = 0;
    std::uint32_t image_height = 0;
    std::uint32_t channels = 0;
    std::uint32_t anchor_count = 0;
    std::uint32_t class_count = 0;
    std::uint32_t max_results = 0;
    float score_threshold = 0.0f;
    std::vector<float> scores;
    std::vector<float> boxes;
};

struct Anchor {
    float y_center;
    float x_center;
    float height;
    float width;
};

struct Detection {
    int class_id = -1;
    std::string class_name;
    float score = 0.0f;
    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
};

template <typename T>
void read_exact(std::ifstream& input, T& value) {
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) {
        throw std::runtime_error("unexpected end of tensor file");
    }
}

void read_bytes(std::ifstream& input, char* destination, std::size_t byte_count) {
    input.read(destination, static_cast<std::streamsize>(byte_count));
    if (!input) {
        throw std::runtime_error("unexpected end of tensor file");
    }
}

std::size_t checked_multiply(std::size_t a, std::size_t b, const std::string& what) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::runtime_error("size overflow while reading " + what);
    }
    return a * b;
}

TensorInput read_tensor_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open tensor file: " + path);
    }

    std::array<char, 8> magic{};
    read_bytes(input, magic.data(), magic.size());
    if (magic != kMagic) {
        throw std::runtime_error("not a TDETENS1 tensor file");
    }

    std::uint32_t version = 0;
    std::uint32_t name_length = 0;
    TensorInput tensor;
    read_exact(input, version);
    read_exact(input, name_length);
    read_exact(input, tensor.image_width);
    read_exact(input, tensor.image_height);
    read_exact(input, tensor.channels);
    read_exact(input, tensor.anchor_count);
    read_exact(input, tensor.class_count);
    read_exact(input, tensor.max_results);
    read_exact(input, tensor.score_threshold);

    if (version != 1) {
        throw std::runtime_error("unsupported tensor file version");
    }
    if (tensor.class_count == 0 || tensor.class_count > kCocoLabels.size()) {
        throw std::runtime_error("unsupported class count in tensor file");
    }
    if (tensor.anchor_count == 0) {
        throw std::runtime_error("tensor file has no anchors");
    }

    tensor.image_name.resize(name_length);
    if (name_length > 0) {
        read_bytes(input, tensor.image_name.data(), tensor.image_name.size());
    }

    const std::size_t score_count = checked_multiply(
        tensor.anchor_count, tensor.class_count, "scores"
    );
    const std::size_t box_count = checked_multiply(tensor.anchor_count, 4, "boxes");
    tensor.scores.resize(score_count);
    tensor.boxes.resize(box_count);
    read_bytes(
        input,
        reinterpret_cast<char*>(tensor.scores.data()),
        checked_multiply(score_count, sizeof(float), "score bytes")
    );
    read_bytes(
        input,
        reinterpret_cast<char*>(tensor.boxes.data()),
        checked_multiply(box_count, sizeof(float), "box bytes")
    );
    return tensor;
}

std::vector<Anchor> generate_efficientdet_anchors(int image_width, int image_height) {
    constexpr int min_level = 3;
    constexpr int max_level = 7;
    constexpr int num_scales = 3;
    constexpr std::array<float, 3> aspect_ratios = {1.0f, 2.0f, 0.5f};

    std::vector<Anchor> anchors;
    for (int level = min_level; level <= max_level; ++level) {
        const int stride = 1 << level;
        const int feature_height = static_cast<int>(std::ceil(image_height / static_cast<float>(stride)));
        const int feature_width = static_cast<int>(std::ceil(image_width / static_cast<float>(stride)));

        for (int y = 0; y < feature_height; ++y) {
            for (int x = 0; x < feature_width; ++x) {
                for (int scale = 0; scale < num_scales; ++scale) {
                    for (const float aspect_ratio : aspect_ratios) {
                        const float octave_scale = std::pow(2.0f, scale / static_cast<float>(num_scales));
                        const float ratio_sqrt = std::sqrt(aspect_ratio);
                        const float base_anchor_size = std::min(
                            kAnchorScale * stride / static_cast<float>(std::min(image_width, image_height)),
                            1.0f
                        );
                        anchors.push_back({
                            (y + 0.5f) / feature_height,
                            (x + 0.5f) / feature_width,
                            base_anchor_size * octave_scale / ratio_sqrt,
                            base_anchor_size * octave_scale * ratio_sqrt,
                        });
                    }
                }
            }
        }
    }
    return anchors;
}

float clamp(float value, float low, float high) {
    return std::max(low, std::min(value, high));
}

Detection decode_box(
    const TensorInput& tensor,
    const Anchor& anchor,
    std::size_t anchor_index,
    int class_id,
    float score
) {
    const std::size_t base = anchor_index * 4;
    const float box_y = tensor.boxes[base + 0];
    const float box_x = tensor.boxes[base + 1];
    const float box_h = tensor.boxes[base + 2];
    const float box_w = tensor.boxes[base + 3];

    const float y_center = box_y / kYScale * anchor.height + anchor.y_center;
    const float x_center = box_x / kXScale * anchor.width + anchor.x_center;
    const float height = std::exp(box_h / kHScale) * anchor.height;
    const float width = std::exp(box_w / kWScale) * anchor.width;

    float x_min = (x_center - width / 2.0f) * tensor.image_width;
    float y_min = (y_center - height / 2.0f) * tensor.image_height;
    float x_max = (x_center + width / 2.0f) * tensor.image_width;
    float y_max = (y_center + height / 2.0f) * tensor.image_height;

    x_min = clamp(x_min, 0.0f, static_cast<float>(tensor.image_width));
    y_min = clamp(y_min, 0.0f, static_cast<float>(tensor.image_height));
    x_max = clamp(x_max, 0.0f, static_cast<float>(tensor.image_width));
    y_max = clamp(y_max, 0.0f, static_cast<float>(tensor.image_height));

    Detection detection;
    detection.class_id = class_id;
    detection.class_name = kCocoLabels[static_cast<std::size_t>(class_id)];
    detection.score = score;
    detection.x = x_min;
    detection.y = y_min;
    detection.width = std::max(0.0f, x_max - x_min);
    detection.height = std::max(0.0f, y_max - y_min);
    return detection;
}

float intersection_over_union(const Detection& a, const Detection& b) {
    const float ax2 = a.x + a.width;
    const float ay2 = a.y + a.height;
    const float bx2 = b.x + b.width;
    const float by2 = b.y + b.height;

    const float ix1 = std::max(a.x, b.x);
    const float iy1 = std::max(a.y, b.y);
    const float ix2 = std::min(ax2, bx2);
    const float iy2 = std::min(ay2, by2);
    const float iw = std::max(0.0f, ix2 - ix1);
    const float ih = std::max(0.0f, iy2 - iy1);
    const float intersection = iw * ih;
    const float union_area = a.width * a.height + b.width * b.height - intersection;
    if (union_area <= 0.0f) {
        return 0.0f;
    }
    return intersection / union_area;
}

Detection weighted_merge(const std::vector<Detection>& candidates, const std::vector<std::size_t>& indices) {
    Detection merged = candidates[indices.front()];
    float total_score = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;

    for (const std::size_t index : indices) {
        const Detection& detection = candidates[index];
        const float weight = detection.score;
        total_score += weight;
        x1 += detection.x * weight;
        y1 += detection.y * weight;
        x2 += (detection.x + detection.width) * weight;
        y2 += (detection.y + detection.height) * weight;
        merged.score = std::max(merged.score, detection.score);
    }

    if (total_score > 0.0f) {
        x1 /= total_score;
        y1 /= total_score;
        x2 /= total_score;
        y2 /= total_score;
        merged.x = x1;
        merged.y = y1;
        merged.width = std::max(0.0f, x2 - x1);
        merged.height = std::max(0.0f, y2 - y1);
    }
    return merged;
}

std::vector<Detection> weighted_non_max_suppression(std::vector<Detection> candidates, std::size_t max_results) {
    std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });

    std::vector<Detection> kept;
    std::vector<bool> consumed(candidates.size(), false);

    for (std::size_t index = 0; index < candidates.size(); ++index) {
        if (consumed[index]) {
            continue;
        }

        std::vector<std::size_t> overlapping;
        for (std::size_t other = index; other < candidates.size(); ++other) {
            if (consumed[other] || candidates[other].class_id != candidates[index].class_id) {
                continue;
            }
            if (intersection_over_union(candidates[index], candidates[other]) > kNmsIouThreshold) {
                overlapping.push_back(other);
            }
        }

        for (const std::size_t matched_index : overlapping) {
            consumed[matched_index] = true;
        }

        kept.push_back(weighted_merge(candidates, overlapping));
        if (max_results > 0 && kept.size() >= max_results) {
            break;
        }
    }
    return kept;
}

std::vector<Detection> decode_detections(const TensorInput& tensor) {
    const std::vector<Anchor> anchors = generate_efficientdet_anchors(
        static_cast<int>(tensor.image_width),
        static_cast<int>(tensor.image_height)
    );
    if (anchors.size() != tensor.anchor_count) {
        std::ostringstream message;
        message << "anchor count mismatch: generated " << anchors.size()
                << ", tensor has " << tensor.anchor_count;
        throw std::runtime_error(message.str());
    }

    std::vector<Detection> candidates;
    for (std::size_t anchor_index = 0; anchor_index < tensor.anchor_count; ++anchor_index) {
        int best_class = -1;
        float best_score = -1.0f;
        const std::size_t score_base = anchor_index * tensor.class_count;

        for (std::size_t class_index = 0; class_index < tensor.class_count; ++class_index) {
            const float score = tensor.scores[score_base + class_index];
            if (score > best_score) {
                best_score = score;
                best_class = static_cast<int>(class_index);
            }
        }

        if (best_class < 0 || best_score < tensor.score_threshold) {
            continue;
        }
        if (kCocoLabels[static_cast<std::size_t>(best_class)] == "???") {
            continue;
        }

        Detection detection = decode_box(
            tensor,
            anchors[anchor_index],
            anchor_index,
            best_class,
            best_score
        );
        if (detection.width > 0.0f && detection.height > 0.0f) {
            candidates.push_back(detection);
        }
    }

    return weighted_non_max_suppression(candidates, tensor.max_results);
}

std::string json_escape(const std::string& value) {
    std::ostringstream escaped;
    for (const char ch : value) {
        switch (ch) {
            case '"':
                escaped << "\\\"";
                break;
            case '\\':
                escaped << "\\\\";
                break;
            case '\b':
                escaped << "\\b";
                break;
            case '\f':
                escaped << "\\f";
                break;
            case '\n':
                escaped << "\\n";
                break;
            case '\r':
                escaped << "\\r";
                break;
            case '\t':
                escaped << "\\t";
                break;
            default:
                escaped << ch;
                break;
        }
    }
    return escaped.str();
}

std::string build_text_summary(const TensorInput& tensor, const std::vector<Detection>& detections) {
    std::ostringstream text;
    text << "C++ tensor decoder: " << tensor.image_name << '\n';
    text << "  image_shape=" << tensor.image_width << "x" << tensor.image_height << "x" << tensor.channels << '\n';
    text << "  raw_scores_shape=" << tensor.anchor_count << "x" << tensor.class_count << '\n';
    text << "  raw_boxes_shape=" << tensor.anchor_count << "x4\n";
    text << "  box_format=xywh_pixels\n";
    text << "  score_threshold=" << std::fixed << std::setprecision(2) << tensor.score_threshold << '\n';
    text << "  detections=" << detections.size() << '\n';

    for (std::size_t index = 0; index < detections.size(); ++index) {
        const Detection& detection = detections[index];
        text << "  #" << (index + 1) << " "
             << detection.class_name
             << " (id=" << detection.class_id << ")"
             << " score=" << std::fixed << std::setprecision(4) << detection.score
             << " bbox_xywh=("
             << std::setprecision(1)
             << detection.x << ", "
             << detection.y << ", "
             << detection.width << ", "
             << detection.height << ")"
             << '\n';
    }

    if (detections.empty()) {
        text << "  no detections above threshold\n";
    }
    return text.str();
}

void print_json(const TensorInput& tensor, const std::vector<Detection>& detections) {
    const std::string text = build_text_summary(tensor, detections);

    std::cout << "{";
    std::cout << "\"image\":\"" << json_escape(tensor.image_name) << "\",";
    std::cout << "\"image_shape\":["
              << tensor.image_width << ","
              << tensor.image_height << ","
              << tensor.channels << "],";
    std::cout << "\"score_threshold\":" << std::fixed << std::setprecision(6) << tensor.score_threshold << ",";
    std::cout << "\"max_results\":" << tensor.max_results << ",";
    std::cout << "\"box_format\":\"xywh_pixels\",";
    std::cout << "\"detections\":[";

    for (std::size_t index = 0; index < detections.size(); ++index) {
        const Detection& detection = detections[index];
        if (index > 0) {
            std::cout << ",";
        }
        std::cout << "{";
        std::cout << "\"class\":\"" << json_escape(detection.class_name) << "\",";
        std::cout << "\"class_id\":" << detection.class_id << ",";
        std::cout << "\"score\":" << std::fixed << std::setprecision(6) << detection.score << ",";
        std::cout << "\"box\":{";
        std::cout << "\"x\":" << std::fixed << std::setprecision(3) << detection.x << ",";
        std::cout << "\"y\":" << detection.y << ",";
        std::cout << "\"width\":" << detection.width << ",";
        std::cout << "\"height\":" << detection.height;
        std::cout << "}}";
    }

    std::cout << "],";
    std::cout << "\"text\":\"" << json_escape(text) << "\"";
    std::cout << "}\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <tensor-file>\n";
        return 2;
    }

    try {
        const TensorInput tensor = read_tensor_file(argv[1]);
        const std::vector<Detection> detections = decode_detections(tensor);
        print_json(tensor, detections);
    } catch (const std::exception& error) {
        std::cerr << "decode_tensor error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
