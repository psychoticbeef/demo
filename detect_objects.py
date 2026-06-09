from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".cache" / "matplotlib"),
)

import numpy as np
import requests
from ai_edge_litert.interpreter import Interpreter
from PIL import Image, ImageDraw, ImageFont


MODEL_INPUT_SIZE = 320
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/"
    "efficientdet_lite0/int8/latest/efficientdet_lite0.tflite"
)
TENSOR_FILE_MAGIC = b"TDETENS1"


@dataclass(frozen=True)
class SampleImage:
    name: str
    expected_class: str
    url: str


SAMPLES = [
    SampleImage(
        name="cat",
        expected_class="cat",
        url="https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=640&h=640&fit=crop",
    ),
    SampleImage(
        name="dog",
        expected_class="dog",
        url="https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=640&h=640&fit=crop",
    ),
    SampleImage(
        name="car",
        expected_class="car",
        url="https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?w=640&h=640&fit=crop",
    ),
    SampleImage(
        name="hot_dog",
        expected_class="hot dog",
        url="https://images.unsplash.com/photo-1654851979266-dcd5655a747b?ixlib=rb-4.1.0&q=85&fm=jpg&crop=entropy&cs=srgb&w=640&h=640&fit=crop",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download square sample images, run EfficientDet-Lite0, and draw detections."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Minimum detection score to keep.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=1,
        help="Maximum detections per image.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload the model and sample images even when cached.",
    )
    parser.add_argument(
        "--show-windows",
        action="store_true",
        help="Open each annotated result in a separate desktop window.",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path, *, force: bool = False) -> None:
    if destination.exists() and not force:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "tiny-object-detector-demo/1.0"}
    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    file.write(chunk)


def prepare_sample(sample: SampleImage, raw_dir: Path, processed_dir: Path, *, force: bool) -> Path:
    raw_path = raw_dir / f"{sample.name}.jpg"
    processed_path = processed_dir / f"{sample.name}_{MODEL_INPUT_SIZE}.jpg"

    download_file(sample.url, raw_path, force=force)

    with Image.open(raw_path) as image:
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        if width != height:
            raise ValueError(
                f"{sample.name} downloaded as {width}x{height}; expected an already-square image"
            )

        resized = rgb_image.resize(
            (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            Image.Resampling.LANCZOS,
        )
        processed_dir.mkdir(parents=True, exist_ok=True)
        resized.save(processed_path, quality=95)

    return processed_path


def create_interpreter(model_path: Path) -> Interpreter:
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def run_raw_inference(interpreter: Interpreter, image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    input_details = interpreter.get_input_details()[0]
    image = Image.open(image_path).convert("RGB")
    input_tensor = np.asarray(image, dtype=np.uint8)[None, :, :, :]
    return run_raw_tensor_inference(interpreter, input_tensor, input_details)


def run_flat_input_inference(interpreter: Interpreter, flat_input_path: Path) -> tuple[np.ndarray, np.ndarray]:
    input_details = interpreter.get_input_details()[0]
    expected_shape = tuple(int(value) for value in input_details["shape"])
    expected_bytes = int(np.prod(expected_shape))
    raw_bytes = flat_input_path.read_bytes()
    if len(raw_bytes) != expected_bytes:
        raise ValueError(
            f"{flat_input_path} is {len(raw_bytes)} bytes, expected {expected_bytes}"
        )

    input_tensor = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(expected_shape)
    return run_raw_tensor_inference(interpreter, input_tensor, input_details)


def run_raw_tensor_inference(
    interpreter: Interpreter,
    input_tensor: np.ndarray,
    input_details: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:

    if tuple(input_tensor.shape) != tuple(input_details["shape"]):
        raise ValueError(
            f"input tensor shape {input_tensor.shape} does not match model shape {input_details['shape']}"
        )
    if input_details["dtype"] != np.uint8:
        raise ValueError(f"expected uint8 model input, got {input_details['dtype']}")

    interpreter.set_tensor(input_details["index"], input_tensor)
    interpreter.invoke()

    scores: np.ndarray | None = None
    boxes: np.ndarray | None = None
    for output in interpreter.get_output_details():
        tensor = interpreter.get_tensor(output["index"])[0].astype(np.float32, copy=False)
        if tensor.ndim == 2 and tensor.shape[1] == 4:
            boxes = np.ascontiguousarray(tensor)
        elif tensor.ndim == 2:
            scores = np.ascontiguousarray(tensor)

    if scores is None or boxes is None:
        raise RuntimeError("could not find raw score and box tensors in model outputs")
    if scores.shape[0] != boxes.shape[0]:
        raise RuntimeError(f"score/box anchor mismatch: {scores.shape} vs {boxes.shape}")

    return scores, boxes


def label_color(label: str) -> tuple[int, int, int]:
    palette = [
        (16, 121, 191),
        (221, 87, 70),
        (38, 154, 102),
        (169, 93, 191),
        (214, 143, 42),
    ]
    return palette[sum(label.encode("utf-8")) % len(palette)]


def draw_detections(image_path: Path, detections: list[dict[str, Any]], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for detection in detections:
        label = detection["class"]
        score = detection["score"]
        bbox = detection["box"]
        x0 = int(round(bbox["x"]))
        y0 = int(round(bbox["y"]))
        x1 = int(round(bbox["x"] + bbox["width"]))
        y1 = int(round(bbox["y"] + bbox["height"]))
        color = label_color(label)
        text = f"{label} {score:.2f}"

        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_y = max(0, y0 - text_height - 6)
        draw.rectangle(
            (x0, label_y, x0 + text_width + 8, label_y + text_height + 6),
            fill=color,
        )
        draw.text((x0 + 4, label_y + 3), text, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def ensure_cpp_decoder(root: Path) -> Path:
    source_path = root / "cpp" / "decode_tensor.cpp"
    binary_path = root / "build" / "decode_tensor"
    compiler = shutil.which("clang++") or shutil.which("c++") or shutil.which("g++")

    if compiler is None:
        raise RuntimeError("could not find clang++, c++, or g++ to build the C++ tensor decoder")
    if not source_path.exists():
        raise RuntimeError(f"missing C++ decoder source: {source_path}")

    needs_build = (
        not binary_path.exists()
        or source_path.stat().st_mtime > binary_path.stat().st_mtime
    )
    if not needs_build:
        return binary_path

    binary_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-pedantic",
            str(source_path),
            "-o",
            str(binary_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary_path


def write_tensor_file(
    tensor_path: Path,
    sample: SampleImage,
    scores: np.ndarray,
    boxes: np.ndarray,
    threshold: float,
    max_results: int,
) -> None:
    if scores.dtype != np.float32 or boxes.dtype != np.float32:
        raise ValueError("tensor file writer expects float32 scores and boxes")
    if scores.ndim != 2 or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"unexpected tensor shapes: scores={scores.shape}, boxes={boxes.shape}")
    if scores.shape[0] != boxes.shape[0]:
        raise ValueError(f"score/box anchor mismatch: {scores.shape} vs {boxes.shape}")

    image_name = sample.name.encode("utf-8")
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    with tensor_path.open("wb") as file:
        file.write(TENSOR_FILE_MAGIC)
        file.write(
            struct.pack(
                "<IIIIIIIIf",
                1,
                len(image_name),
                MODEL_INPUT_SIZE,
                MODEL_INPUT_SIZE,
                3,
                scores.shape[0],
                scores.shape[1],
                max_results,
                threshold,
            )
        )
        file.write(image_name)
        file.write(np.ascontiguousarray(scores).tobytes(order="C"))
        file.write(np.ascontiguousarray(boxes).tobytes(order="C"))


def decode_tensor_with_cpp(decoder_path: Path, tensor_path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [str(decoder_path), str(tensor_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def verify_flat_input_tensor(
    interpreter: Interpreter,
    flat_input_path: Path,
    reference_scores: np.ndarray,
    reference_boxes: np.ndarray,
) -> dict[str, Any]:
    flat_scores, flat_boxes = run_flat_input_inference(interpreter, flat_input_path)
    scores_match = bool(np.array_equal(flat_scores, reference_scores))
    boxes_match = bool(np.array_equal(flat_boxes, reference_boxes))
    max_score_delta = float(np.max(np.abs(flat_scores - reference_scores)))
    max_box_delta = float(np.max(np.abs(flat_boxes - reference_boxes)))
    return {
        "flat_input_file": str(flat_input_path),
        "byte_count": flat_input_path.stat().st_size,
        "scores_match_jpeg_path": scores_match,
        "boxes_match_jpeg_path": boxes_match,
        "max_score_delta": max_score_delta,
        "max_box_delta": max_box_delta,
    }


def build_contact_sheet(paths: list[Path], destination: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    if not images:
        return

    padding = 16
    width = (MODEL_INPUT_SIZE * len(images)) + (padding * (len(images) + 1))
    height = MODEL_INPUT_SIZE + (padding * 2)
    sheet = Image.new("RGB", (width, height), (245, 246, 248))

    x = padding
    for image in images:
        sheet.paste(image, (x, padding))
        x += MODEL_INPUT_SIZE + padding

    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=95)


def show_image_windows(paths: list[Path]) -> None:
    import tkinter as tk
    from PIL import ImageTk

    if not paths:
        return

    root = tk.Tk()
    root.withdraw()
    windows: list[tk.Toplevel] = []
    photos: list[ImageTk.PhotoImage] = []

    def window_exists(window: tk.Toplevel) -> bool:
        try:
            return bool(window.winfo_exists())
        except tk.TclError:
            return False

    def close_window(window: tk.Toplevel) -> None:
        try:
            window.destroy()
        except tk.TclError:
            pass
        if not any(window_exists(item) for item in windows):
            root.quit()

    for index, path in enumerate(paths):
        window = tk.Toplevel(root)
        window.title(path.stem.replace("_", " "))
        window.geometry(f"+{80 + index * (MODEL_INPUT_SIZE + 32)}+80")

        with Image.open(path) as source_image:
            photo = ImageTk.PhotoImage(source_image.convert("RGB"))

        label = tk.Label(window, image=photo)
        label.pack()
        window.resizable(False, False)
        window.protocol("WM_DELETE_WINDOW", lambda window=window: close_window(window))

        photos.append(photo)
        windows.append(window)

    try:
        root.mainloop()
    finally:
        root.destroy()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    model_path = root / "data" / "model" / "efficientdet_lite0.tflite"
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    flat_input_dir = root / "outputs" / "input_tensors"
    tensor_dir = root / "outputs" / "tensors"
    output_dir = root / "outputs"

    download_file(MODEL_URL, model_path, force=args.force_download)

    interpreter = create_interpreter(model_path)
    cpp_decoder_path = ensure_cpp_decoder(root)
    summary: dict[str, Any] = {
        "model": "EfficientDet-Lite0 INT8",
        "model_input_shape": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3],
        "raw_output_tensors": {
            "scores": ["anchor_count", "class_count"],
            "boxes": ["anchor_count", 4],
        },
        "decoder": "cpp/decode_tensor.cpp",
        "score_threshold": args.threshold,
        "max_results": args.max_results,
        "images": [],
    }
    annotated_paths: list[Path] = []

    for sample in SAMPLES:
        processed_path = prepare_sample(
            sample,
            raw_dir,
            processed_dir,
            force=args.force_download,
        )
        scores, boxes = run_raw_inference(interpreter, processed_path)
        flat_input_path = flat_input_dir / f"{processed_path.stem}.rgb"
        flat_input_verification = verify_flat_input_tensor(
            interpreter,
            flat_input_path,
            scores,
            boxes,
        )
        tensor_path = tensor_dir / f"{sample.name}.tensorbin"
        write_tensor_file(
            tensor_path,
            sample,
            scores,
            boxes,
            threshold=args.threshold,
            max_results=args.max_results,
        )
        decoded = decode_tensor_with_cpp(cpp_decoder_path, tensor_path)
        detections = decoded["detections"]
        annotated_path = output_dir / f"{sample.name}_detections.jpg"
        draw_detections(processed_path, detections, annotated_path)
        annotated_paths.append(annotated_path)

        summary["images"].append(
            {
                "name": sample.name,
                "expected_class": sample.expected_class,
                "raw_image": str(raw_dir / f"{sample.name}.jpg"),
                "model_input_image": str(processed_path),
                "flat_input_verification": flat_input_verification,
                "tensor_file": str(tensor_path),
                "annotated_image": str(annotated_path),
                "detections": detections,
                "cpp_decoder_output": decoded["text"],
            }
        )

    grid_path = output_dir / "detections_grid.jpg"
    build_contact_sheet(annotated_paths, grid_path)
    summary["contact_sheet"] = str(grid_path)

    json_path = output_dir / "detections.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        textwrap.dedent(
            f"""
            Wrote:
              {grid_path}
              {json_path}
            """
        ).strip()
    )
    for image in summary["images"]:
        labels = ", ".join(
            f"{item['class']}={item['score']:.2f}" for item in image["detections"]
        )
        print(f"{image['name']}: {labels or 'no detections above threshold'}")
        verification = image["flat_input_verification"]
        print(
            "  flat input: "
            f"{verification['byte_count']} bytes, "
            f"scores_match={verification['scores_match_jpeg_path']}, "
            f"boxes_match={verification['boxes_match_jpeg_path']}"
        )
        print(textwrap.indent(image["cpp_decoder_output"], "  "))

    if args.show_windows:
        show_image_windows(annotated_paths)


if __name__ == "__main__":
    main()
