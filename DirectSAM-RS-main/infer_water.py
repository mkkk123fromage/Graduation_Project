import argparse
import os
import os.path as osp
from ctypes import POINTER, c_float, c_int, cdll
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torch import nn
from tqdm import tqdm
from transformers import AutoImageProcessor

from impl.toolbox import conv_tri, grad2
from model import SegformerWithTextFusion


def safe_prompt_key(water_type, environment):
    return f"{water_type}_{environment}".strip().lower().replace(" ", "_")


def load_nms_solver(lib_path):
    solver = cdll.LoadLibrary(str(lib_path))
    c_float_pointer = POINTER(c_float)
    solver.nms.argtypes = [c_float_pointer, c_float_pointer, c_float_pointer, c_int, c_int, c_float, c_int, c_int]
    return solver, c_float_pointer


def nms_process_one_image(image, save_path, solver, c_float_pointer):
    edge = conv_tri(image, 1)
    edge = np.float32(edge)
    ox, oy = grad2(conv_tri(edge, 4))
    oxx, _ = grad2(ox)
    oxy, oyy = grad2(oy)
    ori = np.mod(np.arctan(oyy * np.sign(-oxy) / (oxx + 1e-5)), np.pi)
    out = np.zeros_like(edge)
    r, s, m, w, h = 1, 5, float(1.01), int(out.shape[1]), int(out.shape[0])
    solver.nms(
        out.ctypes.data_as(c_float_pointer),
        edge.ctypes.data_as(c_float_pointer),
        ori.ctypes.data_as(c_float_pointer),
        r, s, m, w, h,
    )
    edge = np.round(out * 255).astype(np.uint8)
    cv2.imwrite(str(save_path), edge)


def nms_process(result_dir, save_dir, solver, c_float_pointer):
    nms_dir = Path(save_dir) / "nms"
    nms_dir.mkdir(parents=True, exist_ok=True)
    for file in tqdm(list(Path(result_dir).glob("*.npy")), desc="NMS"):
        save_name = nms_dir / f"{file.stem}.png"
        if save_name.exists():
            continue
        image = np.load(file)
        nms_process_one_image(image, save_name, solver, c_float_pointer)


def process_one_image(image, image_processor, model, word_mask, local_text_feature, device, resolution):
    image_processor.size["height"] = resolution
    image_processor.size["width"] = resolution

    encoding = image_processor(image, return_tensors="pt")
    pixel_values = encoding.pixel_values.to(device)
    word_mask = word_mask.to(device).unsqueeze(1)
    local_text_feature = local_text_feature.to(device).unsqueeze(1)

    with torch.no_grad():
        outputs = model(
            pixel_values=pixel_values,
            word_mask=word_mask,
            local_text_feature=local_text_feature,
        )

    logits = outputs.logits.float().cpu()
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    return torch.sigmoid(upsampled_logits)[0, 0].detach().numpy()


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def assert_checkpoint_complete(checkpoint):
    checkpoint = Path(checkpoint)
    candidates = [
        checkpoint / "pytorch_model.bin",
        checkpoint / "model.safetensors",
        checkpoint / "tf_model.h5",
    ]
    if not any(path.exists() for path in candidates):
        names = ", ".join(path.name for path in candidates)
        raise FileNotFoundError(
            f"模型权重缺失：{checkpoint} 中没有 {names}。"
            "官方 GitHub 仓库当前只提供 config.json，需要另行取得 DirectSAM-RS 的训练权重。"
        )


def main():
    parser = argparse.ArgumentParser(description="Water extraction inference with water-type/environment prompts.")
    parser.add_argument("--input-dir", default="./test_origin", help="Input image folder.")
    parser.add_argument("--output-dir", default="./infer_result_water", help="Output folder.")
    parser.add_argument("--checkpoint", default="./weight", help="DirectSAM-RS HuggingFace checkpoint folder.")
    parser.add_argument("--text-features", default="./water_text_feature.pth", help="Generated water text feature file.")
    parser.add_argument("--water-type", default="river", choices=["river", "lake", "pond", "reservoir"])
    parser.add_argument("--environment", default="urban", choices=["urban", "farmland", "forest", "grassland", "desert"])
    parser.add_argument("--prompt-key", default=None, help="Override key in text feature file, e.g. water or all_waters.")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--png", action="store_true", help="Save raw probability as uint8 png instead of npy.")
    parser.add_argument("--nms", action="store_true", help="Run C++ NMS on saved npy outputs.")
    parser.add_argument("--nms-lib", default="./cxx/lib/solve_csa.so")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assert_checkpoint_complete(args.checkpoint)
    if not Path(args.text_features).exists():
        raise FileNotFoundError(
            f"文本特征文件缺失：{args.text_features}。请先运行 build_water_text_features.py 生成。"
        )

    prompt_key = args.prompt_key or safe_prompt_key(args.water_type, args.environment)
    local_dict = torch.load(args.text_features, map_location="cpu")
    feature_key = f"{prompt_key}_feature"
    mask_key = f"{prompt_key}_mask"
    if feature_key not in local_dict or mask_key not in local_dict:
        available = sorted(k[:-8] for k in local_dict if k.endswith("_feature"))
        raise KeyError(f"找不到提示 {prompt_key}。可用提示包括：{available}")

    device = resolve_device(args.device)
    image_processor = AutoImageProcessor.from_pretrained(args.checkpoint, reduce_labels=True)
    model = SegformerWithTextFusion.from_pretrained(args.checkpoint, ignore_mismatched_sizes=True)
    model.to(device)
    model.eval()

    image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    files = [p for p in input_dir.iterdir() if p.suffix.lower() in image_exts]
    for image_path in tqdm(files, desc=f"Infer {prompt_key}"):
        image = Image.open(image_path).convert("RGB")
        probs = process_one_image(
            image,
            image_processor,
            model,
            word_mask=local_dict[mask_key],
            local_text_feature=local_dict[feature_key],
            device=device,
            resolution=args.resolution,
        )
        if args.png:
            cv2.imwrite(str(output_dir / image_path.name), (probs * 255.0).astype(np.uint8))
        else:
            np.save(output_dir / f"{image_path.stem}.npy", probs)

    if args.nms:
        solver, c_float_pointer = load_nms_solver(args.nms_lib)
        nms_process(output_dir, output_dir, solver, c_float_pointer)


if __name__ == "__main__":
    main()
