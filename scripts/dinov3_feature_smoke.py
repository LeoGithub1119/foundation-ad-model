from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test DINOv3 feature extraction on one image.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_processor = AutoImageProcessor.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(args.model_path, local_files_only=True).to(args.device)
    model.eval()

    with Image.open(args.image) as image:
        image = image.convert("RGB")
        inputs = image_processor(images=image, return_tensors="pt")
    inputs = {key: value.to(args.device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    tokens = outputs.last_hidden_state
    cls = tokens[:, 0]
    num_register_tokens = int(getattr(model.config, "num_register_tokens", 0))
    patches = tokens[:, 1 + num_register_tokens :]

    print(f"image={args.image}")
    print(f"pixel_values={tuple(inputs['pixel_values'].shape)}")
    print(f"last_hidden_state={tuple(tokens.shape)}")
    print(f"cls={tuple(cls.shape)}")
    print(f"patch_tokens={tuple(patches.shape)}")
    print(f"num_register_tokens={num_register_tokens}")


if __name__ == "__main__":
    main()
