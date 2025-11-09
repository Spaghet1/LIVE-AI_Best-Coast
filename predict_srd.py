#!/usr/bin/env python3
"""
Run the existing OCR-based field extractor on images in large-receipt-image-dataset-SRD.

This script reuses the OCR + rule-based extractor from train_and_eval.py and
applies it to an arbitrary directory of images (defaulting to the provided
large-receipt-image-dataset-SRD folder). Results are written as JSONL with
one record per image:

  {"id": "<image-stem>", "company": "...", "date": "...", "address": "...", "total": "..."}

Usage examples:
  python predict_srd.py
  python predict_srd.py --input-dir large-receipt-image-dataset-SRD --out eda_outputs/srd_predictions.jsonl --debug

Dependencies:
  - pillow, opencv-python, pytesseract
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

# Reuse OCR and field extraction from the baseline
from train_and_eval import ocr_image, extract_fields, IMAGE_EXTS  # type: ignore

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT / 'large-receipt-image-dataset-SRD'
DEFAULT_OUTPUT = ROOT / 'eda_outputs' / 'srd_predictions.jsonl'


def find_images(input_dir: Path) -> List[Path]:
    imgs: List[Path] = []
    for ext in IMAGE_EXTS:
        imgs.extend(sorted(input_dir.glob(f'*{ext}')))
        imgs.extend(sorted(input_dir.glob(f'*{ext.upper()}')))
    return imgs


def predict_on_dir(input_dir: Path, debug: bool = False) -> Dict[str, Dict[str, str]]:
    results: Dict[str, Dict[str, str]] = {}
    images = find_images(input_dir)
    if debug:
        print(f"Found {len(images)} images in {input_dir}")
    for img_path in images:
        stem = img_path.stem
        try:
            text = ocr_image(img_path)
            fields = extract_fields(text)
            results[stem] = fields
            if debug:
                print(f"{stem}: {fields}")
        except Exception as e:
            if debug:
                print(f"Error processing {img_path}: {e}")
    return results


def save_jsonl(preds: Dict[str, Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for k, v in sorted(preds.items()):
            rec = {'id': k}
            rec.update(v)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main(argv: Optional[List[str]] = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description='Run OCR field extraction on the SRD image directory')
    parser.add_argument('--input-dir', type=str, default=str(DEFAULT_INPUT_DIR), help='Directory containing receipt images')
    parser.add_argument('--out', type=str, default=str(DEFAULT_OUTPUT), help='Path to write JSONL predictions')
    parser.add_argument('--debug', action='store_true', help='Verbose logging')
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    preds = predict_on_dir(input_dir, debug=bool(args.debug))
    save_jsonl(preds, Path(args.out))
    print(f"Saved predictions for {len(preds)} images to {args.out}")


if __name__ == '__main__':
    main()
