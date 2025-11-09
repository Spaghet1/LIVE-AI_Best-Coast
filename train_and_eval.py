#!/usr/bin/env python3
"""
Simple OCR-based receipt field extractor and evaluator.

- Reads images from train_data/images
- Reads ground-truth JSON from train_data/gdt (one JSON per image stem)
- Predicts: company, date, address, total
- Evaluates predictions against ground truth with exact match and fuzzy ratio

This is a minimal baseline that mirrors some simple preprocessing ideas from EDA.ipynb
(grayscale reading via cv2.imdecode) and uses pytesseract OCR + regex/rules.

Usage examples:
  python train_and_eval.py predict --out eda_outputs/predictions.jsonl
  python train_and_eval.py eval --pred eda_outputs/predictions.jsonl

Dependencies:
  - pillow, opencv-python, pytesseract (optional but recommended)

If pytesseract is not installed or Tesseract binary is missing, the script will
raise a clear error. On macOS you can install Tesseract via:
  brew install tesseract
"""
from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Optional imports with graceful fallback
try:
    import cv2
except Exception:
    cv2 = None  # type: ignore

try:
    import pytesseract
except Exception:
    pytesseract = None  # type: ignore

from difflib import SequenceMatcher

ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / 'train_data' / 'images'
GDT_DIR = ROOT / 'train_data' / 'gdt'

IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']


def read_gray(path: Path) -> Optional[np.ndarray]:
    """Robust grayscale reader similar to EDA.ipynb _read_gray."""
    if cv2 is None:
        # Fallback via PIL -> numpy
        try:
            im = Image.open(path).convert('L')
            return np.array(im)
        except Exception:
            return None
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def ocr_image(path: Path) -> str:
    """Run OCR using pytesseract. Apply light pre-processing to improve legibility."""
    if pytesseract is None:
        raise RuntimeError("pytesseract is not installed. Install with: pip install pytesseract; also install Tesseract binary (e.g., brew install tesseract)")
    img_gray = read_gray(path)
    if img_gray is None:
        # last resort: let pytesseract open
        text = pytesseract.image_to_string(Image.open(path))
        return text

    # Simple normalization similar to typical OCR pre-processing
    # - CLAHE for contrast (if OpenCV available)
    # - Adaptive thresholding
    processed = img_gray
    if cv2 is not None:
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(processed)
            processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                              cv2.THRESH_BINARY, 31, 10)
        except Exception:
            pass
    text = pytesseract.image_to_string(processed)
    # Normalize whitespace
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\t", " ", text)
    text = re.sub(r"[ ]+", " ", text)
    # Keep original line breaks
    return text


DATE_PATTERNS = [
    # 27/MAR/2018, 27/Mar/2018, 27-MAR-2018
    r"\b(\d{1,2})[\-/\\](Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[A-Za-z]*[\-/\\](\d{2,4})\b",
    r"\b(\d{1,2})[\-/\\](JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Za-z]*[\-/\\](\d{2,4})\b",
    # 2018-03-27, 2018/03/27
    r"\b(\d{4})[\-/](\d{1,2})[\-/](\d{1,2})\b",
    # 03/27/2018, 3/7/18
    r"\b(\d{1,2})[\-/](\d{1,2})[\-/](\d{2,4})\b",
]

CURRENCY_RE = re.compile(r"(?<![\d])\$?\s*([0-9]{1,3}(?:,[0-9]{3})*|[0-9]+)(?:\.[0-9]{2})?\b")

ADDRESS_KEYWORDS = [
    'street', 'st', 'st.', 'road', 'rd', 'rd.', 'ave', 'avenue', 'blvd', 'dr', 'drive', 'ln', 'lane', 'wy', 'way',
    'no.', 'jalan', 'lot', 'suite', 'unit', 'floor', 'fl', 'hwy', 'highway', 'parkway', 'pkwy', 'city', 'state', 'zip', 'selangor'
]

COMPANY_STOPWORDS = [
    'receipt', 'tax', 'invoice', 'subtotal', 'total', 'change', 'cash', 'credit', 'debit', 'gst', 'vat', 'thank', 'you'
]


def extract_date(text: str) -> str:
    for pat in DATE_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(0).strip()
    return ""


def extract_total(text: str) -> str:
    # Prefer lines containing 'total'
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    total_candidates: List[Tuple[float, str]] = []

    for ln in lines:
        if re.search(r"total\b", ln, flags=re.IGNORECASE):
            nums = re.findall(r"\$?\s*([0-9]{1,3}(?:,[0-9]{3})*|[0-9]+(?:\.[0-9]{2}))", ln)
            for n in nums[::-1]:
                try:
                    val = float(n.replace(',', ''))
                    total_candidates.append((val, n))
                except Exception:
                    pass
    if total_candidates:
        # Choose the rightmost/last numeric on TOTAL line, or max by value if multiple lines
        best = max(total_candidates, key=lambda x: x[0])
        return f"{best[0]:.2f}" if '.' in best[1] else best[1]

    # Otherwise choose max currency-like number in whole text
    values: List[Tuple[float, str]] = []
    for m in CURRENCY_RE.finditer(text):
        s = m.group(1)
        try:
            val = float(s.replace(',', ''))
            values.append((val, s))
        except Exception:
            continue
    if values:
        best = max(values, key=lambda x: x[0])
        # standardize to 2 decimals if original had decimals
        return f"{best[0]:.2f}"
    return ""


def extract_company(text: str) -> str:
    lines = [re.sub(r"[^A-Za-z0-9 .,&()'/-]", "", l).strip() for l in text.splitlines()]
    lines = [l for l in lines if l]
    # Heuristic: first line near top that looks like a business name: mostly letters, few digits, not a stopword
    for l in lines[:15]:
        lc = l.lower()
        if any(sw in lc for sw in COMPANY_STOPWORDS):
            continue
        # If line has letters and at most 4 digits total, and not too short
        letters = sum(ch.isalpha() for ch in l)
        digits = sum(ch.isdigit() for ch in l)
        if letters >= 4 and digits <= 4 and len(l) >= 4:
            # Prefer lines with uppercase words
            if sum(1 for w in l.split() if w.isupper()) >= 1 or letters > digits * 2:
                return l
    # fallback: first non-empty line
    return lines[0] if lines else ""


def extract_address(text: str) -> str:
    lines = [l.strip() for l in text.splitlines()]
    # Normalize commas and spaces
    norm_lines = [re.sub(r"\s+", " ", l) for l in lines if l]
    # Find start index of address using keywords
    idx = None
    for i, l in enumerate(norm_lines):
        ll = l.lower()
        if any(k in ll for k in ADDRESS_KEYWORDS):
            idx = i
            break
    if idx is None:
        return ""
    # Take a window of 1-3 lines around keyword-rich line
    chunk = [norm_lines[idx]]
    for j in range(idx+1, min(idx+4, len(norm_lines))):
        lj = norm_lines[j].lower()
        if any(k in lj for k in ADDRESS_KEYWORDS) or (len(norm_lines[j]) > 10 and not norm_lines[j].isdigit()):
            chunk.append(norm_lines[j])
        else:
            break
    # Join and clean
    addr = ", ".join(chunk)
    # Remove repeated commas/spaces
    addr = re.sub(r"\s*,\s*", ", ", addr)
    addr = re.sub(r"\s+", " ", addr).strip()
    return addr


def extract_fields(text: str) -> Dict[str, str]:
    return {
        'company': extract_company(text),
        'date': extract_date(text),
        'address': extract_address(text),
        'total': extract_total(text),
    }


def find_image_for_stem(stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        p = IMG_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    # also scan subdirs (if any)
    if IMG_DIR.exists():
        for root, _, files in os.walk(IMG_DIR):
            for fn in files:
                name, ext = os.path.splitext(fn)
                if name == stem and ext.lower() in IMAGE_EXTS:
                    return Path(root) / fn
    return None


def load_ground_truth() -> Dict[str, Dict[str, str]]:
    gts: Dict[str, Dict[str, str]] = {}
    if not GDT_DIR.exists():
        return gts
    for fp in sorted(GDT_DIR.glob('*.json')):
        try:
            data = json.loads(fp.read_text())
            # keys: company, date, address, total
            stem = fp.stem
            gts[stem] = {k: str(v) for k, v in data.items() if k in ('company', 'date', 'address', 'total')}
        except Exception:
            continue
    return gts


def fuzzy_ratio(a: str, b: str) -> float:
    a = (a or '').strip()
    b = (b or '').strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(a=a.lower(), b=b.lower()).ratio()


def evaluate(preds: Dict[str, Dict[str, str]], gts: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    fields = ['company', 'date', 'address', 'total']
    totals = {f: {'exact': 0, 'count': 0, 'fuzzy': 0.0} for f in fields}
    paired = set(preds.keys()) & set(gts.keys())
    for k in sorted(paired):
        pred = preds.get(k, {})
        gt = gts.get(k, {})
        for f in fields:
            totals[f]['count'] += 1
            if (pred.get(f, '').strip() == gt.get(f, '').strip()) and gt.get(f, '') != '':
                totals[f]['exact'] += 1
            totals[f]['fuzzy'] += fuzzy_ratio(pred.get(f, ''), gt.get(f, ''))
    # aggregate metrics
    out = {}
    for f in fields:
        c = totals[f]['count'] or 1
        out[f] = {
            'exact_acc': totals[f]['exact'] / c,
            'avg_fuzzy': totals[f]['fuzzy'] / c,
            'n': c,
        }
    # overall average
    overall_exact = sum(totals[f]['exact'] for f in fields) / (sum(totals[f]['count'] for f in fields) or 1)
    overall_fuzzy = sum(totals[f]['fuzzy'] for f in fields) / (sum(totals[f]['count'] for f in fields) or 1)
    out['overall'] = {'exact_acc': overall_exact, 'avg_fuzzy': overall_fuzzy, 'n_fields': sum(totals[f]['count'] for f in fields)}
    return out


def predict_all(debug: bool = False) -> Dict[str, Dict[str, str]]:
    gts = load_ground_truth()
    stems = sorted(gts.keys())
    results: Dict[str, Dict[str, str]] = {}
    for stem in stems:
        img_path = find_image_for_stem(stem)
        if img_path is None:
            if debug:
                print(f"Warning: image not found for {stem}")
            continue
        try:
            text = ocr_image(img_path)
            fields = extract_fields(text)
            results[stem] = fields
            if debug:
                print(f"{stem}: {fields}")
        except Exception as e:
            if debug:
                print(f"Error on {stem}: {e}")
    return results


def save_jsonl(preds: Dict[str, Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for k, v in sorted(preds.items()):
            rec = {'id': k}
            rec.update(v)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_jsonl(pred_path: Path) -> Dict[str, Dict[str, str]]:
    preds: Dict[str, Dict[str, str]] = {}
    with pred_path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            k = str(rec.get('id'))
            preds[k] = {fld: str(rec.get(fld, '')) for fld in ('company', 'date', 'address', 'total')}
    return preds


def main(argv: Optional[List[str]] = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description='Simple OCR baseline for receipts field extraction')
    # Make subcommand optional; show help if missing for a friendlier UX
    sub = parser.add_subparsers(dest='cmd')

    p_pred = sub.add_parser('predict', help='Run OCR and field extraction over evaluation set (by gdt JSON)')
    p_pred.add_argument('--out', type=str, default=str(ROOT / 'eda_outputs' / 'predictions.jsonl'))
    p_pred.add_argument('--debug', action='store_true')

    p_eval = sub.add_parser('eval', help='Evaluate predictions against ground truth')
    p_eval.add_argument('--pred', type=str, default=str(ROOT / 'eda_outputs' / 'predictions.jsonl'))

    args = parser.parse_args(argv)

    if args.cmd is None:
        parser.print_help()
        print("\nExamples:\n  python train_and_eval.py predict --out eda_outputs/predictions.jsonl --debug\n  python train_and_eval.py eval --pred eda_outputs/predictions.jsonl")
        return

    if args.cmd == 'predict':
        preds = predict_all(debug=bool(args.debug))
        save_jsonl(preds, Path(args.out))
        print(f"Saved predictions for {len(preds)} items to {args.out}")
    elif args.cmd == 'eval':
        pred_path = Path(args.pred)
        if not pred_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {pred_path}")
        preds = load_jsonl(pred_path)
        gts = load_ground_truth()
        metrics = evaluate(preds, gts)
        print(json.dumps(metrics, indent=2))

def testTesseract():
    import pytesseract

    try:
        version = pytesseract.get_tesseract_version()
        print("Tesseract is installed:", version)
    except pytesseract.TesseractNotFoundError:
        print("Tesseract OCR is not installed or not in PATH")

if __name__ == '__main__':
    main()
