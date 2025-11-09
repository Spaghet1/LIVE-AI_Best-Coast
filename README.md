Project for LIVE AI Best Coast Hackathon

Simple OCR baseline to read receipt fields (company, date, address, total) from images using the data and processing outlined in EDA.ipynb.

Data layout:
- train_data/images: input receipt images
- train_data/gdt: ground-truth JSON per image (same stem name), with keys: company, date, address, total
- https://www.kaggle.com/datasets/dhiaznaidi/receiptdatasetssd300v2/data
- large-receipt-image-dataset-SRD: dataset to test model on
- https://expressexpense.com/blog/free-receipt-images-ocr-machine-learning-dataset/

Quick start:
1) Install dependencies (Python 3.9+):
   - pip install pillow opencv-python pytesseract
   - Also install the Tesseract OCR binary (macOS: brew install tesseract)

2) Run predictions over the evaluation set (based on JSONs in train_data/gdt):
   - python train_and_eval.py predict --out eda_outputs/predictions.jsonl --debug

3) Evaluate predictions against ground truth:
   - python train_and_eval.py eval --pred eda_outputs/predictions.jsonl

Notes:
- The script applies grayscale reading similar to EDA.ipynb (cv2.imdecode fallback to PIL), simple contrast/thresholding, then pytesseract OCR.
- Field extraction uses simple heuristics and regexes:
  - date: common date patterns (e.g., 27/MAR/2018, 2018-03-27, 03/27/2018)
  - total: value on a line containing the word "TOTAL" or the maximum currency-like number in the text
  - company: likely first business-like line near the top excluding common receipt stopwords
  - address: lines near keywords like "street", "road", "jalan", "suite", etc.

Outputs:
- eda_outputs/predictions.jsonl: one JSON per line with fields {id, company, date, address, total}
- The eval command prints exact-match accuracy and average fuzzy similarity (0-1) per field and overall.

