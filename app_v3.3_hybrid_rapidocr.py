# app_v3.3_hybrid_rapidocr.py
# ==== Lite Checker™ Hybrid (RapidOCR detect-only + Tesseract recognise) ====
BUILD_TAG = "LiteChecker-HYBRID (RapidOCR) v3.3 — 2025-10-09"
print(f">>> USING {BUILD_TAG} <<<")

import streamlit as st
import numpy as np
import cv2
import pytesseract
from pytesseract import Output
from rapidocr_onnxruntime import RapidOCR
from skimage.filters import sobel
from skimage.measure import shannon_entropy

# ---------------- Page / Controls ----------------
st.set_page_config(page_title=f"Lite Checker™ — {BUILD_TAG}", layout="centered")
st.sidebar.header("Lite Checker™ Controls")

# Contrast threshold toggle (AA vs AAA)
contrast_choice = st.sidebar.radio(
    "Contrast threshold",
    ["AA (4.5:1)", "AAA (7.0:1)"],
    index=0
)
AA_THRESHOLD = 4.5 if "AA" in contrast_choice else 7.0
AAA_THRESHOLD = 7.0
CONF_THRESHOLD = 40.0  # OCR confidence floor (0..100)

# Style-split sensitivity (NatWest-like grouping)
STYLE_SPLIT_THRESHOLD = st.sidebar.slider(
    "Style split sensitivity",
    min_value=2.0, max_value=12.0, value=6.0, step=0.5,
    help="Lower = more sensitive (more blocks). Higher = merges more lines."
)

# ---------------- WCAG & helpers ----------------
def _srgb_to_lum_255(v: float) -> float:
    s = v/255.0
    return s/12.92 if s <= 0.03928 else ((s+0.055)/1.055)**2.4

def wcag_contrast_gray(a: float, b: float) -> float:
    L1, L2 = sorted([_srgb_to_lum_255(a), _srgb_to_lum_255(b)], reverse=True)
    return (L1 + 0.05) / (L2 + 0.05)

def directional_bg_sample(gray_full, box, pad_ratio=0.6):
    x, y, w, h = box
    H, W = gray_full.shape[:2]
    pad = max(5, int(pad_ratio * h))
    y0, y1 = max(0, y - pad), min(H, y + h + pad)
    x0, x1 = x, x + w
    upper = gray_full[y0:y, x0:x1]
    lower = gray_full[y + h:y1, x0:x1]
    band = np.concatenate([upper.flatten(), lower.flatten()]) if upper.size + lower.size > 0 else np.array([])
    if band.size == 0:
        return None
    k = max(10, len(band) // 3)
    sorted_bg = np.sort(band)
    bg_dark = float(np.median(sorted_bg[:k]))
    bg_light = float(np.median(sorted_bg[-k:]))
    return bg_dark, bg_light

def word_fg_candidates(gray_roi):
    edges = cv2.Canny(gray_roi, 80, 180)
    p = gray_roi[edges > 0] if np.sum(edges) > 0 else gray_roi.flatten()
    if p.size == 0:
        return None, None
    p_sorted = np.sort(p)
    k = max(1, len(p_sorted) // 10)
    fg_dark = float(np.median(p_sorted[:k]))
    fg_light = float(np.median(p_sorted[-k:]))
    return fg_dark, fg_light

def word_contrast(gray_full, box):
    x, y, w, h = box
    roi = gray_full[y:y + h, x:x + w]
    fgd, fgl = word_fg_candidates(roi)
    bgpair = directional_bg_sample(gray_full, box, 0.6)
    if None in (fgd, fgl) or bgpair is None:
        return None
    bgd, bgl = bgpair
    c_dark = wcag_contrast_gray(fgd, bgl)   # dark text on light bg
    c_light = wcag_contrast_gray(fgl, bgd)  # light text on dark bg
    return float(max(c_dark, c_light))

def edge_entropy_sharpness(gray_full, box):
    x, y, w, h = box
    roi = gray_full[y:y + h, x:x + w]
    ent = float(shannon_entropy(roi)) if roi.size else 0.0
    shp = float(np.mean(sobel(roi))) if roi.size else 0.0
    return ent, shp

def verdict_color_from_ratio(r):
    if r is None or r < 3.0:
        return "🔴 Fail"
    elif r < AA_THRESHOLD:
        return "🟡 Borderline (AA Large only)"
    elif r < AAA_THRESHOLD:
        return "🟡 Pass AA"
    else:
        return "🟢 Pass AAA"

# ---------------- RapidOCR detect-only ----------------
@st.cache_resource(show_spinner=False)
def get_detector():
    # CPU inference; angle classification on; no CUDA on Streamlit Cloud
    return RapidOCR(det_use_cuda=False, rec_use_cuda=False, use_angle_cls=True)

def detect_boxes(img_bgr):
    """
    Use RapidOCR to find text polygons; convert to axis-aligned boxes;
    merge adjacent boxes that lie on the same line.
    """
    ocr = get_detector()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxes, _, _ = ocr(img_rgb)  # boxes = list of polygons [[x,y], ...]
    rects = []
    for poly in boxes or []:
        pts = np.array(poly).astype(int)
        x, y = np.min(pts[:, 0]), np.min(pts[:, 1])
        x2, y2 = np.max(pts[:, 0]), np.max(pts[:, 1])
        rects.append((int(x), int(y), int(x2 - x), int(y2 - y)))

    # Merge horizontally to form lines
    rects = sorted(rects, key=lambda b: (b[1], b[0]))
    merged = []
    for b in rects:
        if not merged:
            merged.append(b); continue
        x, y, w, h = b
        X, Y, W, H = merged[-1]
        same_line = abs((y + h/2) - (Y + H/2)) < max(h, H) * 0.6 and (x <= X + W + 30)
        if same_line:
            nx, ny = min(x, X), min(y, Y)
            nx2, ny2 = max(x + w, X + W), max(y + h, Y + H)
            merged[-1] = (nx, ny, nx2 - nx, ny2 - ny)
        else:
            merged.append(b)
    return merged

# ---------------- Recognise (Tesseract) within detected regions ----------------
def tesseract_words_in_boxes(img_bgr, boxes):
    """Run Tesseract per region; return word dicts with absolute coords + conf."""
    words = []
    for li, (x, y, w, h) in enumerate(boxes, start=1):
        roi = img_bgr[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        variants = [
            gray,
            cv2.bitwise_not(gray),
            cv2.addWeighted(gray, 1.8, cv2.GaussianBlur(gray, (0,0), 1.2), -0.8, 0),
            cv2.bitwise_not(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY, 31, 5))
        ]
        for v in variants:
            data = pytesseract.image_to_data(v, output_type=Output.DICT, config="--oem 3 --psm 6")
            n = len(data['text'])
            for i in range(n):
                t = data['text'][i]
                if not t or not t.strip():
                    continue
                try:
                    conf = float(data['conf'][i])
                except:
                    conf = 0.0
                wx, wy = int(data['left'][i]), int(data['top'][i])
                ww, wh = int(data['width'][i]), int(data['height'][i])
                words.append({
                    "text": t.strip(),
                    "conf": conf,
                    "left": x + wx,
                    "top":  y + wy,
                    "width": ww,
                    "height": wh,
                    "line_num": li
                })
    # de-dup near-identical word boxes
    deduped = []
    for w in sorted(words, key=lambda x: (x['top'], x['left'])):
        duplicate = False
        for k in deduped:
            if w['text'] == k['text'] and abs(w['left'] - k['left']) < 6 and abs(w['top'] - k['top']) < 6:
                duplicate = True; break
        if not duplicate:
            deduped.append(w)
    return deduped

# ---------------- Group, style segmentation, scoring ----------------
def group_by_line(words):
    lines = {}
    for w in words:
        lines.setdefault(w['line_num'], []).append(w)
    blocks = []
    for ln in sorted(lines.keys()):
        ws = sorted(lines[ln], key=lambda z: z['left'])
        xs = [w['left'] for w in ws]
        ys = [w['top'] for w in ws]
        ws_ = [w['width'] for w in ws]
        hs_ = [w['height'] for w in ws]
        x, y = min(xs), min(ys)
        w_box = max([xs[i] + ws_[i] for i in range(len(xs))]) - x
        h_box = max([ys[i] + hs_[i] for i in range(len(ys))]) - y
        blocks.append({"line_num": ln, "words": ws, "bbox": (x, y, w_box, h_box)})
    return blocks

def directional_bg_lum(gray_full, bbox):
    pair = directional_bg_sample(gray_full, bbox, 0.6)
    if pair is None: return 0.0
    _, bg_light = pair
    return float(bg_light)

def line_features(gray_full, block):
    x, y, w, h = block["bbox"]
    roi = gray_full[y:y + h, x:x + w]
    H, W = gray_full.shape[:2]
    size_pct = (h / H) * 100.0
    edges = cv2.Canny(roi, 80, 180)
    stroke_density = float(np.count_nonzero(edges)) / float(roi.size) if roi.size else 0.0
    bg_lum = directional_bg_lum(gray_full, block["bbox"])
    return {"size_pct": size_pct, "stroke_density": stroke_density, "bg_lum": bg_lum}

def style_distance(f1, f2):
    ds = abs(f1["size_pct"] - f2["size_pct"]) / 2.0
    dw = abs(f1["stroke_density"] - f2["stroke_density"]) * 100.0
    db = abs(f1["bg_lum"] - f2["bg_lum"]) / 10.0
    return ds + dw + db

def segment_blocks_by_style(gray_full, line_blocks, thresh=STYLE_SPLIT_THRESHOLD):
    if not line_blocks:
        return []
    feats = [line_features(gray_full, b) for b in line_blocks]
    style_blocks = []
    current = {"lines": [line_blocks[0]], "feats": [feats[0]]}
    for i in range(1, len(line_blocks)):
        d = style_distance(feats[i - 1], feats[i])
        if d > thresh:
            style_blocks.append(current)
            current = {"lines": [line_blocks[i]], "feats": [feats[i]]}
        else:
            current["lines"].append(line_blocks[i])
            current["feats"].append(feats[i])
    style_blocks.append(current)

    # collapse to merged blocks (concat text + union bbox)
    merged = []
    for sb in style_blocks:
        lines = sb["lines"]
        xs = [b["bbox"][0] for b in lines]
        ys = [b["bbox"][1] for b in lines]
        xe = [b["bbox"][0] + b["bbox"][2] for b in lines]
        ye = [b["bbox"][1] + b["bbox"][3] for b in lines]
        x, y, w, h = min(xs), min(ys), max(xe) - min(xs), max(ye) - min(ys)
        words = []
        for b in lines:
            words.extend(sorted(b["words"], key=lambda z: (b["bbox"][1], z["left"])))
        merged.append({"bbox": (x, y, w, h), "words": words})
    return merged

def verdict_color_from_ratio_block(r):
    if r is None or r < 3.0:
        return "🔴 Fail"
    elif r < AA_THRESHOLD:
        return "🟡 Borderline (AA Large only)"
    elif r < AAA_THRESHOLD:
        return "🟡 Pass AA"
    else:
        return "🟢 Pass AAA"

def score_block(img, gray, block):
    x, y, w, h = block['bbox']
    H, W = gray.shape[:2]
    size_pct = round((h / H) * 100.0, 2)

    readable_words, unreadable_words = [], []
    contrast_samples = []

    for wd in block['words']:
        bx = (wd['left'], wd['top'], wd['width'], wd['height'])
        r = word_contrast(gray, bx)
        conf = float(wd['conf'])
        txt = wd['text']
        if r is not None:
            contrast_samples.append(r)
        if (r is None) or (r < AA_THRESHOLD) or (conf < CONF_THRESHOLD):
            unreadable_words.append(txt)
        else:
            readable_words.append(txt)

    block_contrast = round(float(np.median(contrast_samples)), 2) if contrast_samples else None
    ent, shp = edge_entropy_sharpness(gray, block['bbox'])
    entropy_flag = "🟢 Low" if ent < 4 else ("🟡 Moderate" if ent < 6 else "🔴 High (cluttered)")
    sharp_flag = "🔴 Soft / blurred" if shp < 0.01 else ("🟡 Moderate" if shp < 0.06 else "🟢 Crisp")

    verdicts = {
        "Contrast": verdict_color_from_ratio_block(block_contrast),
        "Line weight": "🟢 Regular (approx.)",
        "Entropy": entropy_flag,
        "Sharpness": sharp_flag,
        "Sentence/Line": "🟢 Short" if len(" ".join([w['text'] for w in block['words']])) <= 60 else "🟡 Long fragment",
        "Alignment": "🟢 Left/Center assumed",
        "Edge proximity": "🟢 Safe (demo placeholder)",
        "OCR coverage": "🔴 Missing" if unreadable_words else "🟢 Detected"
    }

    return {
        "text": " ".join([w['text'] for w in block['words']]),
        "bbox": block['bbox'],
        "size_pct": size_pct,
        "verdicts": verdicts,
        "block_contrast": block_contrast,
        "readable_words": readable_words,
        "unreadable_words": unreadable_words
    }

def summarize_style_block(scored):
    text = scored["text"]
    total = len(scored["readable_words"]) + len(scored["unreadable_words"])
    passes = len(scored["readable_words"])
    pass_rate = (passes / total) if total else 0.0
    block_c = scored["block_contrast"] or 0.0

    if pass_rate >= 0.8 and block_c >= 7.0:
        pass_line = "✅ Most text passes AAA."
    elif pass_rate >= 0.8 and block_c >= AA_THRESHOLD:
        pass_line = "✅ Most text passes AA."
    else:
        pass_line = "⚠️ Some text fails accessibility."

    miss_line = None
    if scored["unreadable_words"]:
        miss_line = "⚠️ Missing word(s): " + ", ".join(scored["unreadable_words"]) + \
                    " — if Lite can’t see these, some people may struggle too."

    return {"headline": text, "pass_line": pass_line, "miss_line": miss_line}

# ---------------- OCR-miss catcher (logos/fine print) ----------------
def propose_textlike_boxes(gray):
    edges = cv2.Canny(gray, 80, 180)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    dil = cv2.dilate(edges, kernel, 1)
    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape[:2]
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 14 or h < 10:
            continue
        if w / h < 1.1:
            continue
        # bias bottom 20% for fine print
        if y < int(0.80 * H) and h < 18:
            continue
        boxes.append((x, y, w, h))
    return sorted(boxes, key=lambda b: (b[1], b[0]))

def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix, iy = max(ax, bx), max(ay, by)
    ax2, by2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ax2 - ix) * max(0, by2 - iy)
    union = aw * ah + bw * bh - inter + 1e-6
    return inter / union

# ---------------- UI ----------------
st.title(f"🔎 Lite Checker™ — {BUILD_TAG}")
st.write("Upload an image to get **traffic-light scorecards per block** and a **plain-language summary**. "
         "This build uses **RapidOCR (detect only)** + **Tesseract (recognise)** to match your AI version.")

uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded image", use_column_width=True)

    # 1) Detect regions with RapidOCR (no recognition), then recognise with Tesseract
    det_boxes = detect_boxes(img)
    words = tesseract_words_in_boxes(img, det_boxes)

    # 2) Group into lines -> style-block segmentation
    line_blocks = group_by_line(words)
    style_blocks = segment_blocks_by_style(gray, line_blocks, thresh=STYLE_SPLIT_THRESHOLD)

    # 3) Score each style block
    results = [score_block(img, gray, b) for b in style_blocks]

    # 4) OCR-miss catcher (logos/fine print) — low-contrast regions not recognised by OCR
    textlike = propose_textlike_boxes(gray)
    ocr_boxes = [(w['left'], w['top'], w['width'], w['height']) for w in words]
    miss_regions = []
    for tb in textlike:
        overlaps = any(iou(tb, ob) > 0.3 for ob in ocr_boxes)
        if overlaps:
            continue
        r = word_contrast(gray, tb)
        if r is None or r < AA_THRESHOLD:
            miss_regions.append(tb)

    # 5) Render AI-style summaries + scorecards (per style block)
    any_block_fail = False
    any_unread = False

    for i, res in enumerate(results, start=1):
        summary = summarize_style_block(res)

        st.markdown(f"### Creative — Block {i}")
        st.markdown(f'**Headline:** “{summary["headline"]}”')
        st.write(summary["pass_line"])
        if summary["miss_line"]:
            st.write(summary["miss_line"])
            any_unread = True

        st.table({"Factor": list(res["verdicts"].keys()), "Verdict": list(res["verdicts"].values())})

        # Read / Not readable lists (no per-word tables)
        if res["readable_words"]:
            st.caption("✅ Read: " + " ".join(res["readable_words"]))
        if res["unreadable_words"]:
            st.warning("⚠️ Not readable: " + " ".join(res["unreadable_words"]))

        if "🔴" in res["verdicts"]["Contrast"]:
            any_block_fail = True

    # 6) Global warning for unreadable OCR-missed regions (e.g., bottom logos/fine print)
    if miss_regions:
        st.warning(f"⚠️ Unreadable elements detected (not recognised by OCR): {len(miss_regions)} region(s). "
                   "These may include small logos or fine print.")
        any_block_fail = True

    # 7) Overall verdict
    st.subheader("Overall Lite Verdict")
    if any_block_fail or any_unread:
        st.write("⚠️ **Some text fails accessibility.** Parts of the image include unreadable words or low-contrast regions.")
    else:
        st.write("✅ Strong readability across detected text.")
    st.write("👉 For precise checks — contrast ratios, stroke-width measures, and population-level impact — request a **Readability Confidence Score™ (RCS).**")
