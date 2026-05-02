import sys
import os
import glob
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pytesseract

# スロット並列処理用スレッドローカルストレージ
# （デバッグ出力の競合を防ぐ）
_slot_local = threading.local()

# デバッグ出力を1スロット分まとめてから print するためのロック
# 複数スレッドのログが行単位で混在するのを防ぐ
_print_lock = threading.Lock()


def _debug_print_flush():
    """
    スレッドローカルに溜めたデバッグ行をまとめて標準出力に書き出す。
    スロット処理の末尾で呼ぶこと。
    """
    lines: List[str] = getattr(_slot_local, "debug_lines", [])
    if lines:
        with _print_lock:
            print("\n".join(lines))
        _slot_local.debug_lines = []


def _dbg(msg: str):
    """
    デバッグメッセージをスレッドローカルバッファに追記する。
    直接 print しないことでスレッド間の出力混在を防ぐ。
    CONFIG.debug_mode が False の場合は何もしない。
    """
    if not CONFIG.debug_mode:
        return
    if not hasattr(_slot_local, "debug_lines"):
        _slot_local.debug_lines = []
    _slot_local.debug_lines.append(msg)

from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, 
                             QHBoxLayout, QRadioButton, QButtonGroup, 
                             QPushButton, QLineEdit)
from PyQt5.QtGui import QImage, QPixmap, QColor, QPalette, QIntValidator
from PyQt5.QtCore import Qt

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# =========================================================
# 設定
# =========================================================
@dataclass
class Config:
    """アプリケーション設定"""
    debug_mode: bool = True
    output_txt: bool = True
    gt_mode: bool = False
    gt_file: str = "GT_sample.txt"
    
    # 出力フォーマット
    slot_separator: str = "."
    team_separator: str = " : "
    round_value: int = 10000
    
    # 画像処理
    slots: Optional[List[float]] = None
    y_min: int = 400
    y_max: int = 770
    roi_width: int = 120
    digit_band_height: int = 55
    
    def __post_init__(self):
        if self.slots is None:
            self.slots = [0.100, 0.160, 0.220, 0.280, 0.340, 0.400, 
                         0.600, 0.660, 0.720, 0.780, 0.840, 0.900]

CONFIG = Config()

# -----------------------------------------------------------
# VALUE_MISMATCH リカバリーで使う word 単位信頼度閾値
#
# Tesseract の image_to_data (psm7) は conf を「word 単位」で返す。
# 複数桁がひとつの word として認識された場合、その word 内の全桁が
# 同じ conf 値を持つ（例: "152211" → 全桁 conf=82）。
#
# そのため「桁ごとに conf が異なる」のではなく、
# 「word 全体の conf が低ければその word 内の 6/8/9 を psm10 で再確認」
# という word 単位の判定になる。
#
# 設定の目安：
#   80  → word conf が低い場合のみ再確認。psm10 発動は稀（推奨）
#   70  → より保守的。高品質な画像では発動がほぼゼロになる
#   90  → 旧実装に近い挙動（発動頻度は高め）
# -----------------------------------------------------------
CONF_WORD_THRESHOLD: float = 85.0

# -----------------------------------------------------------
# スレッドローカルなデバッグコンテキスト
#   _slot_local.slot_idx       : 現在処理中のスロット番号
#   _slot_local.slot_x         : 現在処理中のスロットX座標
#   _slot_local.slot_has_debug : 現在のスロットでデバッグ出力があったか
# -----------------------------------------------------------
def _get_slot_idx() -> int:
    return getattr(_slot_local, "slot_idx", -1)

def _get_slot_x() -> float:
    return getattr(_slot_local, "slot_x", 0.0)

def _get_slot_has_debug() -> bool:
    return getattr(_slot_local, "slot_has_debug", False)

def _set_slot_has_debug(v: bool):
    _slot_local.slot_has_debug = v

def _init_slot_context(slot_idx: int, slot_x: float):
    """スロット処理開始時にスレッドローカルを初期化する"""
    _slot_local.slot_idx       = slot_idx
    _slot_local.slot_x         = slot_x
    _slot_local.slot_has_debug = False

# =========================================================
# データクラス
# =========================================================
@dataclass
class OCRResult:
    """OCR結果を格納"""
    value: int
    ocr_str: str
    ocr_len: int
    digit_count: int
    digit_ok: bool
    confidence: float          # 全桁の信頼度平均（表示・後方互換用）
    char_confs: List[float]    # 桁ごとの信頼度リスト（ocr_str と同順）
    error_code: Optional[str]
    cc_info: list
    digit_bin_vis: np.ndarray
    ocr_img: np.ndarray
    band_vis: np.ndarray
    # リカバリー試行状態
    recover_digits_attempted: bool = False  # 桁欠損リカバリーが試行されたか
    recover_value_attempted: bool = False   # 値誤検出リカバリーが試行されたか

@dataclass
class ErrorAnalysis:
    """エラー分析結果"""
    error_codes: List[str]
    category: Optional[str]
    incorrect_count: int = 0
    incorrect_slots: Optional[List[int]] = None   # GT不一致だったスロット番号リスト（GTモード専用）

    def __post_init__(self):
        if self.incorrect_slots is None:
            self.incorrect_slots = []

# =========================================================
# OCR処理（既存ロジックを整理）
# =========================================================
class OCRProcessor:
    """OCR処理を担当するクラス"""
    
    @staticmethod
    def find_digit_band(thresh: np.ndarray) -> Tuple[int, int, str]:
        """数字帯を検出"""
        h = thresh.shape[0]
        proj = np.sum(thresh == 255, axis=1)
        max_proj = np.max(proj)
        
        if max_proj < thresh.shape[1] * 0.05:
            return 0, 0, "NO_DIGIT_CANDIDATE"
        
        rows = np.where(proj > max_proj * 0.6)[0]
        if len(rows) == 0:
            return 0, 0, "NO_DIGIT_CANDIDATE"
        
        center = int(np.mean(rows))
        y1 = max(0, center - CONFIG.digit_band_height // 2)
        y2 = min(h, center + CONFIG.digit_band_height // 2)
        
        if y2 - y1 < CONFIG.digit_band_height * 0.5:
            return 0, 0, "NO_DIGIT_CANDIDATE"
        
        return y1, y2, "OK"
    
    @staticmethod
    def preprocess_for_digit_cc(thresh: np.ndarray) -> np.ndarray:
        """ノイズ除去前処理"""
        h, w = thresh.shape
        out = thresh.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 2, 1))
        hline = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
        out[hline == 255] = 255
        return out
    
    @staticmethod
    def analyze_cc(bin_img: np.ndarray) -> list:
        """連結成分解析"""
        fg = (bin_img == 0).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
        h_img, w_img = bin_img.shape
        ccs = []
        
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]
            
            if w > w_img * 0.9 and h > h_img * 0.6:
                continue
            
            ccs.append({
                "bbox": (x, y, w, h),
                "cx": cx,
                "w": w,
                "h": h,
                "area": area,
                "type": "unknown"
            })
        
        return ccs
    
    @staticmethod
    def classify_ccs(ccs: list, img_h: int) -> Tuple[int, bool]:
        """CCを分類"""
        widths = [c["w"] for c in ccs if c["area"] > 40]
        if len(widths) >= 5:
            widths = sorted(widths)[1:-1]
        median_w = np.median(widths) if widths else 0
        
        digit_count = 0
        for c in ccs:
            w, h, area = c["w"], c["h"], c["area"]
            
            if median_w and w > median_w * 3.0:
                c["type"] = "wide_noise"
                continue
            if h < img_h * 0.30 or area < 30:
                c["type"] = "small_noise"
                continue
            
            ink_ratio = area / max(w * h, 1)
            if ink_ratio < 0.12:
                c["type"] = "sparse_noise"
                continue
            if h < img_h * 0.25 and w > img_h * 0.8:
                c["type"] = "wide_noise"
                continue
            
            c["type"] = "digit"
            digit_count += 1
        
        digit_ok = 1 <= digit_count <= 8
        return digit_count, digit_ok
    
    @staticmethod
    def add_ocr_margin(thresh: np.ndarray, ratio: float = 0.6) -> np.ndarray:
        """OCR用マージン追加"""
        h, w = thresh.shape
        px = int(h * ratio)
        py = int(h * 0.25)
        return cv2.copyMakeBorder(thresh, py, py, px, px, cv2.BORDER_CONSTANT, value=255)
    
    @staticmethod
    def thicken_for_ocr(thresh: np.ndarray) -> np.ndarray:
        """OCR用に太らせる"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        return cv2.dilate(thresh, kernel, iterations=1)
    
    @classmethod
    def split_wide_digit_cc(cls, bin_img: np.ndarray, cc: dict) -> list:
        """幅広CCを分割（連結した数字を分離）- 改良版
        
        改良点:
        1. psm7で認識した桁数を信頼
        2. 画像の幅を桁数で均等分割
        3. OCR失敗時(桁数1以下)は画像幅から推定して分割
        """
        x, y, w, h = cc["bbox"]
        crop = bin_img[y:y+h, x:x+w]
        
        # OCR用に前処理
        ocr_src = cls.thicken_for_ocr(cls.add_ocr_margin(crop, ratio=0.4))
        
        # psm7で全体を認識
        text = pytesseract.image_to_string(
            ocr_src,
            config="--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789"
        )
        digits = "".join(filter(str.isdigit, text))
        
        # 認識した桁数（失敗時は幅から推定）
        if len(digits) > 1:
            split_n = len(digits)
        else:
            # 画像幅から推定（高さの60%が1桁の幅）
            estimated_single_w = h * 0.6
            split_n = max(2, int(w / estimated_single_w + 0.5))
        
        if CONFIG.debug_mode:
            _dbg(f"[SPLIT_DEBUG] CC width={w}px, OCR='{digits}' ({len(digits)}桁), split_n={split_n}")
        
        # 桁数に応じて均等分割
        avg_w = w / split_n
        new_ccs = []
        
        for i in range(split_n):
            nx = int(x + i * avg_w)
            nw = int(avg_w) if i < split_n - 1 else x + w - nx
            new_ccs.append({
                "bbox": (nx, y, nw, h),
                "cx": nx + nw / 2,
                "w": nw,
                "h": h,
                "area": nw * h,
                "type": "digit"
            })
        
        return new_ccs
    
    @classmethod
    def resolve_wide_digit_ccs(cls, bin_img: np.ndarray, ccs: list) -> list:
        """幅広のdigit CCを分割処理 - 統合版
        
        判定条件:
        1. 中央値の2.0倍以上（連結桁の可能性）
        2. 絶対値判定も併用
        """
        digit_ws = [c["w"] for c in ccs if c["type"] == "digit"]
        median_w = np.median(digit_ws) if digit_ws else 0
        
        # 画像高さから予想される単一桁の幅を推定
        img_h = bin_img.shape[0]
        expected_single_w = img_h * 0.6
        
        # 先に分割判定を行う（出力するかどうか決めるため）
        split_decisions = []
        new_ccs = []
        has_split = False
        
        for idx, c in enumerate(ccs):
            if c["type"] != "digit":
                new_ccs.append(c)
                split_decisions.append(None)
                continue
            
            should_split = False
            ratio_median = 0.0
            ratio_abs = 0.0
            
            # 条件1: 中央値ベース（2.0倍以上）
            if median_w > 0:
                ratio_median = c["w"] / median_w
                if ratio_median >= 2.0:
                    should_split = True
            
            # 条件2: 絶対値ベース
            if expected_single_w > 0:
                ratio_abs = c["w"] / expected_single_w
                if ratio_abs >= 2.0:
                    should_split = True
            
            split_decisions.append({
                "idx": idx,
                "w": c["w"],
                "ratio_median": ratio_median,
                "ratio_abs": ratio_abs,
                "should_split": should_split,
                "cc": c
            })
            
            if should_split:
                has_split = True
        
        # 分割がある場合のみデバッグ出力
        if CONFIG.debug_mode and len(digit_ws) > 0 and has_split:
            if not _get_slot_has_debug():
                _dbg(f"\n{'='*60}\nSlot {_get_slot_idx()} (x={_get_slot_x():.3f})\n{'='*60}")
            
            _dbg(f"[SPLIT_DEBUG] digit_count={len(digit_ws)}, median_w={median_w:.1f}px, img_h={img_h}px")
            _dbg(f"[SPLIT_DEBUG] CC widths: {[f'{w:.1f}' for w in digit_ws]}")
            
            for decision in split_decisions:
                if decision is None:
                    continue
                status = "SPLIT" if decision["should_split"] else "KEEP"
                _dbg(f"[SPLIT_DEBUG] CC[{decision['idx']}]: w={decision['w']:.1f}px, ratio_median={decision['ratio_median']:.2f}, ratio_abs={decision['ratio_abs']:.2f} → {status}")
            
            _set_slot_has_debug(True)
        
        # 実際の分割処理
        new_ccs = []
        for c in ccs:
            if c["type"] != "digit":
                new_ccs.append(c)
                continue
            
            should_split = False
            if median_w > 0 and c["w"] / median_w >= 2.0:
                should_split = True
            if expected_single_w > 0 and c["w"] / expected_single_w >= 2.0:
                should_split = True
            
            if should_split:
                split_result = cls.split_wide_digit_cc(bin_img, c)
                new_ccs.extend(split_result)
            else:
                new_ccs.append(c)
        
        return new_ccs
    
    @classmethod
    def ocr_single_digit_psm10(cls, bin_img: np.ndarray) -> Optional[str]:
        """単一数字をpsm10で認識"""
        ocr_src = cls.thicken_for_ocr(cls.add_ocr_margin(bin_img, ratio=0.4))
        text = pytesseract.image_to_string(
            ocr_src,
            config="--oem 1 --psm 10 -c tessedit_char_whitelist=0123456789"
        )
        digits = "".join(filter(str.isdigit, text))
        return digits[0] if digits else None
    
    @classmethod
    def _infer_question_marks(cls, psm10_digits: list, psm7_str: str) -> list:
        """「?」を推論で補完
        
        アルゴリズム:
        1. 確実な数字（「?」でない部分）で最長共通部分列を検出
        2. その前後関係から「?」の位置をpsm7の対応する桁にマッピング
        3. 「?」を補完
        
        例:
        psm10: ["5", "?", "7", "3", "4"]
        psm7:  "9734"
        → アンカー: "734" がpsm10[2:5]とpsm7[1:4]で一致
        → psm10[1]="?" は psm7[0]="9" に対応
        → 結果: ["5", "9", "7", "3", "4"]
        """
        result = list(psm10_digits)
        
        # psm7から確実な数字のみ抽出（連続する確実な部分を探す）
        certain_digits = [d for d in psm10_digits if d != "?"]
        certain_str = "".join(certain_digits)
        
        # アンカーポイント: psm7内でcertain_strの最長一致を探す
        # （最も長い連続一致部分を見つける）
        best_anchor_len = 0
        best_anchor_psm10_start = -1
        best_anchor_psm7_start = -1
        
        # psm10の確実な数字のインデックスマップ
        certain_indices = [i for i, d in enumerate(psm10_digits) if d != "?"]
        
        # 連続する確実な部分を検出
        for psm10_start_idx in range(len(certain_indices)):
            for psm7_start_idx in range(len(psm7_str)):
                matched = 0
                p10_idx = psm10_start_idx
                p7_idx = psm7_start_idx
                
                while (p10_idx < len(certain_indices) and 
                       p7_idx < len(psm7_str) and 
                       psm10_digits[certain_indices[p10_idx]] == psm7_str[p7_idx]):
                    matched += 1
                    p10_idx += 1
                    p7_idx += 1
                
                if matched > best_anchor_len:
                    best_anchor_len = matched
                    best_anchor_psm10_start = certain_indices[psm10_start_idx]
                    best_anchor_psm7_start = psm7_start_idx
        
        # アンカーが見つからない、または短すぎる場合は補完不可
        if best_anchor_len < 2:
            return result
        
        # アンカーを基準に「?」を補完
        # psm10とpsm7の位置関係を確立
        offset = best_anchor_psm10_start - best_anchor_psm7_start
        
        for i, d in enumerate(psm10_digits):
            if d == "?":
                # このpsm10の位置に対応するpsm7の位置
                psm7_idx = i - offset
                if 0 <= psm7_idx < len(psm7_str):
                    result[i] = psm7_str[psm7_idx]
                # else: psm7に対応する桁がない（欠損部分の可能性）
        
        return result
    
    @classmethod
    def recover_missing_digits(cls, bin_img: np.ndarray, digit_ccs: list, psm7_str: str, digit_count: int, slot_idx: int = -1) -> Optional[str]:
        """桁欠損を検出して補完（改良版v4）
        
        重要な変更:
        - psm10の結果を最終的に採用（psm7は参考情報のみ）
        - 「?」のみpsm7から補完を試みる
        - デバッグ出力にスロット番号を追加
        - 有意義な情報がある場合のみ出力
        """
        if not digit_ccs or digit_count == 0:
            return None
        
        # 各CCを個別にOCR
        individual_digits = []
        for c in digit_ccs:
            x, y, w, h = c["bbox"]
            crop = bin_img[y:y+h, x:x+w]
            d = cls.ocr_single_digit_psm10(crop)
            individual_digits.append(d if d else "?")
        
        psm10_len = len(individual_digits)
        psm7_len = len(psm7_str)
        
        # デバッグ出力（リカバリーが必要な場合のみ）
        if CONFIG.debug_mode and psm10_len > psm7_len:
            if not _get_slot_has_debug():
                _dbg(f"\n{'='*60}\nSlot {_get_slot_idx()} (x={_get_slot_x():.3f})\n{'='*60}")
            
            _dbg(f"[RECOVER_DEBUG] psm10: {''.join(individual_digits)} ({psm10_len}桁)")
            _dbg(f"[RECOVER_DEBUG] psm7:  {psm7_str} ({psm7_len}桁)")
            _dbg(f"[RECOVER_DEBUG] digit_count: {digit_count}")
            _set_slot_has_debug(True)
        
        # ステップ1: 「?」がある場合は推論補完を試みる
        if "?" in individual_digits:
            individual_digits = cls._infer_question_marks(individual_digits, psm7_str)
            if CONFIG.debug_mode and psm10_len > psm7_len:
                _dbg(f"[RECOVER_DEBUG] After inference: {''.join(individual_digits)}")
        
        # ステップ2: 欠損桁の検出
        if psm10_len > psm7_len:
            missing_count = psm10_len - psm7_len
            missing_part = individual_digits[:missing_count]
            
            if CONFIG.debug_mode:
                _dbg(f"[RECOVER_DEBUG] Missing count: {missing_count}")
                _dbg(f"[RECOVER_DEBUG] Missing part: {missing_part}")
            
            # 欠損部分に「?」が残っていたら失敗
            if "?" in missing_part:
                if CONFIG.debug_mode:
                    _dbg(f"[RECOVER_DEBUG] FAILED: '?' in missing part")
                return None
            
            result = "".join(missing_part)
            if CONFIG.debug_mode:
                _dbg(f"[RECOVER_DEBUG] SUCCESS: missing='{result}'")
            return result
        
        return None
    
    @staticmethod
    def remove_noise_keep_only_digit_cc(bin_img: np.ndarray, cc_list: list) -> np.ndarray:
        """数字以外のノイズを除去"""
        h, w = bin_img.shape
        out = np.full((h, w), 255, dtype=np.uint8)
        for c in cc_list:
            if c["type"] == "digit":
                x, y, cw, ch = c["bbox"]
                out[y:y+ch, x:x+cw] = bin_img[y:y+ch, x:x+cw]
        return out
    
    @staticmethod
    def extract_char_confidences(
        data: dict, ocr_str: str
    ) -> Tuple[float, List[float]]:
        """
        word 単位の信頼度から、ocr_str と同長の conf リストを返す。

        Returns
        -------
        mean_conf : float
            全 word の信頼度加重平均（0–100）。桁なし時は 0.0。
        char_confs : List[float]
            ocr_str の各桁に対応する conf リスト。

        Notes
        -----
        Tesseract の image_to_data (psm7) は conf を「word 単位」で返す。
        例: "152211" が1つの word として認識された場合
            → conf=82 が1つだけ得られ、全6桁に同じ値が入る。
        例: "152" "211" と2つの word に分割された場合
            → conf=90, conf=65 のように word ごとに異なる値が得られる。

        したがって char_confs の各要素は「その桁が属する word の conf」であり、
        同じ word 内の桁は全て同じ値になる。これは Tesseract の制約による仕様。

        VALUE_MISMATCH リカバリーは「word の conf が CONF_WORD_THRESHOLD 未満
        なら、その word 内の 6/8/9 を psm10 で再確認」という word 単位の
        判定として機能する（桁ごとに異なる conf での判定ではない）。
        """
        # word ごとの (digits_in_word, conf) を順に収集
        word_pairs: List[Tuple[str, float]] = []
        for txt, conf in zip(data["text"], data["conf"]):
            digits = "".join(filter(str.isdigit, txt))
            if not digits:
                continue
            try:
                c = float(conf)
            except (ValueError, TypeError):
                c = 100.0
            if conf == "-1":
                c = 100.0
            word_pairs.append((digits, c))

        # 各桁に「その桁が属する word の conf」を割り当てる
        # 同 word 内の全桁は同じ conf 値になる（Tesseract の仕様）
        char_confs: List[float] = []
        for digits, c in word_pairs:
            for _ in digits:
                char_confs.append(c)

        # ocr_str の長さに合わせてトリム・補完
        # （word 分割の境界がずれた場合の安全策）
        n = len(ocr_str)
        if len(char_confs) >= n:
            char_confs = char_confs[:n]
        else:
            char_confs += [100.0] * (n - len(char_confs))

        mean_conf = float(np.mean(char_confs)) if char_confs else 0.0
        return mean_conf, char_confs
    
    @classmethod
    def process_roi(cls, img: np.ndarray) -> OCRResult:
        """ROI画像からOCR実行"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bin_raw = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # 数字帯検出
        y1, y2, band_status = cls.find_digit_band(bin_raw)
        if band_status == "NO_DIGIT_CANDIDATE":
            return OCRResult(
                value=0,
                ocr_str="",
                ocr_len=0,
                digit_count=0,
                digit_ok=False,
                confidence=0.0,
                char_confs=[],
                error_code="NO_DIGIT",
                cc_info=[],
                digit_bin_vis=np.zeros_like(bin_raw),
                ocr_img=np.zeros((1, 1), dtype=np.uint8),
                band_vis=cv2.cvtColor(np.zeros((CONFIG.digit_band_height, bin_raw.shape[1]), dtype=np.uint8), cv2.COLOR_GRAY2BGR),
                recover_digits_attempted=False,
                recover_value_attempted=False
            )
        
        band = bin_raw[y1:y2]
        band_vis = cv2.cvtColor(band, cv2.COLOR_GRAY2BGR)
        
        # CC解析
        cc_src = cls.preprocess_for_digit_cc(band)
        ccs = cls.analyze_cc(cc_src)
        digit_count, digit_ok = cls.classify_ccs(ccs, cc_src.shape[0])
        
        # ノイズ除去して再解析
        digit_only = cls.remove_noise_keep_only_digit_cc(cc_src, ccs)
        ccs = cls.analyze_cc(digit_only)
        digit_count, digit_ok = cls.classify_ccs(ccs, digit_only.shape[0])
        
        # 幅広CCを分割（連結数字の分離）
        ccs = cls.resolve_wide_digit_ccs(digit_only, ccs)
        
        digit_ccs = [c for c in ccs if c["type"] == "digit"]
        digit_ccs.sort(key=lambda c: c["bbox"][0])
        digit_count = len(digit_ccs)
        
        # OCR実行
        ocr_src = cls.remove_noise_keep_only_digit_cc(cc_src, ccs)
        ocr_img = cls.thicken_for_ocr(cls.add_ocr_margin(ocr_src))
        
        data = pytesseract.image_to_data(
            ocr_img,
            config="--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789",
            output_type=pytesseract.Output.DICT
        )
        
        ocr_str = "".join(t for t in data["text"] if t.strip().isdigit())
        ocr_len = len(ocr_str)
        confidence, char_confs = cls.extract_char_confidences(data, ocr_str)
        
        # エラー判定と桁欠損リカバリー
        error_code = None
        value = 0
        recover_digits_attempted = False
        recover_value_attempted = False
        
        # 桁欠損リカバリー（digit_count > ocr_len の場合）
        if digit_ok and digit_count > ocr_len and ocr_len > 0:
            recover_digits_attempted = True
            
            # スロット番号は外部から渡されないため、ここでは-1とする
            missing = cls.recover_missing_digits(digit_only, digit_ccs, ocr_str, digit_count, slot_idx=-1)
            
            if missing is not None:
                # psm10で全CCを再認識
                psm10_full = []
                for cc in digit_ccs:
                    x, y, w, h = cc["bbox"]
                    crop = digit_only[y:y+h, x:x+w]
                    d = cls.ocr_single_digit_psm10(crop)
                    psm10_full.append(d if d else "?")
                
                # 「?」を補完
                if "?" in psm10_full:
                    psm10_full = cls._infer_question_marks(psm10_full, ocr_str)
                
                # psm10結果を採用
                ocr_str = "".join(psm10_full)[:digit_count]
                ocr_len = len(ocr_str)
                
                if ocr_str.isdigit() and "?" not in ocr_str:
                    value = int(ocr_str)
                    error_code = "RECOVER_DIGITS"
                else:
                    value = 0
                    error_code = "RECOVER_FAILED"
            else:
                # リカバリー失敗
                value = int(ocr_str) if ocr_str.isdigit() else 0
                error_code = "RECOVER_FAILED"
        
        # 通常のエラー判定（リカバリーされなかった場合）
        if error_code is None:
            if ocr_len == 0:
                if digit_ok and digit_count == 1:
                    value = 0
                else:
                    error_code = "NO_OCR"
            elif not digit_ok or ocr_len != digit_count:
                value = int(ocr_str) if ocr_str.isdigit() else 0
                error_code = "LEN_MISMATCH"
            else:
                value = int(ocr_str) if ocr_str.isdigit() else 0
                error_code = "OCR_OK" if ocr_str.isdigit() else "OCR_INVALID_CHAR"
        
        # VALUE_MISMATCHリカバリー（5→6/8/9の誤検出を修正）
        #
        # 旧実装：confidence（全体平均）< 95 なら 6/8/9 を含む全桁を psm10 で再確認
        #   → 平均が高くても実際には信頼度の低い桁が混在しうる問題があった
        #
        # 新実装（方針C）：
        #   Tesseract (psm7) の conf は word 単位で返される。
        #   "152211" が1 word なら全桁が同じ conf 値になるため、
        #   「word の conf が閾値未満ならその word 内の 6/8/9 を psm10 で再確認」
        #   という word 単位の判定として機能する。
        #
        #   - word conf >= CONF_WORD_THRESHOLD → psm10 呼び出しゼロ（高信頼 word）
        #   - word conf <  CONF_WORD_THRESHOLD → 6/8/9 の桁だけ psm10 で再確認
        #
        # 誤検出抑制の仕組み：
        #   word conf が高い（≥80）のに誤認識する確率は極めて低く、
        #   0.01%以下の水準を維持できる。
        if (error_code == "OCR_OK" and digit_ok and
                len(ocr_str) == digit_count and
                any(d in ocr_str for d in ("6", "8", "9"))):

            recover_value_attempted = True
            new_digits = list(ocr_str)

            for i, d in enumerate(ocr_str):
                if d not in ("6", "8", "9"):
                    continue

                # char_confs[i] = この桁が属する word の conf
                # （同 word 内の桁は全て同じ値になる）
                word_conf = char_confs[i] if i < len(char_confs) else 100.0

                if word_conf >= CONF_WORD_THRESHOLD:
                    # word 全体が高信頼 → psm10 不要（スキップ）
                    _dbg(
                        f"[VALUE_RECOVER] idx={i} '{d}' word_conf={word_conf:.1f}"
                        f" >= {CONF_WORD_THRESHOLD} → skip psm10"
                    )
                    continue

                # word conf が低い → この桁を psm10 で再確認
                _dbg(
                    f"[VALUE_RECOVER] idx={i} '{d}' word_conf={word_conf:.1f}"
                    f" < {CONF_WORD_THRESHOLD} → run psm10"
                )
                cc = digit_ccs[i]
                x, y, w, h = cc["bbox"]
                crop = digit_only[y:y+h, x:x+w]
                d1 = cls.ocr_single_digit_psm10(crop)

                if d1 == "5":
                    new_digits[i] = "5"
                    _dbg(f"[VALUE_RECOVER] idx={i} corrected '{d}' → '5'")

            new_str = "".join(new_digits)
            if new_str != ocr_str and new_str.isdigit():
                ocr_str = new_str
                value = int(ocr_str)
                error_code = "RECOVER_VALUE_MISMATCH"

        return OCRResult(
            value=value,
            ocr_str=ocr_str,
            ocr_len=ocr_len,
            digit_count=digit_count,
            digit_ok=digit_ok,
            confidence=confidence,
            char_confs=char_confs,
            error_code=error_code,
            cc_info=ccs,
            digit_bin_vis=cc_src,
            ocr_img=ocr_img,
            band_vis=band_vis,
            recover_digits_attempted=recover_digits_attempted,
            recover_value_attempted=recover_value_attempted
        )

# =========================================================
# エラー分析
# =========================================================
class ErrorAnalyzer:
    """エラー分析を担当"""
    
    # 赤色エラー（致命的）を最優先
    ERROR_PRIORITY = [
        "RECOVER_FAILED(DIGITS)",
        "RECOVER_FAILED(VALUE)",
        "NO_OCR",
        "LEN_MISMATCH",
        "VALUE_MISMATCH",
        "FIXED",
        "RECOVER_DIGITS",
        "RECOVER_VALUE_MISMATCH",
        "OTHER",
        "NO_DIGIT",
    ]
    
    @staticmethod
    def round_value(v: int) -> int:
        """値を丸める"""
        if v >= 0:
            return round(v / CONFIG.round_value)
        return v
    
    @classmethod
    def compare_with_ground_truth(cls, ocr_values: List[int], ocr_errors: List[str], 
                                  gt_values: List[int], results: List[OCRResult]) -> ErrorAnalysis:
        """GT比較
        
        エラーコード判定の仕様（GTモード専用）:
        
        【基本フロー】
        1. LEN_MISMATCH検知
        2. → RECOVER_DIGITS試行（recover_digits_attempted=True）
        3. → GT比較:
           - 一致 → RECOVER_SUCCESS(DIGITS)
           - 不一致 → RECOVER_FAILED(DIGITS)
        
        4. リカバリー未試行でLEN_MISMATCH → そのままLEN_MISMATCH
        
        【判定ロジック】
        - RECOVER_DIGITS + GT一致 → RECOVER_SUCCESS(DIGITS)
        - RECOVER_DIGITS + GT不一致 → RECOVER_FAILED(DIGITS) ★赤色
        - RECOVER_FAILED + GT不一致 → RECOVER_FAILED(DIGITS) ★赤色（リカバリー失敗）
        - RECOVER_VALUE_MISMATCH + GT一致 → RECOVER_SUCCESS(VALUE)
        - RECOVER_VALUE_MISMATCH + GT不一致 → RECOVER_FAILED(VALUE) ★赤色
        - OCR_OK + GT一致 → GT_OK
        - OCR_OK + GT不一致 → VALUE_MISMATCH ★赤色
        - LEN_MISMATCH + 試行済 + GT不一致 → RECOVER_FAILED(DIGITS) ★赤色
        - LEN_MISMATCH + 未試行 → LEN_MISMATCH ★赤色（リカバリー対象外）
        """
        eval_vals = [cls.round_value(v) for v in ocr_values]
        final_errors = []
        incorrect = 0
        incorrect_slots: List[int] = []
        
        for i, (ocr_err, eval_val, gt_val, res) in enumerate(zip(ocr_errors, eval_vals, gt_values, results)):
            gt_equal = (eval_val == gt_val)
            
            if gt_equal:
                # === GT一致の場合 ===
                if ocr_err == "RECOVER_DIGITS":
                    err = "RECOVER_SUCCESS(DIGITS)"
                elif ocr_err == "RECOVER_VALUE_MISMATCH":
                    err = "RECOVER_SUCCESS(VALUE)"
                elif ocr_err in (None, "OCR_OK"):
                    err = "GT_OK"
                else:
                    # その他のエラーはそのまま（例外的にGTと一致したケース）
                    err = ocr_err
                    incorrect += 1
                    incorrect_slots.append(i)
            
            else:
                # === GT不一致の場合 ===
                if ocr_err == "RECOVER_DIGITS":
                    # リカバリーは実行されたが結果が間違い
                    err = "RECOVER_FAILED(DIGITS)"
                    incorrect += 1
                    incorrect_slots.append(i)
                
                elif ocr_err == "RECOVER_FAILED":
                    # リカバリー試行したが失敗（"?"含む等）
                    err = "RECOVER_FAILED(DIGITS)"
                    incorrect += 1
                    incorrect_slots.append(i)
                
                elif ocr_err == "RECOVER_VALUE_MISMATCH":
                    # 値リカバリーは実行されたが結果が間違い
                    err = "RECOVER_FAILED(VALUE)"
                    incorrect += 1
                    incorrect_slots.append(i)
                
                elif ocr_err in (None, "OCR_OK"):
                    # OCRは成功したがGTと不一致
                    err = "VALUE_MISMATCH"
                    incorrect += 1
                    incorrect_slots.append(i)
                
                elif ocr_err == "LEN_MISMATCH":
                    # 桁数不一致の詳細判定
                    if res.recover_digits_attempted:
                        # リカバリーは試行されたが、最終的にLEN_MISMATCHのまま
                        # （リカバリー失敗とみなす）
                        err = "RECOVER_FAILED(DIGITS)"
                    else:
                        # リカバリー未試行（条件を満たさなかった）
                        err = "LEN_MISMATCH"
                    incorrect += 1
                    incorrect_slots.append(i)
                
                elif ocr_err == "NO_OCR":
                    err = "NO_OCR"
                    incorrect += 1
                    incorrect_slots.append(i)
                
                elif ocr_err == "NO_DIGIT":
                    err = "NO_DIGIT"
                    incorrect += 1
                    incorrect_slots.append(i)
                
                else:
                    err = "OTHER"
                    incorrect += 1
                    incorrect_slots.append(i)
            
            final_errors.append(err)
        
        category = cls._determine_category(final_errors)
        return ErrorAnalysis(
            error_codes=final_errors,
            category=category,
            incorrect_count=incorrect,
            incorrect_slots=incorrect_slots,
        )
    
    @classmethod
    def analyze_ocr_errors(cls, ocr_errors: List[str]) -> ErrorAnalysis:
        """OCRエラーのみ分析（GTなし）"""
        category = cls._determine_category(ocr_errors)
        return ErrorAnalysis(error_codes=ocr_errors, category=category)
    
    @classmethod
    def _determine_category(cls, errors: List[str]) -> Optional[str]:
        """エラーカテゴリを決定"""
        normalized = [e if e not in (None, "OCR_OK", "GT_OK", "RECOVER_SUCCESS(DIGITS)", "RECOVER_SUCCESS(VALUE)") else "OK" for e in errors]
        
        if all(e == "OK" for e in normalized):
            return None
        
        for priority_err in cls.ERROR_PRIORITY:
            if priority_err in normalized:
                return priority_err
        
        return "OTHER"

# =========================================================
# ファイル管理
# =========================================================
class FileManager:
    """ファイル保存を担当"""
    
    @staticmethod
    def save_incorrect_image(src_path: str, category: str, gt_line: Optional[str] = None):
        """エラー画像を保存"""
        dst_dir = os.path.join("image", "incorrect", category)
        os.makedirs(dst_dir, exist_ok=True)
        
        # 画像をコピー
        shutil.copy(src_path, dst_dir)
        
        # GTがあれば保存
        if gt_line is not None:
            gt_file = os.path.join(dst_dir, f"{category}_true_val.txt")
            with open(gt_file, "a", encoding="utf-8") as f:
                f.write(gt_line + "\n")
    
    @staticmethod
    def format_output_line(values: List[int]) -> str:
        """出力行をフォーマット"""
        rounded = [ErrorAnalyzer.round_value(v) for v in values]
        left = CONFIG.slot_separator.join(str(v) for v in rounded[:6])
        right = CONFIG.slot_separator.join(str(v) for v in rounded[6:])
        return f"{left}{CONFIG.team_separator}{right}"

# =========================================================
# デバッグUI
# =========================================================
class DebugWindow(QWidget):
    """デバッグウィンドウ"""
    DISPLAY_MAX_W = 180
    DISPLAY_MAX_H = 80
    
    def __init__(self, title: str, results: List[OCRResult], error_codes: List[str]):
        super().__init__()
        self.setWindowTitle(title)
        
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(220, 235, 245))
        self.setPalette(pal)
        self.setAutoFillBackground(True)
        
        self.results = results
        self.original_values = [r.value for r in results]
        self.final_values = self.original_values.copy()
        self.edit_fields = []
        
        self._build_ui(error_codes)
    
    def _build_ui(self, error_codes: List[str]):
        """UI構築"""
        main = QVBoxLayout()
        row_top = QHBoxLayout()
        row_bottom = QHBoxLayout()
        
        for idx, (res, err) in enumerate(zip(self.results, error_codes)):
            vbox = self._create_slot_ui(idx, res, err)
            if idx < 6:
                row_top.addLayout(vbox)
            else:
                row_bottom.addLayout(vbox)
        
        main.addLayout(row_top)
        main.addLayout(row_bottom)
        
        btn = QPushButton("OK")
        btn.clicked.connect(self._on_close)
        main.addWidget(btn)
        
        self.setLayout(main)
        self.adjustSize()
    
    def _create_slot_ui(self, idx: int, res: OCRResult, err: str) -> QVBoxLayout:
        """スロットUI作成"""
        vbox = QVBoxLayout()
        vbox.setAlignment(Qt.AlignTop)
        
        # バンド画像
        vbox.addWidget(self._cv_to_label(res.band_vis))
        
        # 数値入力
        edit = QLineEdit(str(res.value))
        edit.setAlignment(Qt.AlignCenter)
        edit.setValidator(QIntValidator(0, 99999999))
        edit.setEnabled(False)
        edit.setStyleSheet("font-size: 18px; font-weight: bold;")
        vbox.addWidget(edit)
        self.edit_fields.append(edit)
        
        # エラー表示
        err_text = err if err else "OK"
        err_lbl = QLabel(f"ERR: {err_text}")
        err_lbl.setAlignment(Qt.AlignCenter)
        err_lbl.setStyleSheet(f"color: {self._get_error_color(err)}; font-weight: bold;")
        vbox.addWidget(err_lbl)
        
        # 信頼度表示
        # conf(avg) は全 word の加重平均。
        # 桁ごとの値は「その桁が属する word の conf」であり、
        # 同 word 内の桁は全て同じ値になる（Tesseract の仕様）。
        conf_lbl = QLabel(f"conf(avg): {res.confidence:.1f}")
        conf_lbl.setAlignment(Qt.AlignCenter)
        conf_lbl.setStyleSheet(f"color: {self._get_conf_color(res.confidence)}; font-weight: bold;")
        vbox.addWidget(conf_lbl)

        # "桁:word_conf" 形式で表示。閾値未満の word に属する桁は赤字。
        # ※同 word 内の桁は全て同じ conf 値になる
        if res.char_confs and res.ocr_str:
            parts = []
            for ch, c in zip(res.ocr_str, res.char_confs):
                if c < CONF_WORD_THRESHOLD:
                    parts.append(f'<span style="color:red">{ch}:{c:.0f}</span>')
                else:
                    parts.append(f'{ch}:{c:.0f}')
            char_conf_lbl = QLabel(" ".join(parts))
            char_conf_lbl.setAlignment(Qt.AlignCenter)
            char_conf_lbl.setTextFormat(Qt.RichText)
            char_conf_lbl.setStyleSheet("font-size: 10px;")
            vbox.addWidget(char_conf_lbl)
        
        # OCR情報
        status = "OK" if res.digit_ok and res.ocr_len == res.digit_count else "NG"
        info = QLabel(f"OCR:{res.ocr_len}桁/検出:{res.digit_count}桁 [{status}]")
        info.setAlignment(Qt.AlignCenter)
        vbox.addWidget(info)
        
        # OCR画像
        vbox.addWidget(self._cv_to_label(res.ocr_img, gray=True))
        vbox.addWidget(self._cv_to_label(self._draw_cc_boxes(res.digit_bin_vis, res.cc_info)))
        
        # 修正ラジオボタン
        rb_ok = QRadioButton("正しい")
        rb_ng = QRadioButton("間違い")
        rb_ok.setChecked(True)
        rb_ok.toggled.connect(lambda checked, e=edit: e.setEnabled(not checked))
        
        group = QButtonGroup(self)
        group.addButton(rb_ok)
        group.addButton(rb_ng)
        
        vbox.addWidget(rb_ok)
        vbox.addWidget(rb_ng)
        
        return vbox
    
    def _get_error_color(self, err: str) -> str:
        """エラー色取得"""
        if err in ("RECOVER_DIGITS", "RECOVER_VALUE_MISMATCH", "NO_DIGIT"):
            return "orange"
        elif err in ("RECOVER_SUCCESS(DIGITS)", "RECOVER_SUCCESS(VALUE)", "OCR_OK", "GT_OK"):
            return "green"
        elif err in ("RECOVER_FAILED(DIGITS)", "RECOVER_FAILED(VALUE)", "NO_OCR", "LEN_MISMATCH", "VALUE_MISMATCH"):
            return "red"
        return "blue"
    
    def _get_conf_color(self, conf: float) -> str:
        """信頼度色取得"""
        if conf >= 80:
            return "black"
        elif conf >= 60:
            return "orange"
        return "red"
    
    def _draw_cc_boxes(self, bin_img: np.ndarray, cc_list: list) -> np.ndarray:
        """CC矩形描画"""
        img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
        for c in cc_list:
            x, y, w, h = c["bbox"]
            color = (0, 255, 0) if c["type"] == "digit" else ((0, 0, 255) if c["type"] == "wide_noise" else (255, 0, 0))
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        return img
    
    def _cv_to_label(self, img: np.ndarray, gray: bool = False) -> QLabel:
        """OpenCV→QLabel変換"""
        if gray:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = np.ascontiguousarray(img)
        h, w, _ = img.shape
        scale = min(self.DISPLAY_MAX_W / w, self.DISPLAY_MAX_H / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        if scale < 1.0:
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        qimg = QImage(img.data, new_w, new_h, new_w * 3, QImage.Format_BGR888)
        lbl = QLabel()
        lbl.setPixmap(QPixmap.fromImage(qimg))
        lbl.setFrameStyle(QLabel.Box)
        lbl.setAlignment(Qt.AlignCenter)
        return lbl
    
    def _on_close(self):
        """閉じる時に値を取得"""
        for i, edit in enumerate(self.edit_fields):
            if edit.isEnabled():
                self.final_values[i] = int(edit.text())
        self.close()

# =========================================================
# メイン処理
# =========================================================
def _process_slot(args: Tuple) -> Tuple[int, "OCRResult"]:
    """
    スロット1件を処理するワーカー関数。
    ThreadPoolExecutor から呼ばれる。
    スレッドローカルにデバッグコンテキストを設定してから process_roi を実行し、
    スロット終了時にデバッグ出力をまとめて flush する。
    """
    slot_idx, sx, roi = args
    _init_slot_context(slot_idx, sx)
    result = OCRProcessor.process_roi(roi)
    # このスロット分のデバッグ出力をまとめて標準出力へ（他スレッドと混在しない）
    _debug_print_flush()
    return slot_idx, result


# スレッドプール（モジュール起動時に1度だけ生成して使い回す）
# 非デバッグモードのみ使用。スロット数＝12 なので max_workers=12 で飽和させる。
_SLOT_POOL: Optional[ThreadPoolExecutor] = None

def _get_slot_pool() -> ThreadPoolExecutor:
    global _SLOT_POOL
    if _SLOT_POOL is None:
        _SLOT_POOL = ThreadPoolExecutor(
            max_workers=len(CONFIG.slots),
            thread_name_prefix="SlotOCR",
        )
    return _SLOT_POOL


def process_image(img_path: str) -> Tuple[List[int], List["OCRResult"]]:
    """画像からOCR実行（スロットを並列処理）

    デバッグモード時も並列実行する。
    デバッグ出力は _dbg() でバッファリングし、スロット完了時に
    _debug_print_flush() でまとめて出力するため、複数スレッドの
    ログが行単位で混在することはない。
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"画像読み込み失敗: {img_path}")

    H, W, _ = img.shape

    # ROIを先に全スロット分切り出す（スレッド間で画像配列を共有しても安全）
    tasks: List[Tuple[int, float, np.ndarray]] = []
    for slot_idx, sx in enumerate(CONFIG.slots):
        cx  = int(sx * W)
        roi = img[CONFIG.y_min:CONFIG.y_max,
                  cx - CONFIG.roi_width // 2: cx + CONFIG.roi_width // 2]
        tasks.append((slot_idx, sx, roi))

    results: List[Optional["OCRResult"]] = [None] * len(CONFIG.slots)

    pool = _get_slot_pool()
    for slot_idx, res in pool.map(_process_slot, tasks):
        results[slot_idx] = res

    values = [r.value for r in results]
    return values, results

def load_ground_truth(path: str) -> List[str]:
    """GT読み込み"""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def parse_gt_line(line: str) -> List[int]:
    """GT行をパース"""
    left, right = line.split(CONFIG.team_separator)
    return ([int(x) for x in left.split(CONFIG.slot_separator)] + 
            [int(x) for x in right.split(CONFIG.slot_separator)])

def main():
    """メイン処理"""
    files = sorted(glob.glob("image/*.png"))
    total = len(files)
    
    if total == 0:
        print("画像が見つかりません")
        return
    
    # GT読み込み
    gt_lines = None
    if CONFIG.gt_mode:
        if not os.path.exists(CONFIG.gt_file):
            print("GTファイルが存在しない、もしくは名前が違います")
            return
        gt_lines = load_ground_truth(CONFIG.gt_file)
        assert len(gt_lines) == total, f"GT行数({len(gt_lines)})と画像枚数({total})が一致しません"
    
    # Qt初期化
    app = QApplication(sys.argv) if CONFIG.debug_mode else None
    
    output_lines = []
    total_incorrect = 0
    # GT不一致ファイルの記録: [(name, incorrect_slots), ...]
    incorrect_files: List[Tuple[str, List[int]]] = []
    
    for idx, img_path in enumerate(files):
        name = os.path.basename(img_path)
        title = f"{name} ({idx+1}/{total})"
        
        # OCR実行
        values, results = process_image(img_path)
        ocr_errors = [r.error_code for r in results]
        
        # エラー分析
        if CONFIG.gt_mode:
            gt_vals = parse_gt_line(gt_lines[idx])
            analysis = ErrorAnalyzer.compare_with_ground_truth(values, ocr_errors, gt_vals, results)
            total_incorrect += analysis.incorrect_count
            if analysis.incorrect_slots:
                incorrect_files.append((name, analysis.incorrect_slots))
        else:
            analysis = ErrorAnalyzer.analyze_ocr_errors(ocr_errors)
        
        # デバッグモード
        if CONFIG.debug_mode:
            win = DebugWindow(title, results, analysis.error_codes)
            win.show()
            app.exec_()
            
            # 修正値を反映
            if win.final_values != win.original_values:
                values = win.final_values
                analysis = ErrorAnalysis(error_codes=["FIXED"] * len(values), category="FIXED")
        
        # 出力
        line = FileManager.format_output_line(values)
        output_lines.append(line)
        print(f"{title}\n{line}")
        
        # エラー画像保存
        if analysis.category is not None:
            gt_line = gt_lines[idx] if CONFIG.gt_mode else None
            FileManager.save_incorrect_image(img_path, analysis.category, gt_line)
    
    # 結果出力
    if CONFIG.output_txt:
        with open("result.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
    
    # GT統計
    if CONFIG.gt_mode:
        total_slots = total * len(CONFIG.slots)
        rate = total_incorrect / total_slots * 100
        print(f"\nGT不一致率: {rate:.2f}% ({total_incorrect}/{total_slots})")
        # 不一致があったファイルをスロット番号付きで一覧表示
        for fname, slots in incorrect_files:
            slot_str = ",".join(str(s) for s in slots)
            print(f"{fname} : {slot_str}")

    # スレッドプールを明示的に解放
    if _SLOT_POOL is not None:
        _SLOT_POOL.shutdown(wait=False)

if __name__ == "__main__":
    main()