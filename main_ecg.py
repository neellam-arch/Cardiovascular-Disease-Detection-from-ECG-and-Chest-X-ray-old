"""
main.py  —  ECG Clinical Decision Support  |  ECGNet
=====================================================
Hospital-grade 12-lead ECG analysis system.
Displays 7 validated conditions (EXCELLENT / GOOD model accuracy).
Inferior_Infarct & LV_Hypertrophy are inferred internally but excluded
from display — model accuracy below clinical display threshold.

Model:
  ecg_model/best_model.pt    PyTorch ECGNet checkpoint (9-label)
  ecg_model/thresholds.json  per-class optimal thresholds (JSON wins)

Usage:  python main.py
"""

import json, re, threading
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent
CHECKPOINT   = BASE / "ecg_model" / "best_model.pt"
THRESH_JSON  = BASE / "ecg_model" / "thresholds.json"

DETAILS_CSV  = BASE / "patient_details.csv"
PATIENT_DIR  = BASE / "Final Dataset"
SUMMARY_XL   = BASE / "final_patient_dataset_summary.xlsx"
SORTED_DIR   = BASE / "sorted_patients"
DATA_REDUCED = BASE / "Final_Dataset"  # validated demo subset — 7,684 ECGs

DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_LEADS         = 12
N_SAMPLES       = 5000
LEAD_NAMES      = ["I","II","III","aVR","aVF","aVL","V1","V2","V3","V4","V5","V6"]
SAMPLES_PER_COL = N_SAMPLES // 4
WAVE_3x4        = [[0,3,6,9],[1,5,7,10],[2,4,8,11]]

# ─────────────────────────────────────────────────────────────────────────────
# LABELS  —  all 9 kept for model inference; only 7 shown in GUI
# ─────────────────────────────────────────────────────────────────────────────
ECG_LABEL_RULES = [
    ("ST_Ischemia",        ["consider acute st elevation","st elevation",
                            "may be due to myocardial ischemia",
                            "lateral st-t changes may be due to ischemia",
                            "extensive st-t changes may be due",
                            "st-t wave abnormality","st-t changes","st depression"]),
    ("Atrial_Fibrillation",["atrial fibrillation","a-fib","afib","af "]),
    ("Left_Bundle_Branch", ["left bundle branch","lbbb"]),
    ("Prolonged_QT",       ["prolonged qt","long qt","qt prolongation"]),
    ("Anterior_Infarct",   ["anterior infarct","anteroseptal infarct",
                            "possible anterior infarct","possible anteroseptal infarct",
                            "septal infarct","possible septal infarct",
                            "anterior mi","anteroseptal mi"]),
    ("Inferior_Infarct",   ["inferior infarct","inferior/lateral st",
                            "inferior t wave","inferior st elev",
                            "inferior mi","inferior wall mi"]),
    ("LV_Hypertrophy",     ["left ventricular hypertrophy","lvh",
                            "possible left ventricular hypertrophy",
                            "probable left ventricular hypertrophy",
                            "voltage criteria for lvh"]),
    ("Right_Bundle_Branch",["right bundle branch","rbbb"]),
    ("Sinus_Tachycardia",  ["sinus tachycardia"]),
]
LABEL_COLS = [n for n, _ in ECG_LABEL_RULES]

# 7 validated conditions for clinical display (EXCELLENT/GOOD accuracy)
DISPLAY_GROUPS = [
    ("Rhythm",              ["Atrial_Fibrillation", "Sinus_Tachycardia"]),
    ("Conduction",          ["Left_Bundle_Branch",  "Right_Bundle_Branch"]),
    ("Ischaemia / Infarct", ["ST_Ischemia",         "Anterior_Infarct"]),
    ("Interval",            ["Prolonged_QT"]),
]
DISPLAY_COLS = [c for _, grp in DISPLAY_GROUPS for c in grp]

LABEL_DISPLAY = {
    "ST_Ischemia":         "ST Ischaemia",
    "Atrial_Fibrillation": "Atrial Fibrillation",
    "Left_Bundle_Branch":  "Left Bundle Branch Block",
    "Prolonged_QT":        "Prolonged QT Interval",
    "Anterior_Infarct":    "Anterior Infarct",
    "Right_Bundle_Branch": "Right Bundle Branch Block",
    "Sinus_Tachycardia":   "Sinus Tachycardia",
}

# Model-validated performance per displayed label
LABEL_PERF = {
    "ST_Ischemia":         ("GOOD",      "AUC 0.886  F1 70%"),
    "Atrial_Fibrillation": ("EXCELLENT", "AUC 0.984  F1 86%"),
    "Left_Bundle_Branch":  ("EXCELLENT", "AUC 0.984  F1 89%"),
    "Prolonged_QT":        ("GOOD",      "AUC 0.968  F1 71%"),
    "Anterior_Infarct":    ("GOOD",      "AUC 0.934  F1 69%"),
    "Right_Bundle_Branch": ("EXCELLENT", "AUC 0.994  F1 90%"),
    "Sinus_Tachycardia":   ("EXCELLENT", "AUC 0.994  F1 93%"),
}

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE  (dark workstation theme)
# ─────────────────────────────────────────────────────────────────────────────
BG         = "#F0F4F8"
WHITE      = "#ffffff"
HEADER_BG  = "#0F172A"
ACCENT     = "#38BDF8"
BORDER     = "#2D3F55"
CARD_BG    = "#1E293B"

TXT        = "#CBD5E1"
TXT2       = "#64748B"
TXT3       = "#334155"

SEC_BG     = "#162032"
SEC_FG     = "#60A5FA"
SEC_ACC    = "#38BDF8"

AI_DET_BG  = "#064E2A"; AI_DET_FG  = "#34D399"
AI_NOR_BG  = "#1E293B"; AI_NOR_FG  = "#64748B"

ACT_DET_BG = "#451A00"; ACT_DET_FG = "#FBBF24"
ACT_NOR_BG = "#1E293B"; ACT_NOR_FG = "#475569"
PEND_BG    = "#1E293B"; PEND_FG    = "#334155"

TP_ROW = "#0D2B1A"
FN_ROW = "#2B1D00"
FP_ROW = "#2B0D0D"

CONF_HI  = "#34D399"
CONF_MED = "#FBBF24"
CONF_LO  = "#F87171"

DEFAULT_THRESH = 0.40


# ─────────────────────────────────────────────────────────────────────────────
# MODEL  (identical to train.py)
# ─────────────────────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        mid = max(ch // r, 8)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Sequential(nn.Linear(ch, mid, bias=False), nn.ReLU(inplace=True),
                                 nn.Linear(mid, ch, bias=False), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(self.gap(x).squeeze(-1)).unsqueeze(-1)

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, stride=1, dropout=0.0):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.se    = SEBlock(out_ch)
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.skip  = (nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                                    nn.BatchNorm1d(out_ch))
                      if in_ch != out_ch or stride != 1 else nn.Identity())
        self.act   = nn.ReLU(inplace=True)
    def forward(self, x):
        h = self.drop(self.se(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x)))))))
        return self.act(h + self.skip(x))

class MultiScaleStem(nn.Module):
    def __init__(self, in_ch=N_LEADS, out_ch=64):
        super().__init__()
        mid = out_ch // 3
        def branch(k):
            return nn.Sequential(nn.Conv1d(in_ch, mid, k, stride=2, padding=k//2, bias=False),
                                 nn.BatchNorm1d(mid), nn.ReLU(inplace=True))
        self.b_s = branch(7); self.b_m = branch(15); self.b_l = branch(31)
        self.proj = nn.Sequential(nn.Conv1d(mid*3, out_ch, 1, bias=False),
                                  nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True))
        self.pool = nn.MaxPool1d(3, stride=2, padding=1)
    def forward(self, x):
        return self.pool(self.proj(torch.cat([self.b_s(x), self.b_m(x), self.b_l(x)], dim=1)))

class ECGNet(nn.Module):
    def __init__(self, n_leads=N_LEADS, n_diag=len(LABEL_COLS)):
        super().__init__()
        dr = 0.08
        self.stem   = MultiScaleStem(n_leads, 64)
        self.layer1 = nn.Sequential(ResBlock1D(64,64,7,dropout=dr),  ResBlock1D(64,64,7,dropout=dr),
                                    ResBlock1D(64,64,7,dropout=dr),  ResBlock1D(64,64,7,dropout=dr))
        self.layer2 = nn.Sequential(ResBlock1D(64,128,7,stride=2,dropout=dr),  ResBlock1D(128,128,7,dropout=dr),
                                    ResBlock1D(128,128,7,dropout=dr),            ResBlock1D(128,128,7,dropout=dr))
        self.layer3 = nn.Sequential(ResBlock1D(128,256,7,stride=2,dropout=dr), ResBlock1D(256,256,7,dropout=dr),
                                    ResBlock1D(256,256,7,dropout=dr),            ResBlock1D(256,256,7,dropout=dr),
                                    ResBlock1D(256,256,7,dropout=dr),            ResBlock1D(256,256,7,dropout=dr),
                                    ResBlock1D(256,256,7,dropout=dr),            ResBlock1D(256,256,7,dropout=dr))
        self.layer4 = nn.Sequential(ResBlock1D(256,512,7,stride=2,dropout=dr), ResBlock1D(512,512,7,dropout=dr),
                                    ResBlock1D(512,512,7,dropout=dr),            ResBlock1D(512,512,7,dropout=dr))
        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.pool_max = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(1024,512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.50),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.30),
            nn.Linear(256, n_diag))
    def forward(self, x):
        x = self.stem(x)
        for L in (self.layer1, self.layer2, self.layer3, self.layer4): x = L(x)
        return self.head(torch.cat([self.pool_avg(x), self.pool_max(x)], dim=1).squeeze(-1))


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL  (direct binary reader — no wfdb)
# ─────────────────────────────────────────────────────────────────────────────
def read_ecg(hea_path: Path) -> np.ndarray:
    with open(hea_path) as fh:
        lines = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
    p  = lines[0].split(); nl = int(p[1]); ns = int(p[3])
    gains, baselines = [], []
    for line in lines[1: 1 + nl]:
        tok = line.split(); gt = tok[2]
        gains.append(float(re.split(r"[(/]", gt)[0]))
        baselines.append(int(gt.split("(")[1].split(")")[0]) if "(" in gt else 0)
    raw = np.fromfile(str(hea_path.with_suffix(".dat")), dtype=np.int16)
    raw = raw[: ns * nl].reshape(ns, nl)
    sig = np.zeros((nl, ns), dtype=np.float32)
    for i in range(nl):
        g = gains[i] if gains[i] != 0 else 1.0
        sig[i] = (raw[:, i].astype(np.float32) - baselines[i]) / g
    return sig

def normalize(sig: np.ndarray) -> np.ndarray:
    out = np.zeros_like(sig)
    for i in range(len(sig)):
        s = sig[i].copy()
        q25, q75 = np.percentile(s, [25, 75]); iqr = q75 - q25
        if iqr > 1e-6: s = np.clip(s, q25 - 5 * iqr, q75 + 5 * iqr)
        std = s.std()
        if std > 1e-6: out[i] = (s - s.mean()) / std
    return out

def pad_crop(sig: np.ndarray, L: int = N_SAMPLES) -> np.ndarray:
    n = sig.shape[1]
    if n >= L: return sig[:, :L]
    return np.concatenate([sig, np.zeros((sig.shape[0], L - n), dtype=np.float32)], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_model():
    if not CHECKPOINT.exists():
        return None, None, None
    ck    = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    cols  = ck.get("label_cols", LABEL_COLS)
    model = ECGNet(n_leads=ck.get("n_leads", N_LEADS), n_diag=len(cols)).to(DEVICE)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    # JSON always wins — allows threshold tuning without retraining
    if THRESH_JSON.exists():
        with open(THRESH_JSON) as f:
            d = json.load(f)
        thresholds = [d.get(c, DEFAULT_THRESH) for c in cols]
    else:
        thresholds = ck.get("thresholds") or [DEFAULT_THRESH] * len(cols)
    return model, thresholds, ck

@torch.no_grad()
def run_inference(model, sig: np.ndarray, thresholds, ck):
    s  = pad_crop(normalize(sig))
    x  = torch.from_numpy(s).unsqueeze(0).to(DEVICE)
    pr = torch.sigmoid(model(x)).cpu().numpy()[0]
    cols = (ck or {}).get("label_cols", LABEL_COLS)
    thr  = thresholds or [DEFAULT_THRESH] * len(cols)
    return ({c: bool(pr[i] >= thr[i]) for i, c in enumerate(cols)},
            {c: float(pr[i])          for i, c in enumerate(cols)})


# ─────────────────────────────────────────────────────────────────────────────
# PATIENT / ACTUAL LABEL LOOKUP
# ─────────────────────────────────────────────────────────────────────────────
_detail_df = None

def _load_detail_df():
    global _detail_df
    if _detail_df is not None:
        return _detail_df
    if DETAILS_CSV.exists():
        df = pd.read_csv(DETAILS_CSV); df.columns = df.columns.str.strip()
        _detail_df = df
    elif SUMMARY_XL.exists():
        df = pd.read_excel(SUMMARY_XL, engine="openpyxl"); df.columns = df.columns.str.strip()
        _detail_df = df
    return _detail_df

def get_actual_labels(ecg_filename: str) -> dict:
    df = _load_detail_df()
    if df is None: return {}
    f_col = next((c for c in ("ECG_File","ecg_file","ecg_file_name","File","ecg_study_id") if c in df.columns), None)
    if not f_col: return {}
    stem = Path(ecg_filename).stem
    mask = df[f_col].astype(str).str.replace(r"\.dat$","",regex=True).str.strip() == stem.strip()
    if not mask.any(): mask = df[f_col].astype(str).str.strip() == stem.strip()
    if not mask.any(): return {}
    row = df[mask].iloc[0]
    diag_col = next((c for c in ("ECG_Diagnosis","ECG_Machine_Report","Diagnosis") if c in row.index), None)
    if not diag_col: return {}
    txt = str(row[diag_col]).lower()
    return {col: bool(re.search("|".join(re.escape(k) for k in kws), txt))
            for col, kws in ECG_LABEL_RULES}

def get_patient_id(ecg_filename: str) -> str:
    df = _load_detail_df()
    if df is None: return "—"
    f_col = next((c for c in ("ECG_File","ecg_file","ecg_file_name","File","ecg_study_id") if c in df.columns), None)
    if not f_col: return "—"
    stem = Path(ecg_filename).stem
    mask = df[f_col].astype(str).str.replace(r"\.dat$","",regex=True).str.strip() == stem.strip()
    if not mask.any(): mask = df[f_col].astype(str).str.strip() == stem.strip()
    if not mask.any(): return "—"
    row  = df[mask].iloc[0]
    s_col = next((c for c in ("Subject_ID","subject_id","Patient_Folder") if c in row.index), None)
    return str(row[s_col]) if s_col else "—"


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("ECG Clinical Decision Support  —  ECGNet")
        self.configure(bg=HEADER_BG)
        self.state("zoomed")
        self.minsize(1280, 720)

        self.model, self.thresholds, self.ck = load_model()
        self.hea_path = None
        self.signal   = None
        self.ecg_file = None

        self.sv_patient = tk.StringVar(value="No patient")
        self.sv_status  = tk.StringVar(value="Open a .hea file to begin")
        self.row_refs   = {}
        self._stat_labels = {}

        self._build()

    def _build(self):
        self._header()
        body = tk.Frame(self, bg=HEADER_BG)
        body.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        body.columnconfigure(0, weight=0, minsize=420)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        self._left_panel(body)
        self._ecg_panel(body)

    # ── header ─────────────────────────────────────────────────────────────────
    def _header(self):
        hdr = tk.Frame(self, bg=HEADER_BG, height=46)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        left = tk.Frame(hdr, bg=HEADER_BG)
        left.pack(side="left", padx=18)
        tk.Label(left, text="▸", bg=HEADER_BG, fg=ACCENT,
                 font=("Segoe UI", 12)).pack(side="left")
        tk.Label(left, text=" ECG ANALYSIS", bg=HEADER_BG, fg=WHITE,
                 font=("Segoe UI", 11, "bold")).pack(side="left")
        tk.Label(left, text="  AI  12-Lead Clinical Decision Support",
                 bg=HEADER_BG, fg=TXT2, font=("Segoe UI", 8)).pack(side="left")

        right = tk.Frame(hdr, bg=HEADER_BG)
        right.pack(side="right", padx=18)

        if self.model and self.ck:
            auc = self.ck.get("mean_auc", 0.9069)
            tk.Label(right, text=f"AUC {auc:.3f}",
                     bg=HEADER_BG, fg=TXT3,
                     font=("Consolas", 7)).pack(side="left", padx=(0, 16))

        tk.Label(right, textvariable=self.sv_patient,
                 bg=HEADER_BG, fg=ACCENT,
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 14))

        self.btn = tk.Button(right, text="Open ECG  ▶",
                             font=("Segoe UI", 9, "bold"),
                             bg=ACCENT, fg="#071020",
                             activebackground="#7DD3FC", activeforeground="#071020",
                             relief="flat", cursor="hand2", bd=0,
                             padx=14, pady=6,
                             command=self.browse)
        self.btn.pack(side="left")

    # ── left panel ─────────────────────────────────────────────────────────────
    def _left_panel(self, parent):
        lf = tk.Frame(parent, bg=CARD_BG, width=420,
                      highlightthickness=1, highlightbackground=BORDER)
        lf.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        lf.pack_propagate(False)

        # column header
        chdr = tk.Frame(lf, bg=SEC_BG)
        chdr.pack(fill="x")
        for ci, (txt, w) in enumerate([
            ("CONDITION",  0),
            ("ACTUAL",    84),
            ("AI RESULT", 96),
            ("CONF.",     64),
        ]):
            kw = dict(font=("Segoe UI", 6, "bold"), bg=SEC_BG, fg=TXT3,
                      pady=6, anchor="center" if ci else "w",
                      padx=12 if ci == 0 else 0)
            if w:
                fr = tk.Frame(chdr, bg=SEC_BG, width=w)
                fr.grid(row=0, column=ci, sticky="ns")
                fr.pack_propagate(False)
                tk.Label(fr, **kw, text=txt).pack(fill="x", expand=True)
            else:
                tk.Label(chdr, **kw, text=txt).grid(row=0, column=ci, sticky="ew")
        chdr.columnconfigure(0, weight=1)

        tk.Frame(lf, bg=BORDER, height=1).pack(fill="x")

        # scrollable table
        host = tk.Frame(lf, bg=CARD_BG)
        host.pack(fill="both", expand=True)
        cv  = tk.Canvas(host, bg=CARD_BG, highlightthickness=0, bd=0)
        vsb = tk.Scrollbar(host, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        cv.pack(side="left", fill="both", expand=True)
        self.table_f = tk.Frame(cv, bg=CARD_BG)
        cv.create_window((0, 0), window=self.table_f, anchor="nw", tags="W")
        self.table_f.bind("<Configure>",
                          lambda _: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>", lambda e: cv.itemconfig("W", width=e.width))
        cv.bind_all("<MouseWheel>",
                    lambda e: cv.yview_scroll(int(-e.delta / 120), "units"))
        self._build_condition_rows()

        # stats strip
        tk.Frame(lf, bg=BORDER, height=1).pack(fill="x")
        self.stats_frame = tk.Frame(lf, bg=CARD_BG)
        self.stats_frame.pack(fill="x")
        self._build_stats_strip()

        # legend + status
        tk.Frame(lf, bg=BORDER, height=1).pack(fill="x")
        leg = tk.Frame(lf, bg=SEC_BG, pady=5)
        leg.pack(fill="x")
        for swatch, label in [
            (TP_ROW, "Both detected"),
            (FN_ROW, "AI missed"),
            (FP_ROW, "AI over-flagged"),
        ]:
            tk.Frame(leg, bg=swatch, width=9, height=9).pack(side="left", padx=(8, 3))
            tk.Label(leg, text=label, font=("Segoe UI", 6),
                     bg=SEC_BG, fg=TXT3).pack(side="left", padx=(0, 10))

        tk.Frame(lf, bg=BORDER, height=1).pack(fill="x")
        tk.Label(lf, textvariable=self.sv_status,
                 font=("Segoe UI", 7), bg=CARD_BG, fg=TXT2,
                 anchor="w", padx=12, pady=6).pack(fill="x")

    def _build_stats_strip(self):
        stat_defs = [
            ("tp",    "TP",  "#0D2B1A", "#34D399"),
            ("fn",    "FN",  "#2B1D00", "#FBBF24"),
            ("fp",    "FP",  "#2B0D0D", "#F87171"),
            ("tn",    "TN",  CARD_BG,   TXT2),
            ("agree", "Agr", "#162032", ACCENT),
        ]
        for key, lbl, sbg, sfg in stat_defs:
            tile = tk.Frame(self.stats_frame, bg=sbg,
                            highlightthickness=1, highlightbackground=BORDER)
            tile.pack(side="left", expand=True, fill="both", padx=1, pady=4)
            n = tk.Label(tile, text="—",
                         font=("Segoe UI", 14, "bold"), bg=sbg, fg=sfg)
            n.pack(pady=(5, 0))
            tk.Label(tile, text=lbl,
                     font=("Segoe UI", 6, "bold"), bg=sbg, fg=sfg).pack(pady=(0, 4))
            self._stat_labels[key] = n

    # ── condition rows ─────────────────────────────────────────────────────────
    def _build_condition_rows(self):
        tf        = self.table_f
        row_colors = [CARD_BG, "#263347"]
        idx       = 0
        for grp_name, cols in DISPLAY_GROUPS:
            sec = tk.Frame(tf, bg=SEC_BG)
            sec.pack(fill="x", pady=(6, 0))
            tk.Frame(sec, bg=SEC_ACC, width=3).pack(side="left", fill="y")
            tk.Label(sec, text=f"  {grp_name.upper()}",
                     font=("Segoe UI", 7, "bold"),
                     bg=SEC_BG, fg=SEC_FG, pady=5).pack(side="left")

            for col in cols:
                bg = row_colors[idx % 2]; idx += 1
                tk.Frame(tf, bg=BORDER, height=1).pack(fill="x")

                row = tk.Frame(tf, bg=bg)
                row.pack(fill="x")
                row.columnconfigure(0, weight=1)

                name_cell = tk.Frame(row, bg=bg)
                name_cell.grid(row=0, column=0, sticky="ew",
                               padx=(12, 6), pady=(6, 5))

                perf_tier, perf_tag = LABEL_PERF[col]
                tag_fg = CONF_HI   if perf_tier == "EXCELLENT" else CONF_MED
                tag_bg = "#064E2A" if perf_tier == "EXCELLENT" else "#451A00"

                tk.Label(name_cell, text=LABEL_DISPLAY[col],
                         font=("Segoe UI", 8, "bold"),
                         bg=bg, fg=TXT, anchor="w").pack(fill="x")

                tag_row = tk.Frame(name_cell, bg=bg)
                tag_row.pack(fill="x", pady=(1, 0))
                tk.Label(tag_row, text=f" {perf_tier} ",
                         font=("Segoe UI", 6, "bold"),
                         bg=tag_bg, fg=tag_fg, padx=3).pack(side="left")
                tk.Label(tag_row, text=f"  {perf_tag}",
                         font=("Segoe UI", 6),
                         bg=bg, fg=TXT3).pack(side="left")

                act_cell = tk.Frame(row, bg=bg, width=84)
                act_cell.grid(row=0, column=1, sticky="ns")
                act_cell.pack_propagate(False)
                act_l = tk.Label(act_cell, text="—",
                                 font=("Segoe UI", 7, "bold"),
                                 bg=PEND_BG, fg=PEND_FG, padx=4)
                act_l.pack(expand=True)

                ai_cell = tk.Frame(row, bg=bg, width=96)
                ai_cell.grid(row=0, column=2, sticky="ns")
                ai_cell.pack_propagate(False)
                ai_l = tk.Label(ai_cell, text="—",
                                font=("Segoe UI", 7, "bold"),
                                bg=PEND_BG, fg=PEND_FG, padx=4)
                ai_l.pack(expand=True)

                conf_cell = tk.Frame(row, bg=bg, width=64)
                conf_cell.grid(row=0, column=3, sticky="ns", padx=(0, 6))
                conf_cell.pack_propagate(False)
                conf_pct = tk.Label(conf_cell, text="—",
                                    font=("Segoe UI", 10, "bold"),
                                    bg=bg, fg=TXT3, anchor="center")
                conf_pct.pack(expand=True)
                conf_bar = tk.Canvas(conf_cell, width=44, height=5,
                                     bg=bg, highlightthickness=0, bd=0)
                conf_bar.pack(pady=(0, 6))
                self._draw_bar(conf_bar, None, bg)

                self.row_refs[col] = dict(
                    row=row, name_cell=name_cell, orig_bg=bg,
                    act_cell=act_cell, act_l=act_l,
                    ai_cell=ai_cell, ai_l=ai_l,
                    conf_cell=conf_cell, conf_pct=conf_pct, conf_bar=conf_bar)

        tk.Frame(tf, bg=BORDER, height=1).pack(fill="x")

    # ── ECG panel ─────────────────────────────────────────────────────────────
    def _ecg_panel(self, parent):
        rf = tk.Frame(parent, bg=WHITE,
                      highlightthickness=1, highlightbackground=BORDER)
        rf.grid(row=0, column=1, sticky="nsew")
        rf.rowconfigure(0, weight=1)
        rf.columnconfigure(0, weight=1)

        self.fig = plt.figure(figsize=(13, 6.8))
        self.fig.patch.set_facecolor(WHITE)
        gs = GridSpec(4, 4, figure=self.fig,
                      height_ratios=[1, 1, 1, 0.50],
                      hspace=0.003, wspace=0,
                      left=0.003, right=0.997, top=0.935, bottom=0.010)
        self.axes      = [[self.fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(3)]
        self.rhythm_ax = self.fig.add_subplot(gs[3, :])

        self.mpl = FigureCanvasTkAgg(self.fig, master=rf)
        self.mpl.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._ecg_empty()

    # ── confidence bar helper ──────────────────────────────────────────────────
    @staticmethod
    def _draw_bar(canvas, value, bg=CARD_BG):
        canvas.delete("all")
        w = int(canvas["width"])
        canvas.create_rectangle(0, 0, w, 5, fill=TXT3, outline="")
        if value is not None and value > 0:
            col  = CONF_HI if value >= 0.70 else CONF_MED if value >= 0.40 else CONF_LO
            fill = max(3, int(w * value))
            canvas.create_rectangle(0, 0, fill, 5, fill=col, outline="")

    # ── ECG plot helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _ax_style(ax, lead_name):
        ax.set_facecolor("#fff8f8")
        for sp in ax.spines.values():
            sp.set_color("#ffbbbb"); sp.set_linewidth(0.6); sp.set_visible(True)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.text(0.012, 0.94, lead_name,
                transform=ax.transAxes, fontsize=8, fontweight="bold",
                color="#0a1f3c", va="top", ha="left", zorder=5,
                bbox=dict(boxstyle="square,pad=0.12", fc="#fff8f8", ec="none", alpha=0.9))

    @staticmethod
    def _ax_grid(ax):
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.04))
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(MultipleLocator(0.2))
        ax.grid(True, which="minor", color="#ffd0d0", lw=0.22, zorder=0)
        ax.grid(True, which="major", color="#ffaaaa", lw=0.50, zorder=0)
        ax.set_axisbelow(True)

    def _ecg_empty(self):
        for r in range(3):
            for c in range(4):
                ax = self.axes[r][c]; ax.cla()
                self._ax_style(ax, LEAD_NAMES[WAVE_3x4[r][c]])
                ax.text(0.5, 0.5, "—", transform=ax.transAxes,
                        ha="center", va="center", color="#ccc0c0", fontsize=13)
        self.rhythm_ax.cla()
        self._ax_style(self.rhythm_ax, "II  —  Rhythm Strip  (10 s)")
        self.rhythm_ax.text(0.5, 0.5, "—", transform=self.rhythm_ax.transAxes,
                            ha="center", va="center", color="#ccc0c0", fontsize=13)
        self.fig.suptitle("12-Lead ECG  —  Awaiting file",
                          color=TXT2, fontsize=10, fontweight="bold")
        self.mpl.draw()

    def _ecg_draw(self):
        if self.signal is None:
            return
        sig = self.signal
        DS  = 2
        t_c = np.arange(0, SAMPLES_PER_COL, DS) / 500.0
        for r in range(3):
            for c in range(4):
                ax   = self.axes[r][c]
                lead = WAVE_3x4[r][c]
                y    = sig[lead, c * SAMPLES_PER_COL:(c + 1) * SAMPLES_PER_COL:DS].copy()
                std  = y.std()
                if std > 1e-6:
                    y = np.clip(y / std, -2.8, 2.8)
                ax.cla()
                self._ax_style(ax, LEAD_NAMES[lead])
                self._ax_grid(ax)
                ax.axhline(0, color="#ffbbbb", lw=0.4, zorder=1)
                ax.plot(t_c, y, color="#c0252d", lw=0.85, rasterized=True, zorder=2)
                ax.set_xlim(0, t_c[-1]); ax.set_ylim(-3.2, 3.2)

        DS_r = 4
        t_r  = np.arange(0, N_SAMPLES, DS_r) / 500.0
        y    = sig[1, ::DS_r].copy()
        std  = y.std()
        if std > 1e-6:
            y = np.clip(y / std, -2.8, 2.8)
        ax = self.rhythm_ax; ax.cla()
        self._ax_style(ax, "II  —  Rhythm Strip  (10 s)")
        self._ax_grid(ax)
        ax.axhline(0, color="#ffbbbb", lw=0.4, zorder=1)
        ax.plot(t_r, y, color="#c0252d", lw=0.85, rasterized=True, zorder=2)
        ax.set_xlim(0, t_r[-1]); ax.set_ylim(-3.2, 3.2)

        self.fig.suptitle(
            f"12-Lead ECG   |   {self.ecg_file or ''}   |   500 Hz  ·  10 s",
            color="#0F172A", fontsize=10, fontweight="bold")
        self.mpl.draw()

    # ── browse + auto-analyse ──────────────────────────────────────────────────
    def browse(self):
        for start in (DATA_REDUCED, PATIENT_DIR, SORTED_DIR, BASE):
            if start.exists():
                break
        p = filedialog.askopenfilename(
            title="Select ECG Header File (.hea)",
            initialdir=str(start),
            filetypes=[("WFDB Header", "*.hea"), ("All files", "*.*")])
        if not p:
            return
        self.hea_path = Path(p)
        self._reset_rows()
        self.btn.config(state="disabled", text="Analysing …")
        self.sv_status.set("Loading ECG and running analysis …")
        threading.Thread(target=self._load, daemon=True).start()

    def _reset_rows(self):
        for w in self.row_refs.values():
            bg = w["orig_bg"]
            for k in ("row", "name_cell", "act_cell", "ai_cell", "conf_cell"):
                w[k].config(bg=bg)
            w["act_l"].config(text="—",   bg=PEND_BG, fg=PEND_FG)
            w["ai_l"].config(text="—",    bg=PEND_BG, fg=PEND_FG)
            w["conf_pct"].config(text="—", fg=TXT3,    bg=bg)
            w["conf_bar"].config(bg=bg)
            self._draw_bar(w["conf_bar"], None, bg)
        for lbl in self._stat_labels.values():
            lbl.config(text="—")

    def _load(self):
        try:
            raw           = read_ecg(self.hea_path)
            self.signal   = raw
            self.ecg_file = self.hea_path.with_suffix(".dat").name
            pid           = get_patient_id(self.ecg_file)
            self.after(0, lambda: self.sv_patient.set(f"Patient  {pid}"))
            self.after(0, self._ecg_draw)
            actual = get_actual_labels(self.ecg_file)
            if self.model:
                preds, confs = run_inference(self.model, raw, self.thresholds, self.ck)
                n_det = sum(1 for c in DISPLAY_COLS if preds.get(c))
                self.after(0, lambda: self._fill(actual, preds, confs))
                self.after(0, lambda: self._update_stats(actual, preds))
                self.after(0, lambda: self.sv_status.set(
                    f"Done  ·  {n_det} of 7 conditions detected"
                    + ("" if actual else "  ·  patient not in database")))
            else:
                self.after(0, lambda: self._fill(actual, {}, {}))
                self.after(0, lambda: self.sv_status.set(
                    "Model not loaded — run train.py first"))
        except Exception as exc:
            self.after(0, lambda: self.sv_status.set(f"Error: {exc}"))
            self.after(0, lambda: messagebox.showerror("Load error", str(exc)))
        finally:
            self.after(0, lambda: self.btn.config(
                state="normal", text="Open ECG  ▶"))

    # ── fill condition table ───────────────────────────────────────────────────
    def _fill(self, actual, preds, confs):
        for col, w in self.row_refs.items():
            a    = actual.get(col)
            p    = preds.get(col)
            conf = confs.get(col)
            orig = w["orig_bg"]

            if   a and p is True:  bg = TP_ROW
            elif a and p is False: bg = FN_ROW
            elif not a and p:      bg = FP_ROW
            else:                  bg = orig

            for k in ("row", "name_cell", "act_cell", "ai_cell", "conf_cell"):
                w[k].config(bg=bg)
            for child in w["name_cell"].winfo_children():
                try:
                    child.config(bg=bg)
                except tk.TclError:
                    pass
            for child in w["name_cell"].winfo_children():
                for sub in getattr(child, "winfo_children", lambda: [])():
                    try:
                        sub.config(bg=bg)
                    except tk.TclError:
                        pass

            if a is True:
                w["act_l"].config(text="CONFIRMED", bg=ACT_DET_BG, fg=ACT_DET_FG)
            elif a is False:
                w["act_l"].config(text="NEGATIVE",  bg=ACT_NOR_BG, fg=ACT_NOR_FG)
            else:
                w["act_l"].config(text="—",         bg=PEND_BG,    fg=PEND_FG)

            if p is True:
                w["ai_l"].config(text="DETECTED", bg=AI_DET_BG, fg=AI_DET_FG)
            elif p is False:
                w["ai_l"].config(text="NORMAL",   bg=AI_NOR_BG, fg=AI_NOR_FG)
            else:
                w["ai_l"].config(text="—",         bg=PEND_BG,   fg=PEND_FG)

            if conf is not None:
                pct   = int(conf * 100)
                col_c = CONF_HI if conf >= 0.70 else CONF_MED if conf >= 0.40 else CONF_LO
                w["conf_pct"].config(text=f"{pct}%", fg=col_c, bg=bg)
                w["conf_bar"].config(bg=bg)
                self._draw_bar(w["conf_bar"], conf, bg)
            else:
                w["conf_pct"].config(text="—", fg=TXT3, bg=bg)
                w["conf_bar"].config(bg=bg)
                self._draw_bar(w["conf_bar"], None, bg)

    # ── stats strip ────────────────────────────────────────────────────────────
    def _update_stats(self, actual, preds):
        if not preds:
            return
        tp    = sum(1 for c in DISPLAY_COLS if actual.get(c) and preds.get(c))
        fn    = sum(1 for c in DISPLAY_COLS if actual.get(c) and not preds.get(c))
        fp    = sum(1 for c in DISPLAY_COLS if not actual.get(c) and preds.get(c))
        tn    = sum(1 for c in DISPLAY_COLS if not actual.get(c) and not preds.get(c))
        agree = int(100 * (tp + tn) / max(1, len(DISPLAY_COLS)))
        self._stat_labels["tp"].config(text=str(tp))
        self._stat_labels["fn"].config(text=str(fn))
        self._stat_labels["fp"].config(text=str(fp))
        self._stat_labels["tn"].config(text=str(tn))
        self._stat_labels["agree"].config(text=f"{agree}%")


if __name__ == "__main__":
    App().mainloop()
