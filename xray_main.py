"""
xray_main.py  —  Cardiac X-ray Clinical Decision Support
Single button: browse + auto-analyse.
"""

import json, threading
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image

ROOT      = Path(__file__).parent
MODEL_DIR = ROOT / "xray_model"
SUMMARY   = ROOT / "final_patient_dataset_summary.xlsx"

# ── palette ───────────────────────────────────────────────────────────────────
NAVY        = "#1E3A5F"
NAVY_DARK   = "#152D4B"
ACCENT      = "#2563EB"
ACCENT_LT   = "#DBEAFE"
XRAY_BG     = "#070B11"
PAGE_BG     = "#F1F5F9"
CARD        = "#FFFFFF"
BORDER      = "#E2E8F0"
BORDER_DARK = "#CBD5E1"
PANEL_HDR   = "#F8FAFC"

TEXT        = "#111827"
TEXT_SEC    = "#374151"
TEXT_MUT    = "#6B7280"
TEXT_DIM    = "#9CA3AF"
TEXT_W      = "#FFFFFF"

# row highlight backgrounds
TP_BG   = "#F0FDF4"
FN_BG   = "#FFFBEB"
FP_BG   = "#FEF2F2"
INFO_BG = "#EFF6FF"

# badge colours (solid pill)
B_CONF_BG  = "#D97706"; B_CONF_FG  = "#FFFFFF"   # clinical confirmed – amber
B_NEG_BG   = "#E5E7EB"; B_NEG_FG   = "#374151"   # negative – grey
B_NR_FG    = "#9CA3AF"                            # not in report – muted text
B_DET_BG   = "#059669"; B_DET_FG   = "#FFFFFF"   # AI detected – green
B_NDET_BG  = "#E5E7EB"; B_NDET_FG  = "#374151"   # AI not detected – grey

BAR_DET  = "#059669"
BAR_NDET = "#D1D5DB"

FONT = "Segoe UI"
MONO = "Consolas"

LABEL_COLS  = ["Atelectasis", "Cardiomegaly", "Edema",
               "Lung Opacity", "No Finding",  "Pleural Effusion"]
NUM_CLASSES = 6

# Only these 3 are shown in the UI (highest accuracy on full dataset)
DISPLAY_COLS = ["Cardiomegaly", "Edema", "Pleural Effusion"]
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

LABEL_INFO = {
    "Atelectasis":      "Lung collapse from cardiac compression",
    "Cardiomegaly":     "Enlarged heart — cardiomyopathy / HF",
    "Edema":            "Fluid in lungs from failing heart",
    "Lung Opacity":     "Opacity pattern — cardiogenic oedema",
    "No Finding":       "No abnormality detected — normal study",
    "Pleural Effusion": "Fluid around lungs — left heart failure",
}


# ── model ─────────────────────────────────────────────────────────────────────
class XrayNet(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", num_classes=NUM_CLASSES):
        super().__init__()
        self._is_densenet = "densenet" in backbone_name.lower()

        if self._is_densenet:
            import torchxrayvision as txrv
            self.backbone = txrv.models.DenseNet(weights=backbone_name)
            in_feat = 1024
        elif backbone_name == "efficientnet_b4":
            from torchvision.models import efficientnet_b4
            self.backbone = efficientnet_b4(weights=None)
            in_feat = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            from torchvision.models import efficientnet_b0
            self.backbone = efficientnet_b0(weights=None)
            in_feat = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(in_feat, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        if self._is_densenet:
            feat_map = self.backbone.features(x)
            feat_map = F.relu(feat_map, inplace=True)
            pooled   = F.adaptive_avg_pool2d(feat_map, (1, 1))
            feats    = pooled.view(pooled.size(0), -1)
        else:
            feats = self.backbone(x)
        return self.head(feats)


def preprocess(img, img_size=224, backbone_name="efficientnet_b0"):
    if "densenet" in backbone_name.lower():
        arr = np.array(img.convert("L").resize((img_size, img_size)), dtype=np.float32)
        arr = (arr / 255.0) * 2048.0 - 1024.0
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])(img.convert("RGB")).unsqueeze(0)


# ── database ──────────────────────────────────────────────────────────────────
_db = None

def load_db():
    global _db
    if _db is None:
        _db = pd.read_excel(SUMMARY, engine="openpyxl")
        _db.columns = _db.columns.str.strip()
        _db["cxr_study_id"] = (_db["cxr_study_id"].astype(str)
                                .str.split(".").str[0])
    return _db


def lookup_patient(study_id):
    db  = load_db()
    row = db[db["cxr_study_id"] == study_id]
    if row.empty:
        return {}
    row  = row.iloc[0]
    info = {
        "subject_id": str(int(row.get("subject_id", 0))),
        "age":    (str(int(row.get("age", 0)))
                   if pd.notna(row.get("age")) else "—"),
        "gender": str(row.get("gender", "—")).capitalize(),
        "view":   str(row.get("cxr_ViewPosition", "—")),
    }
    labels = {}
    for col in LABEL_COLS:
        v = row.get(col, float("nan"))
        labels[col] = ("unknown" if pd.isna(v)
                       else "present" if float(v) == 1.0 else "absent")
    return {"info": info, "labels": labels}


# ── confidence bar ─────────────────────────────────────────────────────────────
class ConfBar(tk.Canvas):
    W, H = 160, 8

    def __init__(self, parent, **kw):
        kw.setdefault("bg", CARD)
        super().__init__(parent, width=self.W, height=self.H,
                         highlightthickness=0, **kw)
        self.create_rectangle(0, 0, self.W, self.H, fill="#E5E7EB", outline="")
        self._bar = self.create_rectangle(0, 0, 0, self.H,
                                          fill=BAR_NDET, outline="")

    def set(self, prob, color=BAR_NDET):
        w = int(self.W * max(0.0, min(1.0, prob)))
        self.coords(self._bar, 0, 0, w, self.H)
        self.itemconfig(self._bar, fill=color)


# ── application ───────────────────────────────────────────────────────────────
class DiagnosticApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Cardiac X-ray AI  —  Clinical Decision Support")
        self.configure(bg=PAGE_BG)
        self.state("zoomed")
        self.minsize(1200, 750)

        self._path         = None
        self._model        = None
        self._thresholds   = {}
        self._img_size     = 224
        self._backbone_name = "efficientnet_b0"
        self._pil_img      = None

        self._build_header()
        body = tk.Frame(self, bg=PAGE_BG)
        body.pack(fill="both", expand=True, padx=12, pady=(8, 0))
        self._build_xray_panel(body)
        self._build_results_panel(body)
        self._build_status_bar()
        self._load_model()

    # ── header ────────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self, bg=NAVY, height=68)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        left = tk.Frame(hdr, bg=NAVY)
        left.pack(side="left", padx=24, pady=14)

        tk.Label(left, text="CARDIAC X-RAY AI",
                 bg=NAVY, fg=TEXT_W,
                 font=(FONT, 16, "bold")).pack(anchor="w")
        tk.Label(left, text="Clinical Decision Support  ·  EfficientNet-B4  ·  3 Cardiac Conditions",
                 bg=NAVY, fg="#7AA3CC",
                 font=(FONT, 10)).pack(anchor="w")

        right = tk.Frame(hdr, bg=NAVY)
        right.pack(side="right", padx=24, pady=18)

        self.sv_model_tag = tk.StringVar(value="")
        tk.Label(right, textvariable=self.sv_model_tag,
                 bg=NAVY, fg="#4A7FA8",
                 font=(MONO, 9)).pack(side="left", padx=(0, 20))

        self.btn = tk.Button(
            right, text="  Open & Analyse  ▶",
            font=(FONT, 11, "bold"), bg=ACCENT, fg=TEXT_W,
            activebackground="#1D4ED8", activeforeground=TEXT_W,
            relief="flat", cursor="hand2", bd=0,
            padx=20, pady=10,
            command=self.open_and_analyse)
        self.btn.pack(side="left")

    # ── x-ray panel ───────────────────────────────────────────────────────────
    def _build_xray_panel(self, parent):
        frame = tk.Frame(parent, bg=XRAY_BG,
                         highlightthickness=2,
                         highlightbackground=BORDER_DARK)
        frame.pack(side="left", fill="both", expand=True,
                   padx=(0, 8), pady=(0, 12))

        top = tk.Frame(frame, bg="#0D1117", height=36)
        top.pack(fill="x")
        top.pack_propagate(False)

        self.sv_study = tk.StringVar(value="NO STUDY LOADED")
        tk.Label(top, textvariable=self.sv_study,
                 bg="#0D1117", fg="#475569",
                 font=(MONO, 10)).pack(side="left", padx=14, pady=8)

        self.sv_view_badge = tk.StringVar(value="")
        tk.Label(top, textvariable=self.sv_view_badge,
                 bg="#0D1117", fg="#38BDF8",
                 font=(FONT, 10, "bold")).pack(side="right", padx=14)

        self.fig = plt.figure(facecolor=XRAY_BG)
        self.ax  = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_facecolor(XRAY_BG)
        self.ax.axis("off")
        self.ax.text(0.5, 0.5, "Select a chest X-ray to begin",
                     ha="center", va="center", color="#1E293B",
                     fontsize=14, transform=self.ax.transAxes)

        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.mpl_canvas.get_tk_widget().pack(fill="both", expand=True)

    # ── results panel ─────────────────────────────────────────────────────────
    def _build_results_panel(self, parent):
        panel = tk.Frame(parent, bg=CARD, width=500,
                         highlightthickness=2,
                         highlightbackground=BORDER_DARK)
        panel.pack(side="right", fill="y", pady=(0, 12))
        panel.pack_propagate(False)

        # patient card
        pt = tk.Frame(panel, bg=NAVY)
        pt.pack(fill="x")

        inner = tk.Frame(pt, bg=NAVY)
        inner.pack(fill="x", padx=18, pady=14)

        r1 = tk.Frame(inner, bg=NAVY)
        r1.pack(fill="x")
        self.sv_pid = tk.StringVar(value="No patient loaded")
        tk.Label(r1, textvariable=self.sv_pid,
                 bg=NAVY, fg=TEXT_W,
                 font=(FONT, 14, "bold")).pack(side="left")
        self.sv_view_tag = tk.StringVar(value="")
        tk.Label(r1, textvariable=self.sv_view_tag,
                 bg=NAVY, fg="#38BDF8",
                 font=(FONT, 11, "bold")).pack(side="right")

        self.sv_age = tk.StringVar(value="")
        tk.Label(inner, textvariable=self.sv_age,
                 bg=NAVY, fg="#7AA3CC",
                 font=(FONT, 10)).pack(anchor="w", pady=(4, 0))

        # summary bar
        self.sum_frame = tk.Frame(panel, bg=PAGE_BG)
        self.sum_frame.pack(fill="x")
        inner_s = tk.Frame(self.sum_frame, bg=PAGE_BG)
        inner_s.pack(fill="x", padx=18, pady=12)

        self.sv_sum = tk.StringVar(value="No analysis yet")
        self.lbl_sum = tk.Label(inner_s, textvariable=self.sv_sum,
                                 bg=PAGE_BG, fg=TEXT_MUT,
                                 font=(FONT, 13, "bold"))
        self.lbl_sum.pack(side="left")

        self.sv_sum_sub = tk.StringVar(value="")
        self.lbl_sum_sub = tk.Label(inner_s, textvariable=self.sv_sum_sub,
                                     bg=PAGE_BG, fg=TEXT_MUT,
                                     font=(FONT, 10))
        self.lbl_sum_sub.pack(side="left")

        # column headers
        tk.Frame(panel, bg=BORDER, height=1).pack(fill="x")
        chdr = tk.Frame(panel, bg=PANEL_HDR)
        chdr.pack(fill="x")

        for i, (txt, anc) in enumerate([
            ("CONDITION",    "w"),
            ("CLINICAL",     "center"),
            ("AI RESULT",    "center"),
        ]):
            tk.Label(chdr, text=txt,
                     bg=PANEL_HDR, fg=TEXT_DIM,
                     font=(FONT, 9, "bold"), anchor=anc,
                     padx=14 if i == 0 else 0, pady=8
                     ).grid(row=0, column=i, sticky="ew")
        chdr.columnconfigure(0, weight=1)
        chdr.columnconfigure(1, minsize=110)
        chdr.columnconfigure(2, minsize=140)
        tk.Frame(panel, bg=BORDER, height=1).pack(fill="x")

        # scrollable condition rows
        host = tk.Frame(panel, bg=CARD)
        host.pack(fill="both", expand=True)
        cnv = tk.Canvas(host, bg=CARD, highlightthickness=0)
        vsb = tk.Scrollbar(host, orient="vertical", command=cnv.yview)
        self._rows_frame = tk.Frame(cnv, bg=CARD)
        self._rows_frame.bind(
            "<Configure>",
            lambda e: cnv.configure(scrollregion=cnv.bbox("all")))
        cnv.create_window((0, 0), window=self._rows_frame, anchor="nw")
        cnv.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        cnv.pack(side="left", fill="both", expand=True)
        cnv.bind_all("<MouseWheel>",
                     lambda e: cnv.yview_scroll(int(-e.delta / 120), "units"))

        self.row_refs = {}
        for i, col in enumerate(DISPLAY_COLS):
            rbg = CARD if i % 2 == 0 else PANEL_HDR
            self._make_row(col, rbg)

        # legend
        tk.Frame(panel, bg=BORDER_DARK, height=1).pack(fill="x", side="bottom")
        leg = tk.Frame(panel, bg=PANEL_HDR, pady=8)
        leg.pack(fill="x", side="bottom")

        for bg_c, txt in [
            (TP_BG,   "Record + AI both detected"),
            (FN_BG,   "In record — AI missed"),
            (FP_BG,   "Absent — AI flagged"),
            (INFO_BG, "AI detected — not in report"),
        ]:
            tk.Frame(leg, bg=bg_c, width=14, height=14,
                     highlightthickness=1,
                     highlightbackground=BORDER_DARK
                     ).pack(side="left", padx=(10, 4))
            tk.Label(leg, text=txt, bg=PANEL_HDR, fg=TEXT_SEC,
                     font=(FONT, 9)).pack(side="left", padx=(0, 10))

    def _make_row(self, col, rbg):
        tk.Frame(self._rows_frame, bg=BORDER, height=1).pack(fill="x")
        row = tk.Frame(self._rows_frame, bg=rbg)
        row.pack(fill="x")
        row.columnconfigure(0, weight=1)
        row.columnconfigure(1, minsize=110)
        row.columnconfigure(2, minsize=140)

        # condition name + description
        nf = tk.Frame(row, bg=rbg)
        nf.grid(row=0, column=0, sticky="ew", padx=(14, 6), pady=(10, 9))
        tk.Label(nf, text=col, bg=rbg, fg=TEXT,
                 font=(FONT, 12, "bold"), anchor="w").pack(anchor="w")
        tk.Label(nf, text=LABEL_INFO.get(col, ""),
                 bg=rbg, fg=TEXT_MUT,
                 font=(FONT, 9), anchor="w",
                 wraplength=200).pack(anchor="w")

        # clinical record badge
        act_var = tk.StringVar(value="—")
        act_lbl = tk.Label(row, textvariable=act_var,
                           bg=rbg, fg=TEXT_MUT,
                           font=(FONT, 10, "bold"),
                           anchor="center", padx=8, pady=4)
        act_lbl.grid(row=0, column=1, padx=4)

        # AI result: badge + percentage + bar
        ai_frame = tk.Frame(row, bg=rbg)
        ai_frame.grid(row=0, column=2, padx=(4, 14), pady=8, sticky="ew")

        top_r = tk.Frame(ai_frame, bg=rbg)
        top_r.pack(anchor="w")

        ai_lbl_var = tk.StringVar(value="—")
        ai_pct_var = tk.StringVar(value="")
        ai_lbl = tk.Label(top_r, textvariable=ai_lbl_var,
                          bg=rbg, fg=TEXT_MUT,
                          font=(FONT, 10, "bold"),
                          padx=8, pady=3)
        ai_lbl.pack(side="left")

        tk.Label(top_r, textvariable=ai_pct_var,
                 bg=rbg, fg=TEXT_MUT,
                 font=(FONT, 10)).pack(side="left", padx=(6, 0))

        bar = ConfBar(ai_frame, bg=rbg)
        bar.pack(anchor="w", pady=(4, 0))

        self.row_refs[col] = dict(
            row=row, rbg=rbg, nf=nf,
            act_var=act_var, act_lbl=act_lbl,
            ai_lbl_var=ai_lbl_var, ai_pct_var=ai_pct_var,
            ai_lbl=ai_lbl, top_r=top_r,
            bar=bar, ai_frame=ai_frame,
        )

    # ── status bar ────────────────────────────────────────────────────────────
    def _build_status_bar(self):
        bar = tk.Frame(self, bg=NAVY_DARK, height=30)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self.sv_status = tk.StringVar(value="Ready  —  click Open & Analyse to load a study")
        tk.Label(bar, textvariable=self.sv_status,
                 bg=NAVY_DARK, fg="#4A7FA8",
                 font=(FONT, 9), anchor="w", padx=16).pack(side="left", fill="y")

        tk.Label(bar,
                 text="Cardiac X-ray AI  ·  EfficientNet-B4  ·  3 Conditions",
                 bg=NAVY_DARK, fg="#2D4F6E",
                 font=(FONT, 9), anchor="e", padx=16).pack(side="right", fill="y")

    # ── model loading ─────────────────────────────────────────────────────────
    def _load_model(self):
        best_pt = MODEL_DIR / "best_model.pt"
        if not best_pt.exists():
            self._sv(self.sv_status,
                     "No model found  —  run xray_train.py first")
            return

        state = torch.load(best_pt, map_location="cpu", weights_only=False)
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", ""): v for k, v in state.items()}

        key = "backbone.features.8.0.weight"
        if key in state:
            backbone = ("efficientnet_b4"
                        if state[key].shape[0] == 1792
                        else "efficientnet_b0")
            img_size = 320 if backbone == "efficientnet_b4" else 224
        else:
            cfg = MODEL_DIR / "config.json"
            backbone, img_size = "efficientnet_b0", 224
            if cfg.exists():
                c = json.loads(cfg.read_text(encoding="utf-8"))
                backbone = c.get("backbone", backbone)
                img_size = c.get("img_size", img_size)

        self._img_size      = img_size
        self._backbone_name = backbone

        lbl_path = MODEL_DIR / "label_classes.json"
        global LABEL_COLS, NUM_CLASSES
        if lbl_path.exists():
            LABEL_COLS  = json.loads(lbl_path.read_text(encoding="utf-8"))
            NUM_CLASSES = len(LABEL_COLS)

        thr_path = MODEL_DIR / "thresholds.json"
        if thr_path.exists():
            self._thresholds = json.loads(thr_path.read_text(encoding="utf-8"))

        (MODEL_DIR / "config.json").write_text(
            json.dumps({"backbone": backbone,
                        "img_size": img_size,
                        "num_classes": NUM_CLASSES}, indent=2),
            encoding="utf-8")

        try:
            self._model = XrayNet(backbone_name=backbone,
                                  num_classes=NUM_CLASSES)
            self._model.load_state_dict(state)
            self._model.eval()
            n = sum(p.numel() for p in self._model.parameters())
            self._sv(self.sv_model_tag, f"{backbone}  ·  {n:,} params")
            self._sv(self.sv_status,
                     f"Model ready  ·  {backbone}  ·  {n:,} parameters")
        except Exception as e:
            self._sv(self.sv_status, f"Model load error: {e}")

    # ── single button: browse + auto-analyse ──────────────────────────────────
    def open_and_analyse(self):
        path = filedialog.askopenfilename(
            title="Select Chest X-ray PNG",
            initialdir=str(ROOT / "Final_Dataset"),
            filetypes=[("PNG images", "*.png"), ("All files", "*.*")],
        )
        if not path:
            return

        self._path = Path(path)
        try:
            self._pil_img = Image.open(self._path)
        except Exception as e:
            messagebox.showerror("Image error", str(e))
            return

        self._show_image(self._pil_img)

        stem     = self._path.stem
        study_id = stem.replace("cxr_", "")
        self._sv(self.sv_study, f"STUDY  {stem.upper()}")

        data = lookup_patient(study_id)
        if data:
            info = data["info"]
            self._sv(self.sv_pid,
                     f"Patient  {info.get('subject_id', '—')}")
            self._sv(self.sv_age,
                     f"Age {info.get('age', '—')}   ·   "
                     f"Gender {info.get('gender', '—')}")
            self._sv(self.sv_view_tag, info.get("view", ""))
        else:
            self._sv(self.sv_pid, "Patient ID not found in database")
            self._sv(self.sv_age, "")
            self._sv(self.sv_view_tag, "")

        self._reset_rows()

        if self._model is None:
            self._sv(self.sv_status,
                     "No model loaded — run xray_train.py first")
            return

        self.btn.config(state="disabled", text="  Analysing …")
        self._sv(self.sv_status, "Running inference …")
        threading.Thread(target=self._run, daemon=True).start()

    def _show_image(self, pil_img):
        self.ax.cla()
        self.ax.axis("off")
        self.ax.set_facecolor(XRAY_BG)
        self.fig.set_facecolor(XRAY_BG)
        arr = np.array(pil_img.convert("L"))
        self.ax.imshow(arr, cmap="gray", aspect="equal",
                       interpolation="bilinear")
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.mpl_canvas.draw_idle()

    # ── inference thread ──────────────────────────────────────────────────────
    def _run(self):
        try:
            tensor = preprocess(self._pil_img, self._img_size, self._backbone_name)
            with torch.no_grad():
                probs = torch.sigmoid(
                    self._model(tensor)).squeeze(0).tolist()

            preds = {}
            for col, prob in zip(LABEL_COLS, probs):
                thr = self._thresholds.get(col, 0.5)
                preds[col] = {"prob": round(prob, 3),
                              "positive": prob >= thr}

            actual = {}
            try:
                study_id = self._path.stem.replace("cxr_", "")
                data     = lookup_patient(study_id)
                actual   = data.get("labels", {}) if data else {}
            except Exception:
                pass

            self.after(0, lambda p=dict(preds),
                              a=dict(actual): self._fill(p, a))
            self.after(0, lambda: self._sv(
                self.sv_status, f"Analysis complete  ·  {self._path.name}"))
        except Exception as e:
            msg = str(e)
            self.after(0, lambda m=msg: self._sv(
                self.sv_status, f"Error: {m}"))
        finally:
            self.after(0, self._restore_btn)

    # ── fill results ──────────────────────────────────────────────────────────
    def _fill(self, preds, actual):
        if not self.winfo_exists():
            return

        detected = [c for c in DISPLAY_COLS
                    if preds.get(c, {}).get("positive")]

        if not detected:
            self._update_summary(
                "NORMAL  —  No cardiac conditions detected",
                "All displayed conditions below threshold",
                "#166534", TP_BG)
        else:
            n = len(detected)
            self._update_summary(
                f"{n} CONDITION{'S' if n > 1 else ''} DETECTED",
                "  ·  ".join(detected),
                "#991B1B", FP_BG)

        for col, refs in self.row_refs.items():
            try:
                rbg     = refs["rbg"]
                act_val = actual.get(col, "unknown")
                ai_info = preds.get(col, {})
                ai_pos  = ai_info.get("positive", False)
                ai_prob = ai_info.get("prob", 0.0)

                # clinical record badge
                if act_val == "present":
                    act_txt  = "CONFIRMED"
                    act_bg   = B_CONF_BG
                    act_fg   = B_CONF_FG
                    act_relief = "flat"
                elif act_val == "absent":
                    act_txt  = "NEGATIVE"
                    act_bg   = B_NEG_BG
                    act_fg   = B_NEG_FG
                    act_relief = "flat"
                else:
                    act_txt  = "NOT IN REPORT"
                    act_bg   = rbg
                    act_fg   = B_NR_FG
                    act_relief = "flat"

                refs["act_var"].set(act_txt)
                refs["act_lbl"].config(bg=act_bg, fg=act_fg,
                                       relief=act_relief)

                # AI result badge
                if ai_pos:
                    ai_txt = "DETECTED"
                    ai_bg  = B_DET_BG
                    ai_fg  = B_DET_FG
                    bar_col = BAR_DET
                else:
                    ai_txt = "NOT DETECTED"
                    ai_bg  = B_NDET_BG
                    ai_fg  = B_NDET_FG
                    bar_col = BAR_NDET

                refs["ai_lbl_var"].set(ai_txt)
                # always P(present) so label it explicitly — prevents reading
                # "NOT DETECTED 85%" as "85% confident it's absent"
                refs["ai_pct_var"].set(f"  P(+): {int(ai_prob * 100)}%")
                refs["ai_lbl"].config(bg=ai_bg, fg=ai_fg)
                refs["bar"].set(ai_prob, color=bar_col)

                # row background
                if   act_val == "present" and ai_pos:     new_bg = TP_BG
                elif act_val == "present" and not ai_pos: new_bg = FN_BG
                elif act_val == "absent"  and ai_pos:     new_bg = FP_BG
                elif act_val == "unknown" and ai_pos:     new_bg = INFO_BG
                else:                                      new_bg = rbg

                refs["row"].config(bg=new_bg)
                refs["nf"].config(bg=new_bg)
                refs["ai_frame"].config(bg=new_bg)
                refs["top_r"].config(bg=new_bg)
                refs["bar"].config(bg=new_bg)
                refs["ai_pct_var"]  # already set above
                for child in refs["nf"].winfo_children():
                    try:
                        child.config(bg=new_bg)
                    except tk.TclError:
                        pass
                # keep act_lbl on its own badge bg
                refs["act_lbl"].config(bg=act_bg)

            except tk.TclError:
                continue

    def _update_summary(self, main, sub, fg, bg):
        self.sum_frame.config(bg=bg)
        self.lbl_sum.config(bg=bg, fg=fg)
        self.lbl_sum_sub.config(bg=bg, fg=fg)
        self._sv(self.sv_sum, main)
        self._sv(self.sv_sum_sub, f"   {sub}" if sub else "")

    def _reset_rows(self):
        for refs in self.row_refs.values():
            try:
                rbg = refs["rbg"]
                refs["act_var"].set("—")
                refs["ai_lbl_var"].set("—")
                refs["ai_pct_var"].set("")
                refs["act_lbl"].config(bg=rbg, fg=TEXT_MUT, relief="flat")
                refs["ai_lbl"].config(bg=rbg,  fg=TEXT_MUT)
                refs["row"].config(bg=rbg)
                refs["nf"].config(bg=rbg)
                refs["ai_frame"].config(bg=rbg)
                refs["top_r"].config(bg=rbg)
                refs["bar"].set(0)
                refs["bar"].config(bg=rbg)
                for child in refs["nf"].winfo_children():
                    try:
                        child.config(bg=rbg)
                    except tk.TclError:
                        pass
            except tk.TclError:
                continue

    def _restore_btn(self):
        try:
            if self.btn.winfo_exists():
                self.btn.config(state="normal",
                                text="  Open & Analyse  ▶")
        except tk.TclError:
            pass

    @staticmethod
    def _sv(sv, value):
        try:
            sv.set(value)
        except tk.TclError:
            pass


if __name__ == "__main__":
    DiagnosticApp().mainloop()
