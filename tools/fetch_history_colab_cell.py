"""
PURPOSE:  Single paste-and-run Colab cell: mount Drive, fetch fetch_history_vision_colab.py (GitHub or Drive) and run
          it with the settings at the top.
TAGS:     colab cell, download history, binance vision, drive mount, SOURCE github/drive, settings INTERVAL START
          FUNDING OPEN_INTEREST
PITFALLS: Private repo: the GitHub source needs a token from Colab secrets; the hard-coded GITHUB_RAW URL points at a
          specific branch and goes stale.
"""
# ═══════════════════════════════════════════════════════════════════════════════
# خلية Colab واحدة: تركيب Google Drive ← جلب سكربت التحميل (من GitHub أو من Drive) ← تشغيله.
# الصقها كما هي في خلية Colab وشغّلها. المخرجات تُكتب في Drive مباشرة:
#   MyDrive/history_15m/<SYMBOL>.csv.gz   (ملف واحد متصل مضغوط لكل عملة)
#   MyDrive/funding_rate/ , open_interest/ , futures_metrics/ , premium_index_15m/ , crypto_data/
# إعادة تشغيل الخلية نفسها تُكمل من آخر شمعة محفوظة لكل عملة (لا يُعاد تحميل شيء).
# ═══════════════════════════════════════════════════════════════════════════════
import os
import shutil
import subprocess
import sys
import urllib.request

# ── الإعدادات ──
SOURCE = "github"          # "github" (يلزم توكن: المستودع خاص) | "drive"
GITHUB_RAW = ("https://raw.githubusercontent.com/yuosef772424/crypto-signal-prediction/"
              "claude/charming-sagan-kswo2r/tools/fetch_history_vision_colab.py")
DRIVE_SCRIPT = "/content/drive/MyDrive/tools/fetch_history_vision_colab.py"   # مع SOURCE="drive"
DRIVE_ROOT = "/content/drive/MyDrive"
INTERVAL = "15m"
START = "2020-01-01"       # أقدم تاريخ مطلوب (الأرشيف يبدأ 2020-01 لعقود USDT الدائمة)
SYMBOLS = ""               # "" = كل العملات (تشمل المشطوبة) | "BTCUSDT,ETHUSDT" للتجربة
FUNDING = True             # معدّل التمويل (شهري من الأرشيف)
OPEN_INTEREST = True       # الفائدة المفتوحة + نسب long/short + taker (من الأرشيف، ساعياً)
OI_START = "2024-01-01"    # أبكر = أطول (ملف يومي لكل عملة؛ الأرشيف يبدأ 2020-09)
OI_PERIOD = "1h"           # "15m" بدقة الشموع (حجم أكبر ~4 مرات)
PREMIUM = False            # مؤشر العلاوة بنفس الفريم (~45% من حجم الشموع إضافياً)
WORKERS = 8                # عملات تُعالَج بالتوازي (~50MB ذاكرة لكل عملة)
DOWNLOADS = 32             # ملفات ZIP تُحمَّل بالتوازي

# ── 1) Google Drive ──
from google.colab import drive  # noqa: E402
drive.mount("/content/drive")

# ── 2) السكربت ──
SCRIPT = "/content/fetch_history_vision_colab.py"
if SOURCE == "github":
    # المستودع خاص: توكن GitHub من أسرار Colab (🔑 Secrets ← GITHUB_TOKEN، مع Notebook access)
    try:
        from google.colab import userdata
        _tok = userdata.get("GITHUB_TOKEN") or ""
    except Exception:
        _tok = ""
    if not _tok:
        from getpass import getpass
        _tok = getpass("GitHub token (المستودع خاص): ")
    _req = urllib.request.Request(GITHUB_RAW, headers={"Authorization": f"token {_tok.strip()}"})
    with urllib.request.urlopen(_req) as _r, open(SCRIPT, "wb") as _f:
        _f.write(_r.read())
else:
    shutil.copy(DRIVE_SCRIPT, SCRIPT)
print(f"✓ السكربت ({SOURCE}): {os.path.getsize(SCRIPT):,} بايت")

# ── 3) التشغيل (المخرجات تظهر سطراً بسطر) ──
cmd = [sys.executable, "-u", SCRIPT, "--drive-root", DRIVE_ROOT, "--interval", INTERVAL, "--start", START,
       "--workers", str(WORKERS), "--downloads", str(DOWNLOADS), "--oi-start", OI_START,
       "--oi-period", OI_PERIOD, "--vision-only"]
if SYMBOLS:
    cmd += ["--symbols", SYMBOLS]
if FUNDING:
    cmd.append("--funding")
if OPEN_INTEREST:
    cmd.append("--open-interest")
if PREMIUM:
    cmd.append("--premium")
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="")
print("\nرمز الخروج:", proc.wait())
