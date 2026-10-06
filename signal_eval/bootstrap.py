"""
PURPOSE:  Drive bootstrap helpers: gdown/keras imports and download_notebook_from_drive (formerly used to fetch dataprocess.ipynb from Drive).
TAGS:     download_notebook_from_drive, gdown, dataprocess.ipynb, drive bootstrap, load_preprocessed_data_from_drive
PITFALLS: dataprocess.ipynb no longer exists: the runner loads the data/ package instead; this function is kept only so the public name still exists after %run. Needs gdown. Executed into the one shared signal_eval namespace by signal_eval/_loader.py (never imported on its own): names from other modules resolve at call time.
"""
import gdown
import os,keras
import pandas as pd

def download_notebook_from_drive(
    file_id: str,
    notebook_name: str = 'dataprocess.ipynb',
    download_dir: str = '.',
    quiet: bool = False
) -> str:
    """
    تحميل notebook من Google Drive

    Args:
        file_id: معرّف الملف (يُستخرج من رابط المشاركة)
        notebook_name: اسم الملف المحلي
        download_dir: مجلد الحفظ
        quiet: إخفاء تفاصيل التحميل

    Returns:
        المسار الكامل للملف المحمّل
    """
    # ✅ تنظيف file_id من أي معاملات إضافية
    file_id = file_id.split('?')[0].strip()

    output_path = os.path.join(download_dir, notebook_name)
    url = f'https://drive.google.com/uc?id={file_id}'

    print(f"📥 جاري تحميل {notebook_name}...")

    try:
        gdown.download(url, output_path, quiet=quiet, fuzzy=True)
    except Exception as e:
        print(f"❌ خطأ في التحميل: {e}")
        print(f"💡 تأكد من:")
        print(f"   1. الملف مشارك بإعداد 'Anyone with the link'")
        print(f"   2. file_id صحيح: {file_id}")
        print(f"   3. الرابط الكامل: {url}")
        raise

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"❌ فشل التحميل: {output_path}")

    file_size = os.path.getsize(output_path) / 1024
    print(f"✅ تم التحميل: {output_path} ({file_size:.1f} KB)")

    return output_path
