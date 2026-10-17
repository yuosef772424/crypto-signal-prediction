"""
PURPOSE:  Google Drive mount (the only way historical data is read): is_colab() and mount_drive() with a per-session cache.
TAGS:     google drive, mount_drive, is_colab, colab, MyDrive, drive root
PITFALLS: mount_drive returns None outside Colab instead of raising; callers decide. Tests replace mount_drive in the namespace dict (ns['mount_drive'] = ...). Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.

## 3) Google Drive — التركيب (الطريقة الوحيدة للتحميل)

كل تحميل بيانات في هذا الدفتر يمرّ عبر `mount_drive()`: تركيب Drive عبر
`google.colab.drive`، ثم القراءة كملفات محلية من `MyDrive/...`. لا استخدام
لأي رابط عام أو `file_id` قابل للمشاركة، ولا لمكتبة `gdown`.
"""
def is_colab() -> bool:
    """True إن كانت البيئة الحالية Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


#: جذر Drive المُركَّب — يُملأ مرة واحدة لكل جلسة (كاش بسيط).
_DRIVE_ROOT: Dict[str, Optional[Path]] = {"root": None}


def mount_drive(mount_point: Optional[str] = None,
                config: Optional[dict] = None) -> Optional[Path]:
    """يُركِّب Google Drive (مرة واحدة فقط لكل جلسة) ويُرجع مسار ``MyDrive``.

    يُرجع ``None`` خارج Colab بدل رفع استثناء مباشرة، فتبقى الدالة المستدعِية
    هي من تقرّر: بعضها يرفع خطأ واضحاً وبعضها يجرّب بديلاً.
    """
    config = CONFIG if config is None else config
    if not is_colab():
        return None
    if _DRIVE_ROOT["root"] is not None:
        return _DRIVE_ROOT["root"]

    from google.colab import drive

    mount_point = mount_point or config.get("drive_mount_point", "/content/drive")
    drive.mount(mount_point)
    root = Path(mount_point) / "MyDrive"
    _DRIVE_ROOT["root"] = root
    return root
