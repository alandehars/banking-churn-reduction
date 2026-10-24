"""Source loading with reference hash checks; synthetic data must be labelled explicitly."""
import hashlib
import tempfile
import warnings
from pathlib import Path
import gdown
from .inference import read_source_csv,normalize_features
from .schema import DATA_DRIVE_ID,DATA_SOURCE_URL,REFERENCE_SHA256

def load_source_data(data_file, legacy_file=None, strict_reference=True, downloader=None):
    primary = Path(data_file)
    legacy = Path(legacy_file) if legacy_file is not None else None
    temporary = None
    mode = "local_primary"
    chosen = primary
    if not primary.is_file() and legacy is not None and legacy.is_file():
        chosen, mode = legacy, "local_legacy"
    elif not primary.is_file():
        mode = "downloaded"
        primary.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".churn-download-", suffix=".csv", dir=primary.parent, delete=False) as handle:
            temporary = Path(handle.name)
        chosen = temporary
        try:
            result = (downloader or gdown.download)(id=DATA_DRIVE_ID, output=str(temporary), quiet=False)
            if not result or not temporary.is_file() or temporary.stat().st_size == 0:
                raise ValueError("Unduhan tidak menghasilkan file CSV.")
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                "Unduhan sumber gagal. Unggah churn.csv dari DATA_SOURCE_URL ke panel Files Colab, "
                "lalu jalankan ulang. Nama lama Churn_Modelling.csv juga dikenali."
            ) from exc
    try:
        digest = hashlib.sha256(chosen.read_bytes()).hexdigest()
        matches = digest == REFERENCE_SHA256
        if not matches and strict_reference:
            raise ValueError(
                "SHA-256 CSV berbeda dari snapshot sumber yang sudah diperiksa. "
                "Gunakan file dari DATA_SOURCE_URL atau tinjau sumber baru sebelum menonaktifkan STRICT_REFERENCE_HASH."
            )
        if not matches:
            warnings.warn("CSV berbeda dari snapshot sumber. Penggunaan diizinkan secara eksplisit; hasil dapat berubah.")
        frame = read_source_csv(chosen)
        if temporary is not None:
            temporary.replace(primary)  # Hanya file dengan hash/skema yang lolos yang dipasang.
            chosen = primary
        report = {
            "source_url": DATA_SOURCE_URL, "source_drive_id": DATA_DRIVE_ID,
            "source_drive_filename": "churn.csv", "local_file": str(chosen), "read_mode": mode,
            "sha256": digest, "reference_matches": matches,
            "source_rows": len(frame), "source_columns": len(frame.columns),
        }
        return frame, report
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
