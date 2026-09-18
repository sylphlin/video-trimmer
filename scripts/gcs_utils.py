"""scripts/gcs_utils.py - Google Cloud Storage & Google Drive (ADC) Integration.

Provides:
- GCS client initialization via Application Default Credentials (ADC)
- SHA-256 / Google Drive MD5 smart caching for GCS uploads (`raw/` 2-day lifecycle)
- Google Drive API v3 file/folder link resolution and streaming download via ADC (`drive.readonly`)
- Zero-transfer GCS fast path when `gdrive_md5` already matches on the remote GCS blob
"""

import hashlib
import logging
import os
import re
from pathlib import Path
from urllib.parse import urlparse

from google.cloud import storage

logger = logging.getLogger(__name__)

GDRIVE_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/drive.readonly",
]

# Extension to MIME type mapping for audio & video
_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
    ".flv": "video/x-flv",
    ".wmv": "video/x-ms-wmv",
    ".m4v": "video/mp4",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
}


def guess_mime_type(path: Path | str) -> str:
    """Best-effort MIME type inference for media files, defaulting to video/mp4."""
    suffix = Path(path).suffix.lower()
    return _MIME_TYPES.get(suffix, "video/mp4")


def parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """
    Parse a Google Cloud Storage URI (gs://bucket-name/path/to/blob).
    Returns (bucket_name, blob_name).
    """
    parsed = urlparse(gcs_uri)
    if parsed.scheme != "gs":
        raise ValueError(f"無效的 GCS URI（必須以 gs:// 開頭）: {gcs_uri}")
    bucket_name = parsed.netloc
    blob_name = parsed.path.lstrip("/")
    return bucket_name, blob_name


def get_gcs_client(project_id: str = None) -> storage.Client:
    """Return an authenticated Google Cloud Storage client via Application Default Credentials (ADC)."""
    return storage.Client(project=project_id)


def compute_file_sha256(filepath: Path | str) -> str:
    """Compute SHA-256 hex digest of a local file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_file_md5(filepath: Path | str) -> str:
    """Compute MD5 hex digest of a local file (matches Google Drive md5Checksum)."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_gdrive_source(val: str | Path | None) -> bool:
    """Return True if val is a Google Drive URL or gdrive:// URI."""
    if not val:
        return False
    s = str(val).strip()
    return (
        "drive.google.com" in s
        or "docs.google.com" in s
        or s.startswith("gdrive://")
    )


def parse_gdrive_url(url_or_id: str) -> dict:
    """
    Parse a Google Drive file/folder URL or ID into {'id': <id>, 'type': 'file'|'folder'|'unknown'}.
    """
    s = str(url_or_id).strip()
    if s.startswith("gdrive://"):
        rest = s[len("gdrive://"):].strip("/")
        if rest.startswith("folder/"):
            return {"id": rest.split("/", 1)[1].split("?")[0], "type": "folder"}
        if rest.startswith("file/"):
            return {"id": rest.split("/", 1)[1].split("?")[0], "type": "file"}
        return {"id": rest.split("?")[0], "type": "unknown"}

    m_folder = re.search(r"/folders/([a-zA-Z0-9_-]{10,})", s)
    if m_folder:
        return {"id": m_folder.group(1), "type": "folder"}

    m_file = re.search(r"/file/d/([a-zA-Z0-9_-]{10,})", s)
    if m_file:
        return {"id": m_file.group(1), "type": "file"}

    m_id = re.search(r"[?&]id=([a-zA-Z0-9_-]{10,})", s)
    if m_id:
        return {"id": m_id.group(1), "type": "unknown"}

    if re.match(r"^[a-zA-Z0-9_-]{15,}$", s) and not Path(s).exists():
        return {"id": s, "type": "unknown"}

    raise ValueError(f"無法解析 Google Drive 連結或 ID: {url_or_id}")


def get_gdrive_session(project_id: str = None):
    """Return an AuthorizedSession authenticated with ADC and drive.readonly scope."""
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    creds, default_proj = google.auth.default(scopes=GDRIVE_SCOPES)
    quota_proj = (
        project_id
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCP_PROJECT")
        or default_proj
    )
    if quota_proj and hasattr(creds, "with_quota_project"):
        creds = creds.with_quota_project(quota_proj)
    session = AuthorizedSession(creds)
    if quota_proj:
        session.headers["X-Goog-User-Project"] = quota_proj
    return session


def get_gdrive_file_metadata(url_or_id: str, project_id: str = None, session=None) -> dict:
    """Query Google Drive API v3 for file metadata (id, name, mimeType, size, md5Checksum)."""
    parsed = parse_gdrive_url(url_or_id)
    file_id = parsed["id"]
    sess = session or get_gdrive_session(project_id=project_id)
    api_url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
    params = {
        "fields": "id,name,mimeType,size,md5Checksum",
        "supportsAllDrives": "true",
    }
    resp = sess.get(api_url, params=params, timeout=30)
    if resp.status_code in (401, 403):
        raise RuntimeError(
            f"Google Drive API 權限錯誤 ({resp.status_code}): {resp.text}\n"
            f"請執行以下指令授權 ADC 讀取 Google Drive:\n"
            f"  gcloud auth application-default login --scopes=\"https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/drive.readonly\""
        )
    resp.raise_for_status()
    return resp.json()


def download_gdrive_file_with_cache(
    url_or_id: str,
    target_dir: Path | str = None,
    project_id: str = None,
    force_download: bool = False,
) -> Path:
    """
    Download a Google Drive media file via ADC into `target_dir`, verifying remote MD5
    to skip re-downloading when a matching cached file already exists on disk.
    """
    sess = get_gdrive_session(project_id=project_id)
    meta = get_gdrive_file_metadata(url_or_id, project_id=project_id, session=sess)
    file_id = meta["id"]
    filename = meta.get("name") or f"gdrive_{file_id}.mp4"
    remote_md5 = meta.get("md5Checksum")
    remote_size = int(meta.get("size", 0) or 0)

    dest_dir = Path(target_dir or (Path.cwd() / "gdrive_inputs")).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    local_path = dest_dir / filename

    if not force_download and local_path.is_file():
        if remote_md5 and local_path.stat().st_size == remote_size:
            if compute_file_md5(local_path) == remote_md5:
                logger.info("[GDrive Cache Hit] 本地檔案 MD5 與雲端硬碟一致 (%s)，略過下載。", local_path.name)
                return local_path

    size_mb = remote_size / (1024 * 1024)
    logger.info("[GDrive Download] 正在透過 ADC 從 Google Drive 下載 '%s' (%.1f MB)...", filename, size_mb)
    dl_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&supportsAllDrives=true"
    tmp_path = local_path.with_suffix(local_path.suffix + ".part")
    with sess.get(dl_url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp_path.replace(local_path)
    logger.info("[✓] Google Drive 下載完成: %s", local_path)
    return local_path


def upload_file_to_gcs(
    local_path: Path | str,
    bucket_name: str,
    destination_blob_name: str,
    content_type: str = None,
    client: storage.Client = None,
    extra_metadata: dict = None,
    force_upload: bool = False,
) -> str:
    """
    Upload a local file to a GCS bucket with SHA-256 / gdrive_md5 metadata caching.
    Returns the gs:// URI of the uploaded blob.
    """
    local_p = Path(local_path).resolve()
    if not local_p.is_file():
        raise FileNotFoundError(f"找不到本地檔案: {local_p}")

    gcs_client = client or get_gcs_client()
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    try:
        file_sha256 = compute_file_sha256(local_p)
        file_md5 = compute_file_md5(local_p)
        file_size = local_p.stat().st_size
    except OSError:
        file_sha256 = None
        file_md5 = None
        file_size = None

    if not force_upload and file_sha256 and file_size is not None:
        try:
            if blob.exists():
                blob.reload()
                remote_meta = blob.metadata or {}
                if (
                    blob.size == file_size
                    and (
                        remote_meta.get("sha256") == file_sha256
                        or remote_meta.get("gdrive_md5") == file_md5
                    )
                ):
                    gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
                    logger.info("[GCS Cache Hit] 遠端物件雜湊相符 (%s)，秒級略過重傳！", gcs_uri)
                    return gcs_uri
        except Exception:
            pass

    if content_type:
        blob.content_type = content_type

    if file_sha256 or extra_metadata:
        meta = {}
        if file_sha256:
            meta["sha256"] = file_sha256
        if file_md5:
            meta["gdrive_md5"] = file_md5
        if extra_metadata:
            meta.update(extra_metadata)
        blob.metadata = meta

    logger.info("[*] 上傳 %s 至 gs://%s/%s...", local_p.name, bucket_name, destination_blob_name)
    blob.upload_from_filename(str(local_p))
    gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
    logger.info("[✓] 上傳完成: %s", gcs_uri)
    return gcs_uri


def delete_gcs_blob(gcs_uri: str, client: storage.Client = None) -> None:
    """
    Delete a blob by its gs:// URI. Used to clean up ephemeral raw/ uploads
    when requested. Missing blobs are treated as already cleaned up.
    """
    try:
        bucket_name, blob_name = parse_gcs_uri(gcs_uri)
        gcs_client = client or get_gcs_client()
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.delete()
        logger.info("[✓] 清理暫存 GCS 物件: %s", gcs_uri)
    except Exception as e:
        logger.debug("清理 GCS 物件時略過例外 (%s): %s", gcs_uri, e)
