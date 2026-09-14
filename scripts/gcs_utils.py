"""scripts/gcs_utils.py - Google Cloud Storage Upload & Ephemeral Blob Management.

Provides GCS client initialization via Application Default Credentials (ADC),
MIME type inference, upload utility for staging local media for Vertex AI,
and safe deletion for ephemeral upload cleanup.
"""

import logging
from pathlib import Path
from urllib.parse import urlparse

from google.cloud import storage

logger = logging.getLogger(__name__)

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


def upload_file_to_gcs(
    local_path: Path | str,
    bucket_name: str,
    destination_blob_name: str,
    content_type: str = None,
    client: storage.Client = None,
) -> str:
    """
    Upload a local file to a GCS bucket.
    Returns the gs:// URI of the uploaded blob.
    """
    local_p = Path(local_path).resolve()
    if not local_p.is_file():
        raise FileNotFoundError(f"找不到本地檔案: {local_p}")

    gcs_client = client or get_gcs_client()
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    if content_type:
        blob.content_type = content_type

    logger.info("[*] 上傳 %s 至 gs://%s/%s...", local_p.name, bucket_name, destination_blob_name)
    blob.upload_from_filename(str(local_p))
    gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
    logger.info("[✓] 上傳完成: %s", gcs_uri)
    return gcs_uri


def delete_gcs_blob(gcs_uri: str, client: storage.Client = None) -> None:
    """
    Delete a blob by its gs:// URI. Used to clean up ephemeral raw/ uploads
    right after Gemini has completed inference. Missing blobs are treated as
    already cleaned up.
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
