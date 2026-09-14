"""Unit tests for scripts/gcs_utils.py using Python unittest with mocks."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.gcs_utils import (
    delete_gcs_blob,
    get_gcs_client,
    guess_mime_type,
    parse_gcs_uri,
    upload_file_to_gcs,
)


class TestGCSUtils(unittest.TestCase):
    def test_guess_mime_type(self):
        self.assertEqual(guess_mime_type("video.mp4"), "video/mp4")
        self.assertEqual(guess_mime_type(Path("video.MOV")), "video/quicktime")
        self.assertEqual(guess_mime_type("audio.wav"), "audio/wav")
        self.assertEqual(guess_mime_type("audio.mp3"), "audio/mpeg")
        self.assertEqual(guess_mime_type("unknown.bin"), "video/mp4")

    def test_parse_gcs_uri_valid(self):
        bucket, blob = parse_gcs_uri("gs://my-bucket/path/to/video.mp4")
        self.assertEqual(bucket, "my-bucket")
        self.assertEqual(blob, "path/to/video.mp4")

        bucket, blob = parse_gcs_uri("gs://my-bucket/video.mp4")
        self.assertEqual(bucket, "my-bucket")
        self.assertEqual(blob, "video.mp4")

    def test_parse_gcs_uri_invalid(self):
        with self.assertRaises(ValueError):
            parse_gcs_uri("http://storage.googleapis.com/bucket/video.mp4")
        with self.assertRaises(ValueError):
            parse_gcs_uri("/local/path/video.mp4")

    @patch("scripts.gcs_utils.storage.Client")
    def test_get_gcs_client(self, mock_client_cls):
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance
        client = get_gcs_client(project_id="test-proj")
        mock_client_cls.assert_called_once_with(project="test-proj")
        self.assertEqual(client, mock_instance)

    def test_upload_file_to_gcs_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            upload_file_to_gcs("/non/existent/file.mp4", "bucket", "dest")

    @patch("scripts.gcs_utils.Path.is_file", return_value=True)
    def test_upload_file_to_gcs_success(self, mock_is_file):
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        gcs_uri = upload_file_to_gcs(
            local_path="/fake/video.mp4",
            bucket_name="my-bucket",
            destination_blob_name="raw/uuid_video.mp4",
            content_type="video/mp4",
            client=mock_client,
        )

        mock_client.bucket.assert_called_once_with("my-bucket")
        mock_bucket.blob.assert_called_once_with("raw/uuid_video.mp4")
        self.assertEqual(mock_blob.content_type, "video/mp4")
        mock_blob.upload_from_filename.assert_called_once()
        self.assertEqual(gcs_uri, "gs://my-bucket/raw/uuid_video.mp4")

    def test_delete_gcs_blob(self):
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        delete_gcs_blob("gs://my-bucket/raw/temp.mp4", client=mock_client)
        mock_client.bucket.assert_called_once_with("my-bucket")
        mock_bucket.blob.assert_called_once_with("raw/temp.mp4")
        mock_blob.delete.assert_called_once()

    def test_delete_gcs_blob_suppresses_exceptions(self):
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.delete.side_effect = Exception("Blob not found")
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        # Should not raise exception
        delete_gcs_blob("gs://my-bucket/raw/missing.mp4", client=mock_client)


if __name__ == "__main__":
    unittest.main()
