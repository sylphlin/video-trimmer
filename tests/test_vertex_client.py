"""Unit tests for scripts/gemini_client.py using Python unittest with mocks."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.exceptions import GeminiAPIError
from scripts.gemini_client import (
    _extract_text_from_response,
    get_gemini_client,
    load_env_file,
    run_gemini_inference,
    stage_video_to_gcs,
)


class TestVertexClient(unittest.TestCase):
    def setUp(self):
        self._orig_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._orig_env)

    def test_load_env_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            env_file = tmp_path / ".env"
            env_file.write_text("GOOGLE_CLOUD_PROJECT=test-env-proj\nVIDEO_TRIMMER_BUCKET=test-bucket\n# Comment\n")

            with patch("scripts.gemini_client.Path.cwd", return_value=tmp_path):
                # Ensure vars are not set
                os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
                os.environ.pop("VIDEO_TRIMMER_BUCKET", None)

                load_env_file()
                self.assertEqual(os.environ.get("GOOGLE_CLOUD_PROJECT"), "test-env-proj")
                self.assertEqual(os.environ.get("VIDEO_TRIMMER_BUCKET"), "test-bucket")

    @patch("google.genai.Client")
    def test_get_gemini_client_explicit_args(self, mock_genai_client):
        mock_instance = MagicMock()
        mock_genai_client.return_value = mock_instance

        client = get_gemini_client(project_id="my-proj", location="us-central1")
        mock_genai_client.assert_called_once_with(vertexai=True, project="my-proj", location="us-central1")
        self.assertEqual(client, mock_instance)

    @patch("scripts.gemini_client.load_env_file")
    @patch("google.genai.Client")
    def test_get_gemini_client_env_resolution(self, mock_genai_client, mock_load_env):
        os.environ["GOOGLE_CLOUD_PROJECT"] = "env-project"
        os.environ["GOOGLE_CLOUD_LOCATION"] = "asia-east1"

        get_gemini_client()
        mock_genai_client.assert_called_once_with(vertexai=True, project="env-project", location="asia-east1")

    @patch("scripts.gemini_client.load_env_file")
    @patch("google.genai.Client")
    @patch("google.auth.default", return_value=(MagicMock(), "adc-default-project"))
    def test_get_gemini_client_adc_fallback(self, mock_auth_default, mock_genai_client, mock_load_env):
        os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
        os.environ.pop("GCP_PROJECT", None)
        os.environ.pop("GOOGLE_CLOUD_LOCATION", None)
        os.environ.pop("GCP_REGION", None)

        get_gemini_client()
        mock_genai_client.assert_called_once_with(vertexai=True, project="adc-default-project", location="global")

    @patch("scripts.gemini_client.load_env_file")
    @patch("google.auth.default", side_effect=Exception("No ADC"))
    def test_get_gemini_client_missing_project_raises(self, mock_auth_default, mock_load_env):
        os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
        os.environ.pop("GCP_PROJECT", None)

        with self.assertRaises(GeminiAPIError):
            get_gemini_client()

    def test_stage_video_to_gcs_direct_gs_uri(self):
        uri, mime, is_ephemeral = stage_video_to_gcs("gs://bucket/video.mp4", bucket_name=None)
        self.assertEqual(uri, "gs://bucket/video.mp4")
        self.assertEqual(mime, "video/mp4")
        self.assertFalse(is_ephemeral)

    def test_stage_video_to_gcs_local_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            stage_video_to_gcs("/non/existent/path/video.mp4", bucket_name="my-bucket")

    @patch("scripts.gemini_client.Path.is_file", return_value=True)
    def test_stage_video_to_gcs_missing_bucket(self, mock_is_file):
        with self.assertRaises(GeminiAPIError):
            stage_video_to_gcs("/fake/video.mp4", bucket_name=None)

    @patch("scripts.gemini_client.upload_file_to_gcs", return_value="gs://my-bucket/raw/xyz_video.mp4")
    def test_stage_video_to_gcs_success(self, mock_upload):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_vid:
            uri, mime, is_ephemeral = stage_video_to_gcs(
                video_source=tmp_vid.name,
                bucket_name="my-bucket",
            )
            self.assertEqual(uri, "gs://my-bucket/raw/xyz_video.mp4")
            self.assertEqual(mime, "video/mp4")
            self.assertTrue(is_ephemeral)
            mock_upload.assert_called_once()

    def test_run_gemini_inference_static(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"final_edl": []}'
        mock_response.candidates = [MagicMock(finish_reason="STOP")]
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.thoughts_token_count = 10
        mock_response.usage_metadata.total_token_count = 160
        mock_client.models.generate_content.return_value = mock_response

        mock_types = MagicMock()

        raw_json, usage, duration = run_gemini_inference(
            client=mock_client,
            types_module=mock_types,
            model="gemini-3.8-flash",
            gcs_uri="gs://bucket/video.mp4",
            mime_type="video/mp4",
            prompt="test prompt",
            agentic=False,
        )

        self.assertEqual(raw_json, '{"final_edl": []}')
        self.assertEqual(usage["total_tokens"], 160)
        self.assertGreaterEqual(duration, 0.0)
        mock_types.ThinkingConfig.assert_called_once_with(thinking_budget=3800)
        mock_types.AutomaticFunctionCallingConfig.assert_called_once_with(disable=True)
        mock_types.GenerateContentConfig.assert_called_once_with(
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=65536,
            thinking_config=mock_types.ThinkingConfig.return_value,
            automatic_function_calling=mock_types.AutomaticFunctionCallingConfig.return_value,
        )
        mock_client.models.generate_content.assert_called_once()

    def test_run_gemini_inference_agentic(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"final_edl": []}'
        mock_response.candidates = [MagicMock(finish_reason="STOP")]
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response

        mock_types = MagicMock()

        raw_json, usage, duration = run_gemini_inference(
            client=mock_client,
            types_module=mock_types,
            model="gemini-3.8-flash",
            gcs_uri="gs://bucket/video.mp4",
            mime_type="video/mp4",
            prompt="test prompt",
            agentic=True,
            num_sentences=62,
            video_duration=504.0,
            start_offset="854s",
            end_offset="1382s",
        )

        self.assertEqual(raw_json, '{"final_edl": []}')
        self.assertEqual(usage["total_tokens"], 0)
        mock_types.ThinkingConfig.assert_called_once_with(thinking_budget=3910)
        mock_types.VideoMetadata.assert_called_once_with(start_offset="854s", end_offset="1382s")
        mock_types.GenerateContentConfig.assert_called_once_with(
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=65536,
            thinking_config=mock_types.ThinkingConfig.return_value,
            automatic_function_calling=mock_types.AutomaticFunctionCallingConfig.return_value,
            media_resolution=mock_types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
        )
        mock_client.models.generate_content.assert_called_once()

    def test_run_gemini_inference_prefill_deadline_fallback(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"final_edl": [{"clip_id": 1}]}'
        mock_response.candidates = [MagicMock(finish_reason="STOP")]
        mock_response.usage_metadata = None

        # Simulate PREFILL_REQUEST_DEADLINE_EXCEEDED on Stage 1 (MEDIUM) and Stage 2 (LOW), succeeding on Stage 3 (Static LOW)
        mock_client.models.generate_content.side_effect = [
            RuntimeError("status = DEADLINE_EXCEEDED: Request deadline exceeded before prefill finished. error_code: PREFILL_REQUEST_DEADLINE_EXCEEDED"),
            RuntimeError("status = DEADLINE_EXCEEDED: Request deadline exceeded before prefill finished. error_code: PREFILL_REQUEST_DEADLINE_EXCEEDED"),
            mock_response,
        ]

        mock_types = MagicMock()

        raw_json, usage, duration = run_gemini_inference(
            client=mock_client,
            types_module=mock_types,
            model="gemini-3.8-flash",
            gcs_uri="gs://bucket/video.mp4",
            mime_type="video/mp4",
            prompt="test prompt",
            agentic=True,
        )

        self.assertEqual(raw_json, '{"final_edl": [{"clip_id": 1}]}')
        # Exactly 3 calls (1 per stage, no blind 4x retry on prefill deadline error)
        self.assertEqual(mock_client.models.generate_content.call_count, 3)

    def test_run_gemini_inference_max_tokens_raises(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"final_edl": [{"clip_id": 1'
        mock_response.candidates = [MagicMock(finish_reason="FinishReason.MAX_TOKENS")]
        mock_response.usage_metadata.prompt_token_count = 227587
        mock_response.usage_metadata.thoughts_token_count = 7246
        mock_response.usage_metadata.candidates_token_count = 946
        mock_response.usage_metadata.total_token_count = 235779
        mock_client.models.generate_content.return_value = mock_response

        mock_types = MagicMock()

        with self.assertRaises(GeminiAPIError) as ctx:
            run_gemini_inference(
                client=mock_client,
                types_module=mock_types,
                model="gemini-3.8-flash",
                gcs_uri="gs://bucket/video.mp4",
                mime_type="video/mp4",
                prompt="test prompt",
                agentic=True,
            )
        self.assertIn("max_output_tokens", str(ctx.exception))
        self.assertIn("MAX_TOKENS", str(ctx.exception))

    def test_extract_text_with_thinking_parts(self):
        """Verify that thought parts are filtered and visible text is returned."""
        mock_response = MagicMock()

        thought_part = MagicMock()
        thought_part.text = "This is internal thinking."
        thought_part.thought = True

        answer_part = MagicMock()
        answer_part.text = '{"final_edl": [{"clip_id": 1}]}'
        answer_part.thought = False

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [thought_part, answer_part]
        mock_response.candidates = [mock_candidate]

        extracted = _extract_text_from_response(mock_response)
        self.assertEqual(extracted, '{"final_edl": [{"clip_id": 1}]}')

    def test_extract_text_fallback_to_response_text(self):
        """Verify fallback to response.text when no candidates exist."""
        mock_response = MagicMock(spec=["text"])
        mock_response.text = '{"final_edl": []}'

        extracted = _extract_text_from_response(mock_response)
        self.assertEqual(extracted, '{"final_edl": []}')


if __name__ == "__main__":
    unittest.main()


