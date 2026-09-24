"""video-trimmer 自訂例外階層。

各模組內部一律 raise 這裡定義的例外，不呼叫 sys.exit()；
只有 scripts/video_trimmer.py 的 main() 統一 catch 並決定 exit code。
"""


class VideoTrimmerError(Exception):
    """所有 video-trimmer 自訂例外的基底類別。"""


class FFmpegError(VideoTrimmerError):
    """ffmpeg / ffprobe 呼叫失敗。"""


class GeminiAPIError(VideoTrimmerError):
    """Gemini API（上傳、推論、Interactions）呼叫或回應解析失敗。"""


class InvalidInputError(VideoTrimmerError):
    """使用者提供的輸入（檔案路徑、副檔名等）不合法。"""
