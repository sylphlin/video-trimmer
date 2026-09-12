#!/usr/bin/env python3
"""
auto_rough_cut.py - Compatibility wrapper for video_trimmer.py
--------------------------------------------------------------
The project and CLI have been renamed to `video-trimmer` (video_trimmer.py).
This file is maintained for backward compatibility with existing workflows.
"""
from video_trimmer import main

if __name__ == "__main__":
    main()
