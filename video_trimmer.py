#!/usr/bin/env python3
"""
video_trimmer.py - Transparent root entrypoint for Video Trimmer Agent Skill.
Compliant with Agent Skills Specification (https://agentskills.io/specification).
"""

import sys
from pathlib import Path

# Ensure package root is in sys.path
root_dir = Path(__file__).parent.resolve()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from scripts.acoustic import refine_speech_bounds_locked
from scripts.exporters import generate_fcp7_xml, generate_fcpxml
from scripts.render import render_cut_video
from scripts.transcribe import transcribe_video_whisper
from scripts.video_trimmer import main

if __name__ == "__main__":
    main()
