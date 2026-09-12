#!/usr/bin/env python3
"""
auto_rough_cut.py - Backward compatibility wrapper for video-trimmer.
Transparently forwards all arguments to scripts.video_trimmer.main().
"""

import sys
from pathlib import Path

root_dir = Path(__file__).parent.resolve()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from scripts.video_trimmer import main

if __name__ == "__main__":
    main()
