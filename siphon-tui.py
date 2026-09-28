#!/usr/bin/env python3
import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_root, "siphon-studio"))

from siphon.tui.app import main

if __name__ == "__main__":
    main()
