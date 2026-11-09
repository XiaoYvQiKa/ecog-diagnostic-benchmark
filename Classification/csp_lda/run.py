# -*- coding: utf-8 -*-
"""CSP+LDA classification baseline (Blankertz et al., 2008).

Thin entry point over the shared traditional pipeline. The full protocol
(subject-specific 400 s / 200 s split, electrode QC, per-channel z-score,
label generation) lives in ``../common/``. See ``../README.md``.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.run_traditional import main  # noqa: E402

if __name__ == '__main__':
    main(default_methods=['csp'])
