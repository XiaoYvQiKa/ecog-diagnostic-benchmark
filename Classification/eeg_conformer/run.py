# -*- coding: utf-8 -*-
"""EEG-Conformer trial-based classification (Song et al., 2023).

Thin entry point over the shared deep-learning pipeline. Trial extraction,
downsampling (1000 -> 500 Hz) and the shared training loop live in
``../common/``. See ``../README.md``.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.run_dl import main  # noqa: E402

if __name__ == '__main__':
    main(default_methods=['eegconformer'])
