"""Generate keep-bad-channel data for the electrode-exclusion control experiment.

Identical to preprocess_raw.py except that it skips bad-channel exclusion, so
the effect of dropping the three broken electrodes can be quantified under an
otherwise unchanged pipeline.

Output: data_dtcnet_keepbad/  (spectrograms keep all channels; finger data copied directly)
"""
import numpy as np
import os, sys, shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_raw import load_raw, bandpass_filter, notch_filter, morlet_spec, per_channel_normalize

DATA_DIR = os.environ.get(
    "BCI_RAW_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_raw"),
)
SRC_DIR  = os.environ.get(
    "BCI_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_dtcnet"),
)  # finger data source
OUT_DIR  = os.environ.get(
    "BCI_KEEPBAD_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_dtcnet_keepbad"),
)  # keep bad channels


def process_keepbad(sid):
    print(f'Sub{sid}: loading (keep bad channels)...')
    tr_e, te_e = load_raw(sid, DATA_DIR)

    # key difference: do NOT exclude bad channels (sub1 ch55, sub2 ch21+38, sub3 ch50 all kept)
    # paper order: per-channel normalization → bandpass → notch → Morlet → downsample
    tr_e, te_e = per_channel_normalize(tr_e, te_e)
    print(f'  per-channel normalization done ({tr_e.shape[0]}ch, keep bad channels)')

    for name, eco in [('train', tr_e), ('test', te_e)]:
        eco = bandpass_filter(eco)
        eco = notch_filter(eco)
        eco = morlet_spec(eco)
        eco_ds = eco[:, :, ::10]
        np.save(os.path.join(OUT_DIR, f'sub{sid}_{name}_spec.npy'), eco_ds)
    # finger data unaffected by electrodes, copy directly
    for tag in ['train_finger', 'test_finger']:
        shutil.copy(f'{SRC_DIR}/sub{sid}_{tag}.npy', f'{OUT_DIR}/sub{sid}_{tag}.npy')
    print(f'Sub{sid} done ({tr_e.shape[0]}ch).')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for s in [1, 2, 3]:
        process_keepbad(s)
    print(f'\nDone. Keep-bad-channel data in {OUT_DIR}')
    # verify channel count (should equal original 62/48/64)
    for s, expect in [(1, 62), (2, 48), (3, 64)]:
        spec = np.load(f'{OUT_DIR}/sub{s}_train_spec.npy', mmap_mode='r')
        finger = np.load(f'{OUT_DIR}/sub{s}_train_finger.npy', mmap_mode='r')
        assert spec.shape[0] == expect, f'sub{s} channel {spec.shape[0]} != expected {expect}'
        assert spec.shape[2] == finger.shape[1] == 40000
    print('Verified: channels 62/48/64 (incl. bad), time aligned 40000 @100Hz')


if __name__ == '__main__':
    main()
