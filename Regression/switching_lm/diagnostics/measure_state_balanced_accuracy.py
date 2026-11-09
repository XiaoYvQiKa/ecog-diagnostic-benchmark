"""
Measure the state classifier's TEST-SET balanced accuracy under the FINAL
fixed configuration (K=15, lambda_s=0, same as final_run.py) -- the number
citable in the Discussion section, replacing the validation-set number
from the tune_classifier.py grid search (which used per-subject-optimal,
not final-fixed, hyperparameters).

Run in the same folder as the other files, with data/ populated:

    python measure_state_balanced_accuracy.py
"""
import numpy as np

import config as C
from data_io import load_subject_clean
from state_labels import make_state_labels
from evaluate_decoder import fit_decoder, evaluate

TUNED = {
    1: dict(tau_samples=75, M=30),
    2: dict(tau_samples=150, M=30),
    3: dict(tau_samples=500, M=30),
}
N_SELECT_CHANNELS = 15
LAMBDA_S = 0.0
LAMBDA_K = 1.0


def balanced_accuracy(true_state, pred_state, n_states):
    recalls = []
    for k in range(1, n_states + 1):
        mask = true_state == k
        if mask.sum() == 0:
            continue
        recalls.append(np.mean(pred_state[mask] == true_state[mask]))
    return float(np.mean(recalls)), recalls


all_bal = []
for n in [1, 2, 3]:
    print(f"{'='*60}")
    print(f"SUBJECT {n}")
    print(f"{'='*60}")

    sd = load_subject_clean(n)
    sl = make_state_labels(sd.train_glove, sd.test_glove, subject=n)
    tau_samples, M = TUNED[n]["tau_samples"], TUNED[n]["M"]

    bundle = fit_decoder(
        sd.train_ecog, sd.train_glove, sl.train_state,
        n_select_channels=N_SELECT_CHANNELS, lambda_s=LAMBDA_S,
        tau_samples=tau_samples, lambda_k=LAMBDA_K, M=M,
    )
    res = evaluate(bundle, sd.test_ecog, sd.test_glove, sl.test_state)

    pred_state = res["pred_state_aligned"]
    true_state = res["state_true_aligned"]

    raw_acc = np.mean(pred_state == true_state)
    bal_acc, per_class = balanced_accuracy(true_state, pred_state, C.N_STATES)
    all_bal.append(bal_acc)

    labels = C.FINGER_NAMES + ["rest"]
    print(f"  raw accuracy:      {raw_acc:.3f}")
    print(f"  balanced accuracy: {bal_acc:.3f}  (random chance = {1/C.N_STATES:.3f})")
    print("  per-class recall:")
    for label, r in zip(labels, per_class):
        print(f"    {label:6s}: {r:.3f}")
    print()

print(f"{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"balanced accuracy per subject: {[round(b,3) for b in all_bal]}")
print(f"mean +/- SD: {np.mean(all_bal):.3f} +/- {np.std(all_bal, ddof=1):.3f}")
print(f"range: {min(all_bal):.3f}-{max(all_bal):.3f}")
print()
print("This is the number citable in Discussion (test-set, final fixed")
print("K=15/lambda_s=0 config) -- NOT the same as tune_classifier.py's")
print("validation-set, per-subject-optimal-hyperparameter numbers.")
