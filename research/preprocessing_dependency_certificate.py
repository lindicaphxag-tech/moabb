"""Research-only fixed-LTI preprocessing dependency-cone certificate.

For y = Hx, the train-output dependence on held-out raw samples is
C = P_train H P_test.  ||C dx||_2 <= ||C||_F ||dx||_2.
Raw extraction-window nonoverlap alone cannot certify C = 0 when a
continuous-recording zero-phase/IIR filter has nonlocal impulse response.

Not a MOABB production patch, a general nonlinear-preprocessing theorem,
or evidence of a clinical/BCl accuracy improvement.
"""
import json
import numpy as np


def fir_kernel(displacement, radius=10):
    d = np.abs(displacement)
    return np.maximum(radius + 1 - d, 0) / float((radius + 1) ** 2)


def symmetric_iir_kernel(displacement, alpha=0.97):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    return ((1 - alpha) / (1 + alpha)) * alpha ** np.abs(displacement)


def cross_dependency(a, t, kernel):
    return kernel(a[:, None] - t[None, :])


def certificate(a, t, kernel):
    cross = cross_dependency(a, t, kernel)
    reverse = cross_dependency(t, a, kernel)
    return {
        "raw_index_overlap": int(len(np.intersect1d(a, t))),
        "train_from_test_frobenius": float(np.linalg.norm(cross, "fro")),
        "test_from_train_frobenius": float(np.linalg.norm(reverse, "fro")),
        "strict_zero_cross_dependency": bool(not cross.any() and not reverse.any()),
    }


def main():
    train = np.arange(90, 130, dtype=np.int64)
    test = np.arange(210, 250, dtype=np.int64)
    fir = certificate(train, test, fir_kernel)
    iir = certificate(train, test, symmetric_iir_kernel)
    assert fir["raw_index_overlap"] == iir["raw_index_overlap"] == 0
    assert fir["strict_zero_cross_dependency"]
    assert not iir["strict_zero_cross_dependency"]
    delta = np.zeros(len(test))
    delta[0] = 1.0
    propagated = cross_dependency(train, test, symmetric_iir_kernel) @ delta
    np.testing.assert_allclose(
        propagated, symmetric_iir_kernel(train - test[0]), rtol=0, atol=0
    )
    assert np.any(propagated != 0)
    bound = iir["train_from_test_frobenius"] * np.linalg.norm(delta)
    assert np.linalg.norm(propagated) <= bound + 1e-15

    controls = []
    for gap in (0, 5, 20, 50, 100):
        a = np.arange(100, 140)
        t = np.arange(140 + gap, 180 + gap)
        controls.append({"gap": gap, **certificate(a, t, symmetric_iir_kernel)})
    influence = [x["train_from_test_frobenius"] for x in controls]
    assert all(x > y > 0 for x, y in zip(influence[:-1], influence[1:]))
    print(json.dumps({
        "experiment": "fixed-LTI dependency-cone audit",
        "FIR": fir,
        "IIR": iir,
        "observed_test_impulse_train_output_l2": float(np.linalg.norm(propagated)),
        "certified_Frobenius_upper_bound": float(bound),
        "IIR_gap_controls": controls
    }, indent=2))


if __name__ == "__main__":
    main()
