"""Source-constrained, native-MNE cross-fold influence audit with a Gram certificate.

Given a fixed *linear* finite-recording filter H, held-out raw positions T,
and training output locations A, the exact worst-case unit-L2 propagation is
sqrt(lambda_max(G)), where G = (P_A H P_T)^T (P_A H P_T).

Batching impulses as MNE 'channels' avoids repeated filter initialization.
Do NOT apply this certificate to nonlinear, fitted or time-varying pipelines.
"""
import json
import time
import warnings
import numpy as np
import mne

FS = 250
N = 14 * FS
TEST = np.arange(4*FS, int(4.04*FS))
GAPS = (0.04, 0.3, 0.8, 1.5, 2.5, 4.0)


def run_filter(x, method):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return mne.filter.filter_data(x, FS, 1, 40, method=method,
                                      phase='zero', verbose=False)


def main():
    records=[]
    impulses=np.zeros((len(TEST),N),dtype=np.float64)
    impulses[np.arange(len(TEST)),TEST]=1.0
    for method in ('fir','iir'):
        # Warm both pathways first: first-ever FIR call includes kernel/FFT
        # initialization and is NOT a representative runtime comparison.
        sequential=np.stack([run_filter(row,method) for row in impulses])
        batched=run_filter(impulses,method)
        single_times=[]; batch_times=[]
        for repeat in range(12):
            sequence=('sequential','batched') if repeat%2==0 else ('batched','sequential')
            for arm in sequence:
                t=time.perf_counter()
                if arm=='sequential':
                    np.stack([run_filter(row,method) for row in impulses])
                    single_times.append(time.perf_counter()-t)
                else:
                    run_filter(impulses,method)
                    batch_times.append(time.perf_counter()-t)
        serial_s=float(np.median(single_times))
        batched_s=float(np.median(batch_times))
        err=np.max(np.abs(batched-sequential))
        assert err<=2e-14, (method, err)
        norm_records=[]
        for gap in GAPS:
            start=round((4.04+gap)*FS)
            train=slice(start,start+2*FS)
            C=batched[:,train].T
            gram=C.T@C
            # Gram's symmetry and PSD are algebraic; tiny numerical negativity is allowed.
            symerr=np.linalg.norm(gram-gram.T,'fro')
            assert symerr<=1e-14
            eigval,eigvec=np.linalg.eigh(gram)
            tight=float(np.sqrt(max(eigval[-1],0.0)))
            fro=float(np.sqrt(max(np.trace(gram),0.0)))
            oracle=float(np.linalg.svd(C,compute_uv=False)[0])
            assert np.isclose(tight,oracle,atol=1e-12,rtol=1e-10), (method,gap,tight,oracle)
            assert tight<=fro+1e-13
            if tight>1e-9:
                right=eigvec[:,-1]
                # A direct, independent one-pass intervention verifies extremality.
                signal=np.random.default_rng(20261026).standard_normal(N)*.01
                altered=signal.copy();altered[TEST]+=right
                np.testing.assert_array_equal(signal[train],altered[train])
                observed=np.linalg.norm(run_filter(altered,method)[train]-run_filter(signal,method)[train])
                assert np.isclose(observed,tight,atol=1e-11,rtol=1e-9)
            norm_records.append({'gap_s':gap,'tight_spectral':tight,'frobenius':fro,'spectral_oracle':oracle})
        records.append({'method':method,'max_batched_vs_sequential_error':float(err),
                        'warmed_sequential_median_s':serial_s,'warmed_batched_median_s':batched_s,
                        'warmed_single_machine_speedup':serial_s/batched_s,
                        'norms':norm_records})
    print(json.dumps({'description':'Native MNE batched Gram/spectral certificate with independent SVD oracle',
                      'numpy':np.__version__,'mne':mne.__version__,'test_points':len(TEST),
                      'records':records},indent=2))
if __name__=='__main__':main()
