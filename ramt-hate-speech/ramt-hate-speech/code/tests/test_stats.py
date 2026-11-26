import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from ramt import stats as S


def _data(n=400, seed=0, ties=True):
    r = np.random.RandomState(seed)
    y = r.randint(0, 2, n)
    s = r.rand(n) + 0.4 * y
    if ties:
        s = np.round(s, 1)                      # heavy ties, as with fp16 logits
    return y, s


@pytest.mark.parametrize("ties", [False, True])
def test_weighted_ap_and_auc_match_sklearn(ties):
    y, s = _data(ties=ties)
    assert S.w_average_precision(y, s, np.ones(len(y)))[0] == pytest.approx(average_precision_score(y, s), abs=1e-10)
    assert S.w_roc_auc(y, s, np.ones(len(y)))[0] == pytest.approx(roc_auc_score(y, s), abs=1e-10)
    # integer weights == explicit duplication
    w = np.random.RandomState(1).randint(0, 3, len(y))
    yy, ss = np.repeat(y, w), np.repeat(s, w)
    assert S.w_average_precision(y, s, w)[0] == pytest.approx(average_precision_score(yy, ss), abs=1e-10)
    assert S.w_roc_auc(y, s, w)[0] == pytest.approx(roc_auc_score(yy, ss), abs=1e-10)


def test_weighted_macro_f1_and_mean():
    r = np.random.RandomState(2)
    y, p = r.randint(0, 3, 300), r.randint(0, 3, 300)
    w = r.randint(0, 3, 300)
    assert S.w_macro_f1(y, p, np.ones(300), 3)[0] == pytest.approx(f1_score(y, p, average="macro"))
    assert S.w_macro_f1(y, p, w, 3)[0] == pytest.approx(f1_score(np.repeat(y, w), np.repeat(p, w), average="macro"))
    x = r.rand(300); x[::7] = np.nan
    assert S.w_mean(x, np.ones(300))[0] == pytest.approx(np.nanmean(x))


def test_bootstrap_is_deterministic_and_paired():
    a = np.concatenate([c for _, c in S.boot_counts(50, B=30, seed=3, chunk=7)])
    b = np.concatenate([c for _, c in S.boot_counts(50, B=30, seed=3, chunk=7)])
    assert a.shape == (30, 50) and np.array_equal(a, b) and np.all(a.sum(1) == 50)


def test_holm():
    adj = S.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": 0.03, "b": 0.06, "c": 0.06}
    adj2 = S.holm({"a": 0.01, "nan": float("nan")})
    assert adj2 == {"a": 0.02, "nan": 1.0}                      # NaN never significant, still counted


def test_combined_test_floor_and_direction():
    r = np.random.RandomState(0)
    K, B = 3, 500
    A = np.array([0.80, 0.80, 0.80]); Bv = np.array([0.70, 0.70, 0.70])   # s_D = 0 -> floor must apply
    bA = A[:, None] + r.normal(0, 0.005, (K, B)); bB = Bv[:, None] + r.normal(0, 0.005, (K, B))
    res = S.combined_test(A, Bv, bA, bB, sigma2_pool=0.01 ** 2, df_pool=18)
    assert res["seed_term"] == "pooled floor" and res["diff"] == pytest.approx(0.1)
    assert res["p"] < 1e-3 and res["sign_consistent"] and res["ci_combined"][0] > 0
    # a null effect is not significant
    A2 = np.array([0.70, 0.72, 0.68]); B2 = np.array([0.71, 0.69, 0.70])
    res0 = S.combined_test(A2, B2, A2[:, None] + 0 * bA, B2[:, None] + 0 * bB, 0.01 ** 2, 18)
    assert res0["p"] > 0.2
    # flagged configurations use s_D^2 with K-1 df
    res2 = S.combined_test(A2, B2, A2[:, None] + 0 * bA, B2[:, None] + 0 * bB, 1e-8, 18, flagged=True)
    assert res2["seed_term"] == "s_D^2"


def test_pooled_variance():
    s2, df = S.pooled_seed_variance({"a": [1.0, 2.0, 3.0], "b": [2.0, 2.0], "c": [5.0]})
    assert df == 3 and s2 == pytest.approx((2 * 1.0 + 0.0) / 3)
    assert S.generalised_mean([0.5, 0.5]) == pytest.approx(0.5)


def test_combined_test_nan_is_never_significant():
    nanA = np.array([np.nan, 0.5, 0.6]); B = np.array([0.5, 0.5, 0.5])
    r = S.combined_test(nanA, B, np.full((3, 10), np.nan), np.zeros((3, 10)), 0.01, 10)
    assert np.isnan(r["p"]) and not r["sign_consistent"]
