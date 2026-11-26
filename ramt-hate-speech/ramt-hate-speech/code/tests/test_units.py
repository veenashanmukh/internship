"""Unit tests: rationale aggregation/alignment, random control, losses, attention
masking, heads, schedule quotas, thresholds."""
import numpy as np
import pytest
import torch

from ramt.data.rationales import aggregate, align_to_subwords, random_matched, subword_to_word
from ramt.data.schedule import round_quotas, round_sequence, run_length
from ramt.evaluate import bootstrap_threshold, f1_curve, THRESH_GRID, aopc_faithfulness, plausibility
from ramt.losses import pos_weight_from_labels, rationale_loss, task_loss


# ---------------------------------------------------------------- Section 3 steps 1-2, 5
def test_aggregation_methods():
    v = [[1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]]
    np.testing.assert_allclose(aggregate(v, 4, "soft"), [1, 1 / 3, 1 / 3, 0], rtol=1e-6)
    np.testing.assert_array_equal(aggregate(v, 4, "majority"), [1, 0, 0, 0])
    np.testing.assert_array_equal(aggregate(v, 4, "union"), [1, 1, 1, 0])
    np.testing.assert_array_equal(aggregate(v, 4, "intersection"), [1, 0, 0, 0])
    assert aggregate([], 3).sum() == 0


def test_subword_alignment_even_split_and_specials():
    rho = np.array([1.0, 0.5, 0.0], dtype=np.float32)
    wid = [None, 0, 0, 1, 2, 2, 2, None]                 # [CLS] w0 w0 w1 w2 w2 w2 [SEP]
    q = align_to_subwords(rho, wid)
    np.testing.assert_allclose(q, [0, .5, .5, .5, 0, 0, 0, 0])
    assert q[0] == 0 and q[-1] == 0                      # special tokens never carry mass
    np.testing.assert_allclose(q.sum(), rho.sum())       # word mass preserved
    np.testing.assert_allclose(subword_to_word(q, wid, 3), rho)


def test_alignment_truncated_words_vanish():
    rho = np.array([0, 0, 1.0], dtype=np.float32)
    q = align_to_subwords(rho, [None, 0, 1, None])        # word 2 cut by truncation
    assert q.sum() == 0                                   # -> post becomes ineligible


def test_random_matched_is_fixed_and_preserves_values():
    rho = np.array([1, 0, .5, 0, 0], dtype=np.float32)
    a, b = random_matched(rho, "post-1"), random_matched(rho, "post-1")
    np.testing.assert_array_equal(a, b)                   # fixed per post
    np.testing.assert_array_equal(np.sort(a), np.sort(rho))
    assert any(not np.array_equal(random_matched(rho, f"p{i}"), rho) for i in range(10))


# ---------------------------------------------------------------- losses (D9 / A3)
def test_rationale_loss_empty_batch_is_exact_zero_no_nan():
    B, L = 4, 6
    alpha = torch.softmax(torch.randn(B, L, requires_grad=True), -1)
    q = torch.zeros(B, L)
    loss = rationale_loss(alpha, q, torch.zeros(B, dtype=torch.bool))
    assert loss.item() == 0.0 and torch.isfinite(loss)
    loss.backward()                                       # graph valid, zero gradient


def test_rationale_loss_divides_by_batch_size():
    alpha = torch.tensor([[.5, .5, 0, 0], [.25, .25, .25, .25], [1, 0, 0, 0]])
    q = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0.]])
    elig = torch.tensor([True, False, True])
    got = rationale_loss(alpha, q, elig).item()
    want = (-np.log(.5 + 1e-8) + -(.5 * np.log(1 + 1e-8) + .5 * np.log(0 + 1e-8))) / 3
    assert abs(got - want) < 1e-4


def test_task_losses_finite():
    y = torch.tensor([[1., 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]])
    assert torch.isfinite(task_loss("explicit", torch.randn(2, 6), y, torch.ones(6)))
    assert torch.isfinite(task_loss("implicit", torch.randn(2, 1), y[:, :1], torch.ones(1)))
    assert torch.isfinite(task_loss("hx", torch.randn(2, 3), torch.tensor([0, 2]), None))
    pw = pos_weight_from_labels(np.array([[1], [0], [0], [0], [0]]))
    assert abs(pw.item() - 2.0) < 1e-6                    # sqrt(4/1)


# ---------------------------------------------------------------- attention + heads
def test_attention_only_on_content_tokens(tiny_encoder):
    from ramt.model import RAMT
    m = RAMT(tiny_encoder, ["explicit", "implicit", "hx"], "task").eval()
    ids = torch.tensor([[2, 10, 11, 12, 3, 0, 0], [2, 13, 3, 0, 0, 0, 0]])
    am = (ids != 0).long()
    cm = torch.tensor([[0, 1, 1, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)
    for t, n in (("explicit", 6), ("implicit", 1), ("hx", 3)):
        lg, a = m(ids, am, cm, t)
        assert lg.shape == (2, n)
        assert torch.all(a[~cm] == 0)
        torch.testing.assert_close(a.sum(-1), torch.ones(2))
        assert a.dtype == torch.float32
    mc = RAMT(tiny_encoder, ["hx"], "cls").eval()
    lg, a = mc(ids, am, cm, "hx")
    assert a is None and lg.shape == (2, 3)
    ms = RAMT(tiny_encoder, ["explicit", "hx"], "shared").eval()
    a1 = ms(ids, am, cm, "explicit")[1]; a2 = ms(ids, am, cm, "hx")[1]
    torch.testing.assert_close(a1, a2)                    # one pool for all heads


# ---------------------------------------------------------------- schedule (D8)
def test_quotas_and_exposure_matching():
    sizes = {"explicit": 44603, "implicit": 14274, "hx": 15383}
    q = round_quotas(sizes, 1000, 2.0)
    assert sum(q.values()) == 1000
    assert q == {"explicit": 464, "implicit": 263, "hx": 273}   # largest-remainder rounding (see log)
    rng = np.random.RandomState(0)
    seq = round_sequence(["explicit", "hx"], q, rng)
    assert seq.count("explicit") == 464 and seq.count("hx") == 273 and "implicit" not in seq
    assert run_length(["implicit"], q, 6) == 6 * 263
    assert run_length(["explicit", "implicit", "hx"], q, 6) == 6000


# ---------------------------------------------------------------- thresholds / explanation metrics
def test_f1_curve_matches_sklearn():
    from sklearn.metrics import f1_score
    rng = np.random.RandomState(1)
    y = rng.randint(0, 2, 300); p = rng.rand(300)
    c = f1_curve(y, p)
    for i in (0, 30, 60, 98):
        assert abs(c[i] - f1_score(y, (p >= THRESH_GRID[i]).astype(int), zero_division=0)) < 1e-9
    t = bootstrap_threshold(y, p, 50)
    assert 0.01 <= t <= 0.99


def test_plausibility_and_aopc():
    gold = [np.array([0, 1, 0, 1]), np.array([1, 0, 0])]
    perfect = [np.array([0, .9, .1, .8]), np.array([.9, .1, 0])]
    r = plausibility(perfect, gold)
    assert r["token_f1_at_k"] == 1.0 and r["token_auprc"] == 1.0
    words = [["a", "bad", "c", "bad"], ["bad", "x", "y"]]
    fn = lambda wl: np.array([sum(w == "bad" for w in ws) / 2 for ws in wl], dtype=float)
    f = aopc_faithfulness(fn, words, [np.array([0, 1, 0, 1.]), np.array([1, 0, 0.])], random_draws=3)
    assert f["aopc_comprehensiveness"] > f["random_aopc_comprehensiveness"]
    assert f["comp_minus_random"] > 0


def test_encoder_stored_in_fp16_is_loaded_as_fp32(tiny_encoder, tmp_path):
    """B0 regression: DeBERTa-v3 ships fp16 weights; AMP needs fp32 master weights."""
    from transformers import AutoModel
    from ramt.model import RAMT
    d = tmp_path / "half"
    AutoModel.from_pretrained(tiny_encoder).half().save_pretrained(str(d))
    m = RAMT(str(d), ["hx"], "task")
    assert all(p.dtype == torch.float32 for p in m.parameters())
