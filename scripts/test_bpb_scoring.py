"""Standalone checks for the BPB scoring functions and lm-eval aggregation.

Run with:
    BPB_TASK_DIR=/path/to/oellm-eval/oellm/resources/custom_lm_eval_tasks/bpb \\
        python scripts/test_bpb_scoring.py
"""

import math
import os
import sys
from pathlib import Path

# Point at the bpb task dir inside your oellm-eval checkout.
sys.path.insert(
    0,
    os.environ.get(
        "BPB_TASK_DIR",
        str(Path.home() / "oellm-eval/oellm/resources/custom_lm_eval_tasks/bpb"),
    ),
)

from lm_eval.api.metrics import bits_per_byte  # noqa: E402

import utils  # noqa: E402

failures = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}")
    if not cond:
        failures.append(name)


# ---------------------------------------------------------------- ARC
arc_doc = {
    "question": "Which is a gas?",
    "choices": {"text": ["ice", "steam", "rock", "wood"], "label": ["A", "B", "C", "D"]},
    "answerKey": "B",
}
# gold = "steam" (index 1), 5 bytes
res = [(-8.0, False), (-3.0, True), (-9.0, False), (-10.0, False)]
out = utils.process_results_arc(arc_doc, res)
check("arc: gold ll+bytes", out["bits_per_byte"] == (-3.0, 5), out["bits_per_byte"])
check("arc: acc=1 (argmax is gold)", out["acc"] == 1.0)

# acc_norm normalises by char length: -8/3=-2.67 (ice) beats -3/5=-0.6? No:
# -0.6 > -2.67, so steam still wins.
check("arc: acc_norm=1", out["acc_norm"] == 1.0)

# ---------------------------------------------------------------- OpenBookQA (answerKey has whitespace)
obqa_doc = {
    "question_stem": "A magnet will stick to",
    "choices": {
        "text": ["a plastic cup", "a wooden desk", "a steel nail", "a paper bag"],
        "label": ["A", "B", "C", "D"],
    },
    "answerKey": " C",
}
out = utils.process_results_openbookqa(obqa_doc, [(-5.0, False)] * 3 + [(-1.0, True)])
check(
    "openbookqa: strips answerKey whitespace",
    out["bits_per_byte"] == (-5.0, len("a steel nail")),
    out["bits_per_byte"],
)
check("openbookqa: acc=0 (argmax != gold)", out["acc"] == 0.0)

# ---------------------------------------------------------------- HellaSwag
hs_raw = {
    "activity_label": "Roof shingle removal",
    "ctx_a": "A man is on a roof.",
    "ctx_b": "he",
    "endings": ["starts to rip up shingles.", "flies away.", "eats.", "sings [title]."],
    "label": "0",
}
processed = {
    "query": utils._hellaswag_preprocess(
        hs_raw["activity_label"] + ": " + hs_raw["ctx_a"] + " " + hs_raw["ctx_b"].capitalize()
    ),
    "choices": [utils._hellaswag_preprocess(e) for e in hs_raw["endings"]],
    "gold": int(hs_raw["label"]),
}
out = utils.process_results_hellaswag(processed, [(-2.0, True), (-7.0, False), (-8.0, False), (-9.0, False)])
check(
    "hellaswag: gold pair",
    out["bits_per_byte"] == (-2.0, len("starts to rip up shingles.")),
    out["bits_per_byte"],
)
check("hellaswag: [title] artifact stripped", "[title]" not in processed["choices"][3])

# ---------------------------------------------------------------- PIQA / BoolQ / SocialIQa
out = utils.process_results_piqa(
    {"goal": "open a jar", "sol1": "twist lid", "sol2": "smash it", "label": 0},
    [(-1.5, True), (-4.0, False)],
)
check("piqa: gold pair", out["bits_per_byte"] == (-1.5, len("twist lid")), out["bits_per_byte"])

out = utils.process_results_boolq({"passage": "p", "question": "q", "label": 1}, [(-3.0, False), (-0.5, True)])
check("boolq: gold='yes' 3 bytes", out["bits_per_byte"] == (-0.5, 3), out["bits_per_byte"])

out = utils.process_results_social_iqa(
    {"context": "c", "question": "q", "answerA": "aa", "answerB": "bbbb", "answerC": "cc", "label": "2"},
    [(-5.0, False), (-2.0, True), (-6.0, False)],
)
check("social_iqa: 1-indexed label -> B", out["bits_per_byte"] == (-2.0, 4), out["bits_per_byte"])

# ---------------------------------------------------------------- COPA
copa_doc = {"premise": "The man broke his toe.", "question": "cause", "choice1": "He got a hole in his sock.", "choice2": "He dropped a hammer on his foot.", "label": 1}
check("copa: doc_to_text drops period + connector", utils.doc_to_text_copa(copa_doc) == "The man broke his toe because", utils.doc_to_text_copa(copa_doc))
out = utils.process_results_copa(copa_doc, [(-9.0, False), (-2.0, True)])
check("copa: gold includes leading space", out["bits_per_byte"] == (-2.0, len(" he dropped a hammer on his foot.")), out["bits_per_byte"])

# ---------------------------------------------------------------- CommonsenseQA (cloze)
csqa_doc = {"question": "Where do people keep books?", "choices": {"text": ["shelf", "sky", "fire", "soup", "cloud"], "label": ["A", "B", "C", "D", "E"]}, "answerKey": "A"}
out = utils.process_results_commonsense_qa(csqa_doc, [(-1.0, True)] + [(-6.0, False)] * 4)
check("commonsense_qa: scores answer TEXT not letter", out["bits_per_byte"] == (-1.0, len("shelf")), out["bits_per_byte"])

# ---------------------------------------------------------------- MMLU
out = utils.process_results_mmlu({"question": "2+2?", "choices": ["3", "4", "5", "6"], "answer": 1}, [(-5.0, False), (-0.9, True), (-7.0, False), (-8.0, False)])
check("mmlu: gold pair", out["bits_per_byte"] == (-0.9, 1), out["bits_per_byte"])

# ---------------------------------------------------------------- Winogrande (multiple_input)
wg_doc = {"sentence": "The trophy doesn't fit in the suitcase because _ is too large.", "option1": "the trophy", "option2": "the suitcase", "answer": "1"}
cont = utils.doc_to_target_winogrande(wg_doc)
check("winogrande: continuation extracted", cont == "is too large.", repr(cont))
out = utils.process_results_winogrande(wg_doc, [(-3.0, True), (-4.0, False)])
check("winogrande: bytes on CONTINUATION not context", out["bits_per_byte"] == (-3.0, len("is too large.")), out["bits_per_byte"])

# ---------------------------------------------------------------- malformed doc guard
bad = utils.process_results_arc({"choices": {"text": ["a", "b"], "label": ["A", "B"]}, "answerKey": "Z"}, [(-1.0, True), (-2.0, False)])
check("malformed gold -> neutral pair", bad["bits_per_byte"] == (0.0, 1), bad["bits_per_byte"])

# ---------------------------------------------------------------- aggregation math
# Corpus of 3 docs. BPB = -sum(ll) / sum(bytes) / ln(2)
items = [(-3.0, 5), (-2.0, 4), (-10.0, 20)]
expected = -(-3.0 + -2.0 + -10.0) / (5 + 4 + 20) / math.log(2)
got = bits_per_byte(items)
check("aggregation is byte-weighted corpus BPB", abs(got - expected) < 1e-12, f"{got:.6f} vs {expected:.6f}")

# Sanity: a perfect model (ll=0) gives BPB 0; a model at uniform-random over
# 256 byte values gives 8 bits/byte.
check("perfect model -> 0 bpb", abs(bits_per_byte([(0.0, 10)])) < 1e-12)
random_ll = 10 * math.log(1 / 256)
check("uniform-byte model -> 8.0 bpb", abs(bits_per_byte([(random_ll, 10)]) - 8.0) < 1e-9, bits_per_byte([(random_ll, 10)]))

# Sanity: it is NOT the mean of per-doc ratios (the common bug)
naive = sum(-ll / b / math.log(2) for ll, b in items) / len(items)
check("differs from naive per-doc mean (bug guard)", abs(naive - got) > 1e-6, f"naive={naive:.4f} correct={got:.4f}")

print()
print("FAILURES:", failures if failures else "none")
sys.exit(1 if failures else 0)
