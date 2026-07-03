"""Temporary diagnostic: find which stage of GBSpatial.fit gets SIGKILLed.
Delete after use."""
import os
import resource
import sys


def rss(tag):
    gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    print(f"[{tag}] peak_rss={gb:.2f}GB", flush=True)


rss("start")
sys.argv = ["x", "--split", "forecast108_119"]
from run_benchmark import load_data  # noqa: E402

data = load_data("forecast108_119")
rss("load_data")

from models_texttower import _load_bill_text  # noqa: E402

bills = _load_bill_text()
rss("_load_bill_text")

from models_idealpoint import IdealPoint, _cell_hash_unit  # noqa: E402

train = data["train"]
u = _cell_hash_unit(train, "earlystop42")
rss("cell_hash")

ip = IdealPoint(k=1)
m_idx, r_idx = ip._encode(train, fit=True)
rss("encode")

print("fitting IdealPoint(k=1) on", len(train), "rows", flush=True)
ip = IdealPoint(k=1).fit(train)
rss("idealpoint_fit_done")
print("es_log_loss", ip.es_log_loss, "epochs", ip.epochs_run, flush=True)

from models_litbaselines import GBSpatial  # noqa: E402
from harness import evaluate  # noqa: E402

gb = GBSpatial(text_mode="tfidf")
gb.fit(train)
rss("gb_fit_done")
print("alpha", gb.alpha, "dev_ll", gb.alpha_dev_log_loss, flush=True)
res = evaluate(gb, train, data["test"], "forecast108_119", "test")
rss("gb_eval_done")
print("test log_loss", round(res.overall["log_loss"], 4),
      "acc", round(res.overall["accuracy"], 4), flush=True)
