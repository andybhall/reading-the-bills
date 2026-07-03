"""Issue probes: query the trained text tower with hypothetical bill texts.

The two-tower model maps any text to a discrimination vector a = W_a phi;
a member's score on a probe text is x_m . a — the member-specific component
of predicted support. Because the text encoder is open-vocabulary, we can
ask about issues on any wording WITHOUT needing a rollcall to have occurred.

Measurement (not benchmarking): the tower is fit on ALL yea/nay votes,
congresses 108-119 (no held-out eval is reported, so no split is needed;
documented in Notes/decisions.md). Probes are scored for members serving
in the 118th-119th congresses.

Output: Modified Data/results/issue_probe_scores.parquet
        (member x probe score matrix, z-scored per probe)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from models_texttower import TextTower
from models_forecast import question_bucket  # noqa: F401  (keeps import graph explicit)

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "Modified Data"

PROBES = {
    "guns_restrict": "To expand background checks for firearm purchases and "
                     "ban assault weapons.",
    "guns_expand": "To protect the right to keep and bear arms and allow "
                   "concealed carry across state lines.",
    "immigration_restrict": "To increase border security funding, construct "
                            "physical barriers, and expand deportations of "
                            "undocumented immigrants.",
    "immigration_expand": "To provide a pathway to citizenship for "
                          "undocumented immigrants brought to the United "
                          "States as children.",
    "abortion_restrict": "To prohibit abortion after 20 weeks of pregnancy.",
    "military_spending": "To increase appropriations for the Department of "
                         "Defense and expand military readiness programs.",
    "corporate_tax": "To raise the corporate income tax rate and close "
                     "corporate tax loopholes.",
    "environment": "To reduce greenhouse gas emissions and invest in "
                   "renewable energy and climate resilience.",
    "labor_unions": "To strengthen collective bargaining rights and protect "
                    "the right of workers to organize labor unions.",
    "free_trade": "To implement a free trade agreement reducing tariffs on "
                  "imported goods.",
}


def main():
    votes = pd.read_parquet(MOD / "votes.parquet").dropna(subset=["vote"])
    votes = votes[votes["congress"] >= 108]
    links = pd.read_parquet(MOD / "rollcall_bills.parquet")
    feat = links[["congress", "chamber", "rollnumber", "vote_question",
                  "bill_category", "bill_type", "bill_no"]]
    votes = votes.merge(feat, on=["congress", "chamber", "rollnumber"],
                        how="left", validate="m:1")

    # text-ONLY phi: the learned text->discrimination map cannot lean on
    # question-type/sponsorship/agenda cues (v1 probes with meta features
    # failed the party-separation sanity check — see session log)
    model = TextTower(k=16, use_text=False, use_emb=True, mq_offset=False,
                      calibrate=False, name="probe_tower")
    model.meta_features = False
    model.fit(votes)

    # embed probe texts to phi rows matching the training featurization
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    import torch
    probe_rows = []
    for text in PROBES.values():
        e = st.encode([text], normalize_embeddings=True)[0] * 5.0  # match _phi scaling
        probe_rows.append(e.astype(np.float32))
    x, c, W_a, w_b, gamma = model._modules
    phi = torch.as_tensor(np.stack(probe_rows), device=model._dev)
    with torch.no_grad():
        A = W_a(phi).cpu().numpy()          # (n_probes, k) discrimination vectors
    X = x.weight.detach().cpu().numpy()     # (n_members, k) member positions
    scores = X @ A.T                        # member x probe support component

    inv = {i: m for m, i in model.member_index.items()}
    out = pd.DataFrame(scores, columns=list(PROBES))
    out.insert(0, "icpsr", [inv[i] for i in range(len(inv))])

    mem = pd.read_parquet(MOD / "members.parquet")
    recent = mem[mem.congress >= 118].drop_duplicates("icpsr", keep="last")[
        ["icpsr", "bioname", "party_code", "state_abbrev", "chamber"]]
    out = out.merge(recent, on="icpsr", how="inner")
    for k in PROBES:
        out[k] = (out[k] - out[k].mean()) / out[k].std()
    out.to_parquet(MOD / "results" / "issue_probe_scores.parquet", index=False)
    (MOD / "results" / "issue_probes.json").write_text(json.dumps(PROBES, indent=2))

    # within-issue CONTRASTS difference out any common partisan/agenda
    # component shared by both wordings of an issue
    out["guns_contrast"] = out["guns_restrict"] - out["guns_expand"]
    out["immigration_contrast"] = out["immigration_expand"] - out["immigration_restrict"]

    # sanity: party separation per probe (D mean minus R mean, in SDs).
    # expected signs: + for D-coded positions (guns_restrict, immigration
    # _expand, corporate_tax, environment, labor_unions, contrasts);
    # - for R-coded (guns_expand, immigration_restrict, abortion_restrict,
    # military_spending)
    sep = {k: round(float(out.loc[out.party_code == 100, k].mean()
                          - out.loc[out.party_code == 200, k].mean()), 2)
           for k in list(PROBES) + ["guns_contrast", "immigration_contrast"]}
    print("party separation (D mean - R mean, SD units):")
    print(json.dumps(sep, indent=2))

    for probe in ("guns_restrict", "immigration_expand", "free_trade",
                  "military_spending"):
        print(f"\n=== {probe}: HIGHEST member-specific support (118th-119th members)")
        cols = ["bioname", "party_code", "chamber", "state_abbrev", probe]
        print(out.nlargest(6, probe)[cols].round(2).to_string(index=False))
        print(f"=== {probe}: LOWEST")
        print(out.nsmallest(6, probe)[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
