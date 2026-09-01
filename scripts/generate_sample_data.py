#!/usr/bin/env python3
"""
Generate realistic sample Excel datasets for local development.

Replace these with your actual assignment files:
  - data/Media & Research Articles data.xlsx
  - data/Twitter Posts Data.xlsx

Run: python scripts/generate_sample_data.py
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MEDIA_SAMPLES = [
    {
        "Title": "KEYNOTE-189 Trial Shows OS Benefit with Pembrolizumab in NSCLC",
        "Body": (
            "The phase 3 KEYNOTE-189 trial demonstrated a significant improvement "
            "in overall survival when pembrolizumab was added to chemotherapy in "
            "patients with metastatic non-small cell lung cancer. Progression-free "
            "survival also improved. However, nausea, fatigue, and rash were more "
            "common in the pembrolizumab arm, raising safety concerns."
        ),
    },
    {
        "Title": "FDA Grants Accelerated Approval to Trastuzumab Deruxtecan for HER2+ Breast Cancer",
        "Body": (
            "The DESTINY-Breast03 study supported accelerated approval of "
            "trastuzumab deruxtecan for HER2-positive breast cancer. Efficacy "
            "data showed superior progression free survival versus trastuzumab emtansine. "
            "Interstitial lung disease remains a serious safety risk."
        ),
    },
    {
        "Title": "Nivolumab Plus Ipilimumab in Advanced Melanoma: 5-Year Follow-Up",
        "Body": (
            "Long-term follow-up from CheckMate 067 confirms durable overall survival "
            "benefit with nivolumab plus ipilimumab in advanced melanoma. "
            "Immune-related adverse events including colitis and hepatitis "
            "required careful management."
        ),
    },
    {
        "Title": "Osimertinib First-Line in EGFR-Mutated NSCLC: FLAURA2 Results",
        "Body": (
            "The FLAURA2 trial evaluated osimertinib plus chemotherapy versus "
            "osimertinib alone in EGFR-mutated non-small cell lung cancer. "
            "The combination improved progression free survival. Diarrhea and "
            "hematologic toxicities were increased with combination therapy."
        ),
    },
    {
        "Title": "Dupilumab Shows Efficacy in Moderate-to-Severe Atopic Dermatitis",
        "Body": (
            "Phase 3 trials LIBERTY AD SOLO demonstrated that dupilumab significantly "
            "improved skin clearance in atopic dermatitis. The safety profile was "
            "generally favorable with conjunctivitis as the most common adverse event."
        ),
    },
    {
        "Title": "CAR-T Therapy for Relapsed B-Cell Lymphoma: Real-World Outcomes",
        "Body": (
            "Real-world data on axicabtagene ciloleucel for relapsed large B-cell "
            "lymphoma show response rates consistent with ZUMA-1. Cytokine release "
            "syndrome and neurotoxicity remain significant safety concerns in community settings."
        ),
    },
    {
        "Title": "Semaglutide Reduces Cardiovascular Events in Obesity: SELECT Trial",
        "Body": (
            "The SELECT cardiovascular outcomes trial found semaglutide reduced major "
            "adverse cardiovascular events in patients with obesity and established "
            "cardiovascular disease. Gastrointestinal side effects led to discontinuation "
            "in a subset of patients."
        ),
    },
    {
        "Title": "Aducanumab Controversy Continues in Alzheimer's Disease",
        "Body": (
            "Debate persists over aducanumab approval for Alzheimer's disease following "
            "mixed efficacy signals in the EMERGE and ENGAGE trials. Amyloid-related "
            "imaging abnormalities remain a key safety concern. Many clinicians express "
            "skepticism about the clinical benefit."
        ),
    },
]

TWITTER_SAMPLES = [
    {"Body": "Excited about the new KEYNOTE-189 OS data for pembrolizumab in lung cancer! #oncology"},
    {"Body": "PFS looking strong for trastuzumab deruxtecan in DESTINY-Breast03. Game changer for HER2+ mBC."},
    {"Body": "Worried about ILD risk with T-DXd. Safety profile needs more real-world monitoring."},
    {"Body": "CheckMate 067 5-year OS data for nivo+ipi in melanoma is incredible. Durable responses!"},
    {"Body": "FLAURA2 combo arm had more diarrhea but better PFS. Worth the trade-off? #NSCLC"},
    {"Body": "Dupilumab changed my patient's life with severe eczema. Efficacy is real."},
    {"Body": "CAR-T CRS is no joke. We need better toxicity management protocols."},
    {"Body": "Semaglutide SELECT trial = big win for cardio prevention in obesity."},
    {"Body": "Still not convinced about aducanumab. Mixed efficacy data and ARIA side effects."},
    {"Body": "FDA approval news today — manufacturing update for generic aspirin. Not really clinical data."},
    {"Body": "Great panel at #ASCO on immunotherapy combinations. General opinion: combos are the future."},
    {"Body": "New study name dropped: MARIPOSA-2 for amivantamab in EGFR exon 20. Watching closely."},
]


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    media_df = pd.DataFrame(MEDIA_SAMPLES)
    twitter_df = pd.DataFrame(TWITTER_SAMPLES)

    media_path = DATA_DIR / "Media & Research Articles data.xlsx"
    twitter_path = DATA_DIR / "Twitter Posts Data.xlsx"

    media_df.to_excel(media_path, index=False)
    twitter_df.to_excel(twitter_path, index=False)

    print(f"Created {media_path} ({len(media_df)} rows)")
    print(f"Created {twitter_path} ({len(twitter_df)} rows)")
    print("\nReplace these with your actual assignment files when available.")


if __name__ == "__main__":
    main()
