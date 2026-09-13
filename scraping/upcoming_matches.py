#!/usr/bin/env python3
"""Provider-neutral fixture ingestion. CSV works now; add only authorized APIs."""
from __future__ import annotations
import argparse, os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

REQUIRED = ["event_id","start_time","tour","level","surface","round","player_1_name","player_2_name","player_1_odds","player_2_odds"]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--days",type=int,default=2); ap.add_argument("--output",required=True); a=ap.parse_args()
    load_dotenv(); provider=os.getenv("UPCOMING_PROVIDER","csv").lower()
    if provider != "csv":
        raise ValueError("Connector not configured. Use an authorized feed adapter; scraping bypass is intentionally not included.")
    src=Path(os.getenv("UPCOMING_CSV_PATH","data/upcoming/provider_matches.csv"))
    if not src.exists():
        pd.DataFrame(columns=REQUIRED).to_csv(src,index=False)
        raise FileNotFoundError(f"Template created at {src}. Populate it from an authorized fixture source and rerun.")
    df=pd.read_csv(src); missing=sorted(set(REQUIRED)-set(df.columns))
    if missing: raise ValueError(f"Missing provider columns: {missing}")
    df["start_time"]=pd.to_datetime(df["start_time"],errors="coerce",utc=True)
    now=pd.Timestamp.now(tz="UTC").normalize(); end=now+pd.Timedelta(days=a.days)
    df=df.loc[df["start_time"].between(now,end,inclusive="left")]
    df=df.loc[df["level"].astype(str).str.upper().isin(["ATP","CHALLENGER"])]
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); df.to_parquet(a.output,index=False)
    print(f"Fixtures written: {len(df)} -> {a.output}")
if __name__=="__main__": main()
