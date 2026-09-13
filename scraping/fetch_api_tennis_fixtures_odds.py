#!/usr/bin/env python3
"""Descarga ATP/Challenger de hoy y manana con cuotas desde API-Tennis.

Requiere API_TENNIS_KEY en .env. Usa endpoints documentados, no scraping HTML.
Salida compatible con upcoming_matches.py.
"""
from __future__ import annotations
import argparse, os, sys, time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import httpx
import pandas as pd
from dotenv import load_dotenv

BASE='https://api.api-tennis.com/tennis/'
ALLOWED={'Atp Singles','Challenger Men Singles'}
COLUMNS=['event_id','start_time','tour','level','surface','round','player_1_name','player_2_name','player_1_odds','player_2_odds','bookmaker','odds_updated_at','source']

def call(client, key, method, **params):
    q={'method':method,'APIkey':key,**{k:v for k,v in params.items() if v is not None}}
    for n in range(4):
        try:
            r=client.get(BASE,params=q); r.raise_for_status(); data=r.json()
            if str(data.get('success','1')) not in ('1','True','true'): raise RuntimeError(data.get('error') or data.get('message') or str(data))
            return data.get('result',[])
        except Exception:
            if n==3: raise
            time.sleep(2**n)

def pick(d,*names,default=None):
    for n in names:
        if n in d and d[n] not in (None,''): return d[n]
    return default

def decimal(x):
    try:
        v=float(x); return v if v>1 else None
    except: return None

def odds_index(raw):
    out={}
    items=raw.values() if isinstance(raw,dict) else raw
    for item in items or []:
        if not isinstance(item,dict): continue
        event=str(pick(item,'event_key','match_id','id',default=''))
        home=decimal(pick(item,'Home/Away','home','player1','odd_1','home_od'))
        away=decimal(pick(item,'Home/Away 2','away','player2','odd_2','away_od'))
        # Some responses nest bookmakers/markets. Search shallow nested dicts/lists.
        if not (home and away):
            stack=list(item.values())
            for node in stack:
                if isinstance(node,dict):
                    home=home or decimal(pick(node,'home','Home','1','odd_1'))
                    away=away or decimal(pick(node,'away','Away','2','odd_2'))
                elif isinstance(node,list):
                    for z in node:
                        if isinstance(z,dict):
                            home=home or decimal(pick(z,'home','Home','1','odd_1'))
                            away=away or decimal(pick(z,'away','Away','2','odd_2'))
        if event and home and away:
            out[event]=(home,away,str(pick(item,'odd_bookmakers','bookmaker',default='API-Tennis')))
    return out

def main():
    load_dotenv(); ap=argparse.ArgumentParser(); ap.add_argument('--output',default='data/upcoming/provider_matches.csv'); ap.add_argument('--timezone',default=os.getenv('TIMEZONE','Europe/Madrid')); a=ap.parse_args()
    key=os.getenv('API_TENNIS_KEY');
    if not key: raise RuntimeError('Falta API_TENNIS_KEY en .env')
    tz=ZoneInfo(a.timezone); today=datetime.now(tz).date(); tomorrow=today+timedelta(days=1)
    with httpx.Client(timeout=45,headers={'User-Agent':'SmartTennisAnalytics/2.0'}) as c:
        fixtures=call(c,key,'get_fixtures',date_start=today.isoformat(),date_stop=tomorrow.isoformat())
        try: raw_odds=call(c,key,'get_odds',date_start=today.isoformat(),date_stop=tomorrow.isoformat())
        except Exception as e: print(f'[WARN] No se pudieron obtener cuotas: {e}',file=sys.stderr); raw_odds=[]
    odds=odds_index(raw_odds); rows=[]
    for m in fixtures:
        if not isinstance(m,dict): continue
        event_type=str(pick(m,'event_type_type','league_name',default=''))
        if event_type not in ALLOWED: continue
        p1=pick(m,'event_first_player','player1','home_team'); p2=pick(m,'event_second_player','player2','away_team')
        if not p1 or not p2: continue
        event=str(pick(m,'event_key','match_id','id',default=''))
        date=str(pick(m,'event_date','date',default=today.isoformat())); tm=str(pick(m,'event_time','time',default='00:00'))
        start=pd.to_datetime(f'{date} {tm}',errors='coerce')
        if pd.isna(start): continue
        start=start.tz_localize(tz).tz_convert('UTC')
        o1=o2=None; book=''
        if event in odds: o1,o2,book=odds[event]
        rows.append({'event_id':event,'start_time':start.isoformat(),'tour':pick(m,'tournament_name','league_name',default=''),'level':'ATP' if event_type=='Atp Singles' else 'CHALLENGER','surface':pick(m,'event_ground','surface',default='Unknown'),'round':pick(m,'event_round','round',default=''),'player_1_name':p1,'player_2_name':p2,'player_1_odds':o1,'player_2_odds':o2,'bookmaker':book,'odds_updated_at':datetime.now(tz).isoformat(),'source':'API-Tennis'})
    df=pd.DataFrame(rows,columns=COLUMNS).sort_values(['tour','start_time'],ignore_index=True) if rows else pd.DataFrame(columns=COLUMNS)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(out,index=False,encoding='utf-8-sig')
    print(f'CSV creado: {out} | partidos={len(df)} | con cuotas={int(df.player_1_odds.notna().sum()) if len(df) else 0}')
if __name__=='__main__': main()
