#!/usr/bin/env python3
"""Entrena modelos supervisados temporales para primer set, ganar un set y overs.

Contrato:
- train ajusta candidatos; tune elige algoritmo; train+tune reajusta;
- calibration ajusta Platt; test solo informa;
- simetria mediante aumento P1/P2;
- las líneas over se corrigen para conservar monotonicidad.
"""
from __future__ import annotations
import argparse, gc, json, math, os, time
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

OVER_LINES_BO3=(18.5,19.5,20.5,21.5,22.5,23.5,24.5,25.5)
TARGETS={
 "first_set":"target_player_1_wins_first_set",
 "win_any_set":"target_player_1_wins_any_set",
 **{f"over_{x:.1f}":f"target_over_{str(x).replace('.', '_')}" for x in OVER_LINES_BO3},
}
EXCLUDED_PREFIXES=("target_","winner_","loser_","w_","l_","td_","source_")
EXCLUDED={"split","score","match_date","player_1_id","player_2_id","player_1_name","player_2_name","tourney_id","tourney_name","match_num","processing_sequence"}

def atomic_json(x:Any,p:Path):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.new'); t.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)); os.replace(t,p)

def log(message: str) -> None:
 print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

def elapsed(started: float) -> str:
 seconds=max(time.perf_counter()-started,0.0)
 return f"{seconds/60:.1f} min" if seconds>=60 else f"{seconds:.1f} s"

def features_from_metadata(frame, metadata, training_frame):
 candidates=(metadata.get("no_market_feature_columns") or
             metadata.get("market_feature_columns") or
             metadata.get("features") or [])
 result=[]; excluded_empty=[]; excluded_low=[]; minimum=max(100,int(.01*len(training_frame)))
 for c in candidates:
  if c not in frame or c in EXCLUDED or c.startswith(EXCLUDED_PREFIXES): continue
  values=pd.to_numeric(training_frame[c],errors="coerce")
  count=int(values.notna().sum())
  if count==0: excluded_empty.append(str(c)); continue
  if count<minimum: excluded_low.append(str(c)); continue
  finite=values.dropna().to_numpy(dtype=np.float32,copy=False)
  if len(finite) and not np.isfinite(finite).all(): continue
  result.append(str(c))
 if not result: raise RuntimeError("No hay features numericas validas en metadata")
 log(f"Features seleccionadas: {len(result)}; vacias excluidas: {excluded_empty}; baja cobertura: {len(excluded_low)}")
 return result

def X(frame,features):
 # float32 halves memory versus the original float64 matrix.
 result=pd.DataFrame(index=frame.index)
 for c in features:
  result[c]=pd.to_numeric(frame[c],errors="coerce").astype("float32")
 return result.replace([np.inf,-np.inf],np.nan)

def swap(frame,features):
 out={}; nan=pd.Series(np.nan,index=frame.index,dtype="float32")
 for c in features:
  if c.startswith('player_1_'): out[c]=frame.get('player_2_'+c[9:],nan)
  elif c.startswith('player_2_'): out[c]=frame.get('player_1_'+c[9:],nan)
  elif c.startswith('diff_'): out[c]=-pd.to_numeric(frame[c],errors='coerce')
  elif c.endswith('_p1_p2'):
   v=pd.to_numeric(frame[c],errors='coerce'); out[c]=np.where(v.ne(0),1/v,np.nan)
  elif c in {'elo_expected_p1','surface_elo_expected_p1'}: out[c]=1-pd.to_numeric(frame[c],errors='coerce')
  elif '_player_1_before' in c:
   v=pd.to_numeric(frame[c],errors='coerce')
   out[c]=1-v if any(t in c for t in ('rate','share','probability')) else (-v if any(t in c for t in ('score','intransitivity')) else v)
  else: out[c]=frame[c]
 return X(pd.DataFrame(out,index=frame.index),features)

def sample_temporally(frame, maximum_rows, seed):
 if maximum_rows<=0 or len(frame)<=maximum_rows: return frame
 # Stratified random cap within the historical period. The split itself remains temporal.
 return frame.sample(n=maximum_rows,random_state=seed).sort_values('match_date',kind='mergesort')

def candidate(kind):
 if kind=='logistic_sgd':
  # Constant imputation is O(n*p) and avoids SimpleImputer(strategy='median'),
  # which sorted a 482250 x 482 float64 masked array and caused the 1.73 GiB allocation.
  return Pipeline([
   ('impute',SimpleImputer(strategy='constant',fill_value=0.0,add_indicator=False,keep_empty_features=True)),
   ('scale',MaxAbsScaler()),
   ('model',SGDClassifier(loss='log_loss',penalty='elasticnet',alpha=1e-4,l1_ratio=.05,
                          max_iter=300,tol=1e-4,class_weight='balanced',
                          average=True,random_state=20260910))])
 # HistGradientBoosting accepts NaN directly, so no imputer or duplicate matrix is needed.
 return HistGradientBoostingClassifier(learning_rate=.05,max_iter=220,max_leaf_nodes=31,
                                       min_samples_leaf=50,l2_regularization=2,
                                       early_stopping=True,validation_fraction=.1,
                                       n_iter_no_change=20,random_state=20260910)

def augmented(frame,features,target,mode,maximum_base_rows,seed):
 y=pd.to_numeric(frame[target],errors='coerce')
 other=(pd.to_numeric(frame.get('target_player_2_wins_any_set'),errors='coerce')
        if mode=='win_any_set' else None)
 valid=y.isin([0,1]) & (other.isin([0,1]) if other is not None else True)
 base=sample_temporally(frame.loc[valid],maximum_base_rows,seed)
 y=pd.to_numeric(base[target],errors='coerce')
 other=(pd.to_numeric(base['target_player_2_wins_any_set'],errors='coerce')
        if mode=='win_any_set' else None)
 xo=X(base,features); xs=swap(base,features)
 ys=other if mode=='win_any_set' else ((1-y) if mode=='first_set' else y)
 xa=pd.concat([xo,xs],ignore_index=True,copy=False).astype('float32',copy=False)
 ya=pd.concat([y,ys],ignore_index=True).astype('int8')
 del xo,xs,base
 gc.collect()
 return xa,ya
def expected_calibration_error(y, p, bins=10):
 y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float); edges=np.linspace(0.0,1.0,bins+1)
 indices=np.clip(np.digitize(p,edges[1:-1],right=True),0,bins-1); total=len(y); ece=0.0
 for index in range(bins):
  mask=indices==index
  if mask.any(): ece += float(mask.sum()/total*abs(p[mask].mean()-y[mask].mean()))
 return ece

def calibration_parameters(y,p):
 y=np.asarray(y,dtype=int); p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
 if len(np.unique(y))<2: return math.nan,math.nan
 z=np.log(p/(1-p)).reshape(-1,1); model=LogisticRegression(C=1e6,max_iter=1000).fit(z,y)
 return float(model.intercept_[0]),float(model.coef_[0,0])

def metrics(y,p):
 y=pd.to_numeric(y,errors='coerce'); p=np.asarray(p,float); m=y.isin([0,1])&np.isfinite(p); y=y[m].astype(int); p=np.clip(p[m],1e-4,1-1e-4)
 positive=float(y.mean()); baseline_ll=float(-(positive*math.log(max(positive,1e-12))+(1-positive)*math.log(max(1-positive,1e-12))))
 baseline_brier=float(positive*(1-positive)); intercept,slope=calibration_parameters(y.to_numpy(),p)
 ll=float(log_loss(y,p,labels=[0,1])); brier=float(brier_score_loss(y,p))
 return {'rows':int(len(y)),'positive_rate':positive,'mean_probability':float(p.mean()),'log_loss':ll,
  'baseline_log_loss':baseline_ll,'log_loss_gain_vs_baseline':baseline_ll-ll,'brier':brier,
  'baseline_brier':baseline_brier,'brier_gain_vs_baseline':baseline_brier-brier,
  'auc':float(roc_auc_score(y,p)) if y.nunique()==2 else math.nan,
  'ece_10_bins':expected_calibration_error(y.to_numpy(),p,10),
  'calibration_intercept':intercept,'calibration_slope':slope}

def calibration_bands(market,period,y,p,bins=10):
 data=pd.DataFrame({'target':pd.to_numeric(y,errors='coerce'),'probability':np.asarray(p,float)}).dropna()
 data=data[data.target.isin([0,1])].copy(); data['probability']=data.probability.clip(0,1)
 data['probability_band']=pd.cut(data.probability,np.linspace(0,1,bins+1),include_lowest=True,duplicates='drop')
 result=data.groupby('probability_band',observed=True).agg(rows=('target','size'),mean_probability=('probability','mean'),observed_rate=('target','mean')).reset_index()
 result.insert(0,'period',period); result.insert(0,'market',market); result['calibration_gap']=result.mean_probability-result.observed_rate
 return result

def fit_platt(y,p):
 y=pd.to_numeric(y,errors='coerce'); p=np.asarray(p,float); m=y.isin([0,1])&np.isfinite(p); z=np.log(np.clip(p[m],1e-6,1-1e-6)/(1-np.clip(p[m],1e-6,1-1e-6))).reshape(-1,1)
 model=LogisticRegression(C=1e6).fit(z,y[m].astype(int)); return model

def apply_platt(model,p):
 p=np.clip(np.asarray(p,float),1e-6,1-1e-6); z=np.log(p/(1-p)).reshape(-1,1); return model.predict_proba(z)[:,1]

def temporal_parts(frame):
 split=frame.split.astype('string').str.lower(); train=frame[split.eq('train')].copy(); val=frame[split.eq('validation')].sort_values('match_date').copy(); test=frame[split.eq('test')].copy()
 if val.empty or test.empty: raise RuntimeError('Se requieren validation y test')
 dates=pd.to_datetime(val.match_date).dt.normalize(); unique=np.sort(dates.unique()); cut=unique[len(unique)//2]
 tune=val[dates<cut].copy(); cal=val[dates>=cut].copy()
 if train.empty or tune.empty or cal.empty: raise RuntimeError('Split temporal vacio')
 if pd.to_datetime(train.match_date).max()>=pd.to_datetime(tune.match_date).min() or pd.to_datetime(tune.match_date).max()>=pd.to_datetime(cal.match_date).min() or pd.to_datetime(cal.match_date).max()>=pd.to_datetime(test.match_date).min(): raise RuntimeError('Solapamiento temporal')
 return train,tune,cal,test

def main():
 ap=argparse.ArgumentParser(description='Entrenador temporal y memory-safe de mercados')
 ap.add_argument('--dataset',required=True); ap.add_argument('--metadata',required=True)
 ap.add_argument('--output-dir',required=True); ap.add_argument('--min-rows',type=int,default=1000)
 ap.add_argument('--max-base-train-rows',type=int,default=200000,
                 help='Maximo de filas originales antes del aumento P1/P2; 0 desactiva el limite')
 ap.add_argument('--max-tune-rows',type=int,default=100000)
 ap.add_argument('--max-calibration-rows',type=int,default=100000)
 ap.add_argument('--max-test-rows',type=int,default=0,
                 help='0 evalua todo test; use un limite si falta memoria')
 ap.add_argument('--min-segment-rows',type=int,default=100)
 ap.add_argument('--symmetry-rows',type=int,default=2000)
 a=ap.parse_args(); total_started=time.perf_counter()
 out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
 log(f"Leyendo dataset: {a.dataset}"); frame=pd.read_parquet(a.dataset)
 meta=json.loads(Path(a.metadata).read_text(encoding='utf-8-sig'))
 train,tune,cal,test=temporal_parts(frame)
 features=features_from_metadata(frame,meta,pd.concat([train,tune],copy=False))
 log(f"Filas split: train={len(train):,}, tune={len(tune):,}, calibration={len(cal):,}, test={len(test):,}")
 bundle={'schema_version':2,'method':'memory_safe_direct_markets','features':features,
  'orientation_contract':meta.get('market_orientation_contract',{}),'models':{},'calibrators':{},
  'over_lines_bo3':list(OVER_LINES_BO3),'monotonic_policy':'cumulative_min_in_ascending_line_order',
  'training_caps':{'max_base_train_rows':a.max_base_train_rows,'max_tune_rows':a.max_tune_rows,
                   'max_calibration_rows':a.max_calibration_rows,'max_test_rows':a.max_test_rows}}
 rows=[]; calibration_frames=[]; segment_rows=[]; symmetry_rows=[]
 context_columns=[c for c in ('match_date','competition_type','surface','tourney_level','round','best_of','cold_start_any_player','synthetic_player_any','stats_reliable_both','surface_stats_reliable_both') if c in test.columns]
 predictions=test[context_columns].copy(); market_count=len(TARGETS)
 for market_number,(market,target) in enumerate(TARGETS.items(),start=1):
  market_started=time.perf_counter(); mode='first_set' if market=='first_set' else ('win_any_set' if market=='win_any_set' else 'over')
  train_use=train if mode!='over' else train[pd.to_numeric(train.best_of,errors='coerce').eq(3)]
  tune_use=tune if mode!='over' else tune[pd.to_numeric(tune.best_of,errors='coerce').eq(3)]
  cal_use=cal if mode!='over' else cal[pd.to_numeric(cal.best_of,errors='coerce').eq(3)]
  test_use=test if mode!='over' else test[pd.to_numeric(test.best_of,errors='coerce').eq(3)]
  if target not in frame: raise RuntimeError(f'Falta target {target}')
  valid_train=int(pd.to_numeric(train_use[target],errors='coerce').isin([0,1]).sum())
  if valid_train<a.min_rows: raise RuntimeError(f'Muestra insuficiente {market}: {valid_train:,}')
  log(f"[{market_number}/{market_count}] {market}: valid_train={valid_train:,}")
  xt,yt=augmented(train_use,features,target,mode,a.max_base_train_rows,20260910+market_number)
  log(f"[{market_number}/{market_count}] Matriz train aumentada: {len(xt):,} x {len(features):,} ({xt.memory_usage(deep=True).sum()/2**30:.2f} GiB)")
  tune_eval=sample_temporally(tune_use,a.max_tune_rows,20261010+market_number)
  yv=pd.to_numeric(tune_eval[target],errors='coerce'); vv=yv.isin([0,1]); tune_x=X(tune_eval.loc[vv],features)
  best=None; best_ll=math.inf
  for candidate_number,name in enumerate(('logistic_sgd','hist_gb'),start=1):
   fit_started=time.perf_counter(); log(f"[{market_number}/{market_count}] Candidato {candidate_number}/2 {name}: entrenando...")
   model_candidate=candidate(name).fit(xt,yt)
   pv=model_candidate.predict_proba(tune_x)[:,1]; mm=metrics(yv.loc[vv],pv)
   rows.append({'market':market,'period':'tune','candidate':name,**mm})
   log(f"[{market_number}/{market_count}] {name} listo en {elapsed(fit_started)}; log_loss={mm['log_loss']:.6f}; brier={mm['brier']:.6f}")
   if mm['log_loss']<best_ll: best_ll=mm['log_loss']; best=name
   del model_candidate,pv; gc.collect()
  del tune_x,xt,yt; gc.collect(); log(f"[{market_number}/{market_count}] Seleccionado: {best}")
  fit_frame=pd.concat([train_use,tune_use],copy=False)
  xa,ya=augmented(fit_frame,features,target,mode,a.max_base_train_rows,20261110+market_number)
  refit_started=time.perf_counter(); log(f"[{market_number}/{market_count}] Reentrenando {best} con train+tune: {len(xa):,} filas...")
  model=candidate(best).fit(xa,ya); del xa,ya,fit_frame; gc.collect()
  log(f"[{market_number}/{market_count}] Reentrenamiento listo en {elapsed(refit_started)}")
  cal_eval=sample_temporally(cal_use,a.max_calibration_rows,20261210+market_number)
  yc=pd.to_numeric(cal_eval[target],errors='coerce'); vc=yc.isin([0,1]); cal_x=X(cal_eval.loc[vc],features)
  rawc=model.predict_proba(cal_x)[:,1]; platt=fit_platt(yc.loc[vc],rawc); pc=apply_platt(platt,rawc)
  cm=metrics(yc.loc[vc],pc); rows.append({'market':market,'period':'calibration','candidate':best,**cm})
  calibration_frames.append(calibration_bands(market,'calibration',yc.loc[vc],pc))
  del cal_x,rawc,pc,cal_eval; gc.collect(); log(f"[{market_number}/{market_count}] Calibracion lista; log_loss={cm['log_loss']:.6f}")
  test_eval=sample_temporally(test_use,a.max_test_rows,20261310+market_number)
  ytst=pd.to_numeric(test_eval[target],errors='coerce'); vt=ytst.isin([0,1]); test_x=X(test_eval.loc[vt],features)
  raw=model.predict_proba(test_x)[:,1]; pt=apply_platt(platt,raw); tm=metrics(ytst.loc[vt],pt)
  rows.append({'market':market,'period':'test','candidate':best,**tm})
  calibration_frames.append(calibration_bands(market,'test',ytst.loc[vt],pt))
  segment_source=test_eval.loc[vt].copy(); segment_source['_target']=ytst.loc[vt]; segment_source['_probability']=pt
  for dimension in ('competition_type','surface','tourney_level','best_of','cold_start_any_player','stats_reliable_both','surface_stats_reliable_both'):
   if dimension not in segment_source: continue
   for value,group in segment_source.groupby(dimension,dropna=False):
    if len(group)<a.min_segment_rows: continue
    segment_rows.append({'market':market,'segment':dimension,'segment_value':str(value),**metrics(group['_target'],group['_probability'])})
  predictions.loc[test_eval.loc[vt].index,f'target_{market}']=ytst.loc[vt]
  predictions.loc[test_eval.loc[vt].index,f'probability_{market}']=pt
  # Symmetry audit. First-set probabilities must complement after swapping.
  if mode=='first_set' and a.symmetry_rows!=0:
   symmetry_sample=test_eval.loc[vt].head(a.symmetry_rows) if a.symmetry_rows>0 else test_eval.loc[vt]
   original_probability=apply_platt(platt,model.predict_proba(X(symmetry_sample,features))[:,1])
   swapped_probability=apply_platt(platt,model.predict_proba(swap(symmetry_sample,features))[:,1])
   error=np.abs(original_probability-(1.0-swapped_probability))
   symmetry_rows.append({'market':market,'rows':int(len(error)),'mean_absolute_error':float(error.mean()),'p95_absolute_error':float(np.quantile(error,.95)),'maximum_absolute_error':float(error.max())})
  bundle['models'][market]=model; bundle['calibrators'][market]=platt; bundle.setdefault('selected_algorithms',{})[market]=best
  del test_x,raw,pt,test_eval; gc.collect()
  average=(time.perf_counter()-total_started)/market_number; eta=average*(market_count-market_number)
  log(f"[{market_number}/{market_count}] {market} terminado en {elapsed(market_started)}; test_log_loss={tm['log_loss']:.6f}; ETA {eta/60:.1f} min")
  pd.DataFrame(rows).to_csv(out/'market_model_metrics.partial.csv',index=False)
 over_cols=[f'probability_over_{x:.1f}' for x in OVER_LINES_BO3]; available=[c for c in over_cols if c in predictions]
 if available:
  raw=predictions[available].to_numpy(dtype=np.float32); corrected=np.minimum.accumulate(raw,axis=1); predictions[available]=corrected
  pd.DataFrame({'line':OVER_LINES_BO3[:len(available)],'violations_before':[0]+[int(np.nansum(raw[:,i]>raw[:,i-1])) for i in range(1,len(available))],'violations_after':[0]+[int(np.nansum(corrected[:,i]>corrected[:,i-1])) for i in range(1,len(available))]}).to_csv(out/'market_over_monotonicity.csv',index=False)
 predictions.to_parquet(out/'tennis_market_direct_validation.parquet'); pd.DataFrame(rows).to_csv(out/'market_model_metrics.csv',index=False)
 pd.concat(calibration_frames,ignore_index=True).to_csv(out/'market_calibration_bands.csv',index=False)
 pd.DataFrame(segment_rows).to_csv(out/'market_segment_metrics.csv',index=False)
 pd.DataFrame(symmetry_rows).to_csv(out/'market_symmetry_validation.csv',index=False)
 partial=out/'market_model_metrics.partial.csv'; partial.unlink(missing_ok=True)
 log('Guardando bundle de modelos...'); joblib.dump(bundle,out/'tennis_market_direct_models.joblib',compress=3)
 atomic_json({'schema_version':2,'dataset':str(Path(a.dataset).resolve()),'features':features,'targets':TARGETS,
  'selected_algorithms':bundle['selected_algorithms'],'test_used_for_selection':False,'training_caps':bundle['training_caps'],'probability_clip_for_metrics':[0.0001,0.9999],
  'files':{'models':'tennis_market_direct_models.joblib','metrics':'market_model_metrics.csv','validation':'tennis_market_direct_validation.parquet','monotonicity':'market_over_monotonicity.csv','calibration_bands':'market_calibration_bands.csv','segment_metrics':'market_segment_metrics.csv','symmetry':'market_symmetry_validation.csv'}},out/'tennis_market_direct_metadata.json')
 reloaded=joblib.load(out/'tennis_market_direct_models.joblib'); assert reloaded['features']==features
 log(f"[OK] Finalizado en {elapsed(total_started)}. Artefactos: {out}")
 print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__': main()
