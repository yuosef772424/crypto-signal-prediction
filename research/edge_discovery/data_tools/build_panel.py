import pandas as pd, numpy as np, glob, os
fr={}
for f in sorted(glob.glob('metrics/*.csv.gz')):
    s=os.path.basename(f).replace('.csv.gz','')
    try: d=pd.read_csv(f)
    except Exception as e: print('skip',s,e); continue
    if len(d)<500: print('short',s,len(d)); continue
    d['ts']=pd.to_datetime(d.timestamp,utc=True).dt.floor('h')
    d=d.drop_duplicates('ts').set_index('ts').sort_index()
    d['px']=d.sum_open_interest_value/d.sum_open_interest
    fr[s]=d
cols={'px':'px','oi':'sum_open_interest','oiv':'sum_open_interest_value','tt_acc':'count_toptrader_long_short_ratio','tt_pos':'sum_toptrader_long_short_ratio','ls_acc':'count_long_short_ratio','taker':'sum_taker_long_short_vol_ratio'}
idx=pd.date_range(pd.Timestamp('2024-01-01',tz='UTC'),max(d.index.max() for d in fr.values()),freq='h')
P={k:pd.DataFrame({s:d[v].reindex(idx) for s,d in fr.items()}) for k,v in cols.items()}
for k,v in P.items(): v.to_parquet(f'panel_{k}.parquet')
px=P['px']; print('coins',px.shape, 'missing frac',px.isna().mean().mean().round(3))
# sanity: hourly abs return distribution, detect bad prints
r=np.log(px).diff(); print('abs hourly ret quantiles', r.abs().stack().quantile([.5,.99,.9999]).round(4).to_dict())
print('max jumps', r.abs().max().sort_values().tail(5).round(3).to_dict())
print(px['BTCUSDT'].dropna().iloc[[0,-1]])
