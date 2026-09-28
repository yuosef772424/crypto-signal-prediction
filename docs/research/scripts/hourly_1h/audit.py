import pickle, gzip, numpy as np, pandas as pd
import os
d=pickle.load(gzip.open(os.environ.get('H1_DATA', 'preprocessing_output_1h_h32_s32_h4_latest.pkl.gz'),'rb'))
X=d['X_1h']; lc=d['last_candles']; bp=d['base_params']
LCN=['last_high','last_low','last_close','timestamp','future_close','future_low_min','future_high_max']
L=pd.DataFrame(lc,columns=LCN)
L['ts']=pd.to_datetime(L.timestamp.astype('int64'),utc=True)
names=np.empty(len(L),object)
for b in d['asset_bounds']: names[b['start']:b['end']]=b['name']
L['asset']=names
print('n assets',len(d['asset_bounds']), 'unassigned',pd.isna(names).sum())
print('keys other:',{k:d[k] for k in d if k not in ('X_1h','last_candles','base_params','asset_bounds') and not k.startswith('y_')})
print('time range',L.ts.min(),L.ts.max())
print('minute/second of ts:',L.ts.dt.minute.value_counts().to_dict(), 'hour dist unique',L.ts.dt.hour.nunique())
# per coin spacing
L['dt']=L.groupby('asset').ts.diff().dt.total_seconds()/3600
print('spacing hours value counts (top):'); print(L.dt.value_counts().head(8))
print('non-monotone within coin:', (L.dt<=0).sum())
# alignment
g=L.groupby('timestamp').size()
print('unique timestamps',len(g)); print('group size describe'); print(g.describe())
print('share of samples in groups of size: ', pd.cut(L.timestamp.map(g),[0,1,4,9,19,39,59,83,1000]).value_counts(normalize=True).sort_index().round(3).to_dict())
L['hmod32']=((L.timestamp.astype('int64')//3600e9).astype('int64'))%32
print('phase (hour index mod 32) distribution top:',L.hmod32.value_counts().head(10).to_dict(), 'n phases',L.hmod32.nunique())
# per coin phase
ph=L.groupby('asset').hmod32.nunique(); print('phases per coin value counts',ph.value_counts().to_dict())
# phase for coins
print('coins per phase', L.groupby('hmod32').asset.nunique().to_dict())
# duplicates
print('dup (asset,ts):',L.duplicated(['asset','timestamp']).sum())
# NaN
print('NaN X',np.isnan(X).sum(),'inf',np.isinf(X).sum(),'NaN lc',np.isnan(lc).sum(),'NaN bp',np.isnan(bp).sum())
for k in [k for k in d if k.startswith('y_')]:
    y=d[k]; print(k,'nan',np.isnan(y).sum(),'uniq' if 'class' in k else 'desc', np.unique(y) if 'class' in k else (y.min(),y.max(),y.mean()))
# labels consistency
for t,fc in [('high','future_high_max'),('low','future_low_min'),('close','future_close')]:
    r=L[fc]/L['last_'+t]-1
    c=(r>0).astype(float)
    print(t,'class match',(c.values==d[f'y_{t}_class']).mean(),'up share',d[f'y_{t}_class'].mean(),
          'reg match (maxabs diff)',np.nanmax(np.abs(np.clip(r.values,-1,1)*d.get('reg_target_scale',1.0)-d[f'y_{t}_reg'])), 'ties r==0',(r==0).mean())
# sanity: high>=low etc
print('last_high>=last_close',(L.last_high>=L.last_close).mean(),'fut_high>=fut_close',(L.future_high_max>=L.future_close).mean(),'fut_low<=fut_close',(L.future_low_min<=L.future_close).mean())
# feature per last step vs price
fo=d['feature_order']; ci=fo.index('close')
den=X[:,:,ci]*bp[:,1:2]+bp[:,0:1]
print('bp0 vs window median? checking last close reconstruct:')
print('  rel err last step vs last_close median',np.nanmedian(np.abs(den[:,-1]/L.last_close.values-1)))
print('  rel err last step vs future_close median',np.nanmedian(np.abs(den[:,-1]/L.future_close.values-1)))
# next sample same coin first step vs future_close
L['i']=np.arange(len(L))
nx=L.groupby('asset').i.shift(-1)
ok=(L.dt.shift(-1)==32)&nx.notna()
ii=L.i[ok].values; jj=nx[ok].astype(int).values
print('  next-sample X[0] close vs future_close rel err median',np.nanmedian(np.abs(den[jj,0]/L.future_close.values[ii]-1)), 'n',len(ii))
print('  next-sample X[0] close vs last_close rel err median',np.nanmedian(np.abs(den[jj,0]/L.last_close.values[ii]-1)))
print('bp0 == window median of reconstructed close?', np.nanmedian(np.abs(np.median(den,1)/bp[:,0]-1)))
print('X last-step feature stats:'); print(pd.DataFrame(X[:,-1,:],columns=fo).describe().T[['mean','std','min','max']].round(3))
L.to_pickle('L.pkl')
