import os
import pickle, gzip, numpy as np, pandas as pd
def load():
    d = pickle.load(gzip.open(os.environ.get('H1_DATA', 'preprocessing_output_1h_h32_s32_h4_latest.pkl.gz'), 'rb'))
    lc = d['last_candles']; bp = d['base_params']; X = d['X_1h']
    names = np.empty(len(lc), object)
    for b in d['asset_bounds']: names[b['start']:b['end']] = b['name']
    df = pd.DataFrame({'asset': names, 'timestamp': lc[:, 3], 'entry': lc[:, 2], 'last_high': lc[:, 0], 'last_low': lc[:, 1],
                       'fut_close': lc[:, 4], 'fut_high': lc[:, 6], 'fut_low': lc[:, 5]})
    ci = d['feature_order'].index('close')
    closes = X[:, :, ci].astype('float64') * bp[:, 1:2] + bp[:, 0:1]      # exact reconstruction (audit: err 2e-8)
    lr = np.diff(np.log(np.maximum(closes, 1e-12)), axis=1)
    df['pvol'] = lr.std(1)                                                 # realized vol inside the window (known at entry)
    return d, df, closes
