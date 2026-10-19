# Bundle summary: trim-tails-001-dry

- status: **OK**
- repo_ref: `85453cd68db7706b8325901e796d3de987bb85bb`
- entry: `tools/trim_dead_tails_colab_job.py `
- note: Dry run: counts dead zero-volume tails per coin, writes nothing to Drive. Not queued.
- exit code: 0   duration: 123.2 s
- problems: none

## manifest.json
```json
{
 "config": {
  "data_dir": "history_15m",
  "dry_run": true
 },
 "git_sha": "85453cd68db7706b8325901e796d3de987bb85bb",
 "input_fingerprint": "a6d367aa71be0a14ad8b7705fa9d3475320c55253010a7c9e00aae5c26025a1d",
 "n_configurations": 1,
 "outputs": {
  "summary.json": "4bf9cf564b5efecc0589e7c426c2a9d179228cf53ea0a85c7c8e85a17372c877",
  "trimmed_coins.csv": "cee8622083dd90cafd63f4c46e8f2087fe6a8f49fe8cba3bfe8bb61bd60192c4"
 },
 "row_counts": {
  "trimmed_coins.csv": 167
 },
 "runtime_seconds": 114.096
}
```

## trimmed_coins.csv (first 12 of 168 lines)
```
coin,status,error,file,original_sha256,trimmed_sha256,n_bars_before,n_tail_bars,first_tail_ts,last_tail_ts,last_real_ts,trimmed_utc
1000BTTCUSDT,would_trim,,1000BTTCUSDT.csv.gz,,,7223,1,1649667600000,1649667600000,1649666700000,
1000WHYUSDT,would_trim,,1000WHYUSDT.csv.gz,,,64370,18588,1773738000000,1790466300000,1773737100000,
1000XUSDT,would_trim,,1000XUSDT.csv.gz,,,65518,30396,1763110800000,1790466300000,1763109900000,
42USDT,would_trim,,42USDT.csv.gz,,,32118,23004,1769763600000,1790466300000,1769762700000,
A2ZUSDT,would_trim,,A2ZUSDT.csv.gz,,,40672,17915,1774343700000,1790466300000,1774342800000,
ACXUSDT,would_trim,,ACXUSDT.csv.gz,,,63300,4860,1786093200000,1790466300000,1786092300000,
AGIXUSDT,would_trim,,AGIXUSDT.csv.gz,,,126567,79067,1719306900000,1790466300000,1719306000000,
AI16ZUSDT,would_trim,,AI16ZUSDT.csv.gz,,,60713,31163,1762420500000,1790466300000,1762419600000,
AIUSDT,would_trim,,AIUSDT.csv.gz,,,95278,14551,1777371300000,1790466300000,1777370400000,
AKROUSDT,would_trim,,AKROUSDT.csv.gz,,,47338,10,1653634800000,1653642900000,1653633900000,
ALPACAUSDT,would_trim,,ALPACAUSDT.csv.gz,,,73494,49403,1746004500000,1790466300000,1746003600000,
```
