# Bundle summary: profile-005

- status: **OK**
- repo_ref: `140d021308121101c799202e8e9198dddc7b2a95`
- entry: `tools/profile_run.py `
- note: S03 run profiler AFTER the fixes (profile-004 was the baseline @ d4cc437): prediction batch 2048, pattern-discovery forest capped (rf_max_samples 50000), output tails of failed reports. Same config as profile-004: high+low+close, training batch 512, 2 epochs, T4 via install_cuda_tensorflow. repo_ref
- exit code: 0   duration: 2114.7 s
- problems: none

## manifest.json
```json
{
 "config": {
  "input_batches": 200,
  "install_cuda_tensorflow": "2.21.0",
  "reports": [
   "gap",
   "verification",
   "market_neutral",
   "candle",
   "signals",
   "chicks"
  ],
  "require_gpu": true,
  "settings": {
   "train": {
    "batch_size": 512,
    "epochs": 2,
    "fit_verbose": 2
   }
  }
 },
 "git_sha": "140d021308121101c799202e8e9198dddc7b2a95",
 "input_fingerprint": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
 "n_configurations": 1,
 "outputs": {
  "profile.json": "e340ac4df7fafc4edcab1aaab206a267918c3229f45ed81c2feb8005fb35914f",
  "profile.md": "03ef848592727490bc90152e45c86dfe9e4dc440e89ad9a647324eecff1ec887"
 },
 "row_counts": {
  "profile.json": 21
 },
 "runtime_seconds": 2032.431
}
```
