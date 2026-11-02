# Bundle summary: profile-004

- status: **OK**
- repo_ref: `d4cc437002547c2b6cf7ba5c1b37081370008c40`
- entry: `tools/profile_run.py `
- note: S03 run profiler on the T4 with the NEW default targets (high+low+close, TargetSettings.suspended_targets=()), batch 512. profile-003 measured the old high/low-only model. Installs tensorflow[and-cuda] first (gpu-diag-001). repo_ref moved to include the stdout_tail of failed reports (profile-003: ma
- exit code: 0   duration: 2809.4 s
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
 "git_sha": "d4cc437002547c2b6cf7ba5c1b37081370008c40",
 "input_fingerprint": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
 "n_configurations": 1,
 "outputs": {
  "profile.json": "1afbf87b56ff16b325f7126ed950989f8de73f25dff2f0c1919dac0dcaaf73ff",
  "profile.md": "cc3228f7ebd2f06fb4f0c751e88e50254b6da2e8594c0ff799f8081c032b90d0"
 },
 "row_counts": {
  "profile.json": 20
 },
 "runtime_seconds": 2714.632
}
```
