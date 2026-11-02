# Bundle summary: profile-003

- status: **OK**
- repo_ref: `18ca6e3fc58105fb251b556740288460133ffc56`
- entry: `tools/profile_run.py `
- note: S03 run profiler, batch 512, on the T4: the entry replaces tensorflow + tensorflow-cpu by tensorflow[and-cuda]==2.21.0 first (gpu-diag-001 showed the CPU build wins otherwise; ~2 min) and re-execs. Modifies this runtime's Python environment.
- exit code: 0   duration: 2045.6 s
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
 "git_sha": "18ca6e3fc58105fb251b556740288460133ffc56",
 "input_fingerprint": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
 "n_configurations": 1,
 "outputs": {
  "profile.json": "6978cf1549f2bd4bf10ee6c5876c2890b7ebadefc8611fc1e2fc7b95e46e4d15",
  "profile.md": "a0344779bce12f3b0ff7cf156d3ff71c1d0a51def419476c773c04410c584ce8"
 },
 "row_counts": {
  "profile.json": 20
 },
 "runtime_seconds": 1900.501
}
```
