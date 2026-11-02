# Bundle summary: gpu-diag-001

- status: **OK**
- repo_ref: `d5b2e16496766db2ed1a7796848096ecf36c5b49`
- entry: `tools/gpu_diag.py `
- note: Why does TensorFlow see no GPU on a T4 runtime (profile-002)? Same pip list as profile-002, then a clean tensorflow[and-cuda] reinstall. Modifies this disposable runtime's Python environment.
- exit code: 0   duration: 212.1 s
- problems: none

## manifest.json
```json
{
 "config": {
  "tf_version": "2.21.0",
  "try_fix": true
 },
 "git_sha": "d5b2e16496766db2ed1a7796848096ecf36c5b49",
 "input_fingerprint": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
 "n_configurations": 1,
 "outputs": {
  "gpu_diag.json": "5cbf18d7ba10ac37c552cc66ef593b67c437a5c8057e8a03b5788f86f18566ed"
 },
 "row_counts": {
  "gpu_diag.json": 1
 },
 "runtime_seconds": 138.638
}
```
