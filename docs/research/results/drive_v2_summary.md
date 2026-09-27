# نتائج خطّة `drive_v2` على بيانات Drive (Colab T4، 2026-09-27)

ناتج `anti_memorization_benchmark.py --plan drive_v2 --summary --parallel 4` كما طُبع (commit الكود: `f9dd878`).
عاملان من أربعة (w1، w4) قتلهما النظام (‎-9، نفاد RAM: ~3.4GB لكل عامل × 4 > 12.7GB) فلم تكتمل
`full_xrank_RL_gru_s0` و`full_RL_gru_s2` و`full_xrank_shuf_RL_gru_s0`. التوازي لم يُسرّع (المعالج نواتان):
~900 ثانية لتجربة little مقابل ~210 تسلسلياً في `drive_v1`.

### AM-SUMMARY plan=drive_v2 | 222 assets, 2019-09-09 → 2026-09-18, GPU=1 | parallel=4 | RAM peak 6.5GB total, 3.4GB/worker | 3528s
| run | best@ | test mag best/last | test dir best/last | train mag best/last | train dir best/last | sec |
|---|---|---|---|---|---|---|
| little_RL_gru_emaw1_s0 | 60 | 0.587/0.586 | 0.517/0.517 | 0.721/0.722 | 0.621/0.622 | 989 |
| little_RL_tiny_emaw1_s0 | 60 | 0.611/0.611 | 0.517/0.516 | 0.703/0.704 | 0.583/0.586 | 928 |
| little_RL_gru_raw_s0 | 60 | 0.586/0.586 | 0.517/0.517 | 0.722/0.722 | 0.622/0.622 | 1089 |
| little_shuf_RL_gru_emaw1_s0 | 59 | 0.495/0.496 | 0.499/0.499 | 0.605/0.608 | 0.600/0.602 | 1048 |
| full_RL_gru_emaw1_s0 | 27 | 0.645/0.645 | 0.555/0.554 | 0.659/0.658 | 0.568/0.570 | 1659 |
| full_RL_gru_s1 | 23 | 0.645/0.645 | 0.553/0.553 | 0.655/0.658 | 0.562/0.567 | 1156 |
| little_RL_tiny_raw_s0 | 60 | 0.611/0.611 | 0.516/0.516 | 0.704/0.704 | 0.586/0.586 | 892 |
| full_shuf_RL_gru_emaw1_s0 | 26 | 0.494/0.486 | 0.508/0.511 | 0.515/0.508 | 0.516/0.518 | 1474 |
| full (linear) | - | 0.609 | 0.538 | 0.597 | 0.543 | - |
| full_shuf (linear) | - | 0.477 | 0.511 | 0.507 | 0.512 | - |
| full_xrank (linear) | - | 0.648 | 0.552 | 0.643 | 0.556 | - |
| full_xrank_shuf (linear) | - | 0.472 | 0.512 | 0.510 | 0.513 | - |
| little (linear) | - | 0.601 | 0.524 | 0.670 | 0.550 | - |
| little_shuf (linear) | - | 0.489 | 0.496 | 0.554 | 0.528 | - |
