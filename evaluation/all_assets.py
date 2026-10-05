"""
PURPOSE:  test_all_assets_v4: test a model on every asset separately plus aggregated metrics and Wilson-CI tables.
TAGS:     test_all_assets_v4, test_all_assets, per_asset_ci_table, summarize_verification, aggregated metrics, test_dict
PITFALLS: Tests exclude this module on the general load and exec it explicitly (load_into(ns, exclude=...)). Executed into the one shared evaluation namespace by evaluation/_loader.py (never imported on its own): names from other modules resolve at call time.

## 8️⃣ اختبار كل الأصول (Assets) دفعة واحدة — test_all_assets_v4
"""
# ═══════════════════════════════════════════════════════════════════════════
# اختبار كل الأصول (Assets) دفعة واحدة — نسخة v4.2: مخرجات مضغوطة ومرنة الأهداف
# ═══════════════════════════════════════════════════════════════════════════
#
# ما تغيّر في العرض:
#   • ملخص التحقق من فك التشفير يظهر **مرة واحدة** (جدول واحد لكل الأهداف عبر كل الأصول)
#     بدل تكراره لكل أصل.
#   • لكل أصل: جدول واحد بآخر `n_display` عينة لكل هدف + سطر أداء واحد لكل هدف.
#   • في النهاية: جدول ملخص لكل الأصول + جدول إجمالي لكل هدف.
#   • الأهداف تُمرَّر كنصوص أو TargetSpec؛ والأهداف غير المطلوبة تُتجاهل.
# ═══════════════════════════════════════════════════════════════════════════

def summarize_verification(verification_df: pd.DataFrame) -> pd.DataFrame:
    """يجمع تقارير verify_decoding لكل الأصول في صف واحد لكل هدف."""
    if verification_df is None or verification_df.empty:
        return pd.DataFrame()

    def _col(g, name):
        return pd.to_numeric(g[name], errors='coerce') if name in g else pd.Series(np.nan, index=g.index)

    rows = []
    for tgt, g in verification_df.groupby('target', sort=False):
        n = _col(g, 'n_samples') if 'n_samples' in g else pd.Series(1.0, index=g.index)

        def wmean(name):
            v = _col(g, name)
            m = v.notna() & n.notna() & (n > 0)
            return float(np.average(v[m], weights=n[m])) if m.any() else float('nan')

        n_failed = int((g['status'] == 'FAILED').sum()) if 'status' in g else 0
        row = {
            'target': tgt,
            'kind': g['kind'].iloc[0] if 'kind' in g else '',
            'assets': len(g),
            'status': '✅ ok' if n_failed == 0 else f'❌ {n_failed} فشل',
        }
        if row['kind'] == 'continuous':
            row.update({
                'roundtrip_max_diff': _nanmax(_col(g, 'roundtrip_max_diff')),
                'true_consistency_max_diff': _nanmax(_col(g, 'true_value_consistency_max_diff')),
                'space_ratio_mean': _nanmean(_col(g, 'space_consistency_ratio_mean')),
                'mae_scaled': wmean('mae_scaled'),
                'mae_real': wmean('mae_real'),
            })
        else:
            row.update({
                'roundtrip_max_diff': float('nan'), 'true_consistency_max_diff': float('nan'),
                'space_ratio_mean': float('nan'),
                'mae_scaled': float('nan'), 'mae_real': float('nan'),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def print_verification_summary(verification_df: pd.DataFrame):
    """يطبع ملخص التحقق من فك التشفير مرة واحدة."""
    summary = summarize_verification(verification_df)
    if summary.empty:
        return summary
    print(f"\n{'=' * 100}")
    print("🔍 التحقق من فك التشفير — مرة واحدة لكل الأصول (round-trip / اتساق الفضاءين المطبّع والأصلي)")
    print("=" * 100)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    n_failed = int((verification_df['status'] == 'FAILED').sum()) if 'status' in verification_df else 0
    print("✅ جميع الأهداف والأصول اجتازت التحقق" if n_failed == 0 else f"❌ {n_failed} حالة فشل — راجع verification_summary")
    print("   (mae_scaled / mae_real: متوسط مرجَّح بعدد العينات؛ mae_real يُقارَن فقط إن كانت أسعار الأصول بمقياس متقارب "
          "— تفصيله لكل أصل في جدول الملخص أدناه)")
    return summary


def _overall_metrics_table(all_results: List[Dict], specs: List['TargetSpec']) -> pd.DataFrame:
    """مقاييس مجمّعة (بالعينات) لكل هدف عبر كل الأصول."""
    rows = []
    for s in specs:
        if s.kind == 'continuous':
            errs, pcts, dirs = [], [], []
            for res in all_results:
                r = res.get(s.name)
                if not isinstance(r, dict) or 'error_abs' not in r:
                    continue
                v = np.asarray(r.get('has_truth', np.ones(len(r['error_abs']), dtype=bool)), dtype=bool)
                errs.append(np.asarray(r['error_abs'])[v])
                pcts.append(np.asarray(r['error_pct'])[v])
                dirs.append(np.asarray(r['direction_correct'])[v])
            if errs:
                e, p, d = np.concatenate(errs), np.concatenate(pcts), np.concatenate(dirs)
                _lo, _hi = wilson_ci(d.sum(), len(d))
                rows.append({'target': s.name, 'n': int(len(e)), 'MAE': _nanmean(e),
                             'MAPE_%': _nanmean(p), 'Win_%': float(d.mean() * 100) if len(d) else float('nan'),
                             'Win_CI95_lo_%': _lo * 100, 'Win_CI95_hi_%': _hi * 100})
        else:
            accs = [(r['accuracy'], len(r['correct'])) for r in (res.get(s.name) for res in all_results)
                    if isinstance(r, dict) and 'accuracy' in r]
            if accs:
                tot = sum(n for _, n in accs)
                _acc = float(sum(a * n for a, n in accs) / tot)
                _lo, _hi = wilson_ci(round(_acc / 100 * tot), tot)
                rows.append({'target': s.name, 'n': int(tot), 'Acc_%': _acc,
                             'Acc_CI95_lo_%': _lo * 100, 'Acc_CI95_hi_%': _hi * 100})
    return pd.DataFrame(rows)


def per_asset_ci_table(all_results: List[Dict], specs: List['TargetSpec']) -> pd.DataFrame:
    """جدول لكل (عملة × هدف): n، الدقة/نسبة صحة الاتجاه %، وفاصل Wilson 95% (بنسبة مئوية). n صغير ← فاصل عريض (ضجيج)."""
    rows = []
    for res in all_results:
        for s in specs:
            r = res.get(s.name)
            if not isinstance(r, dict):
                continue
            if s.kind == 'continuous' and 'direction_correct' in r:
                v = np.asarray(r.get('has_truth', np.ones(len(r['direction_correct']), dtype=bool)), dtype=bool)
                d = np.asarray(r['direction_correct'])[v]
            elif s.kind == 'categorical' and 'correct' in r:
                d = np.asarray(r['correct'])
            else:
                continue
            lo, hi = wilson_ci(d.sum(), len(d))
            rows.append({'asset': res.get('asset'), 'target': s.name, 'n': int(len(d)),
                         'acc_%': float(d.mean() * 100) if len(d) else float('nan'),
                         'ci95_lo_%': lo * 100, 'ci95_hi_%': hi * 100})
    return pd.DataFrame(rows)


def test_all_assets_v4(
    model,
    test_dict: Dict,
    timeframes: List[str],
    target_specs,
    verbose: bool = True,
    n_display: int = 1,
    batch_size: int = 256,
    range_frac: float = 0.2,
    min_samples_per_asset: int = 20,
    run_verification: bool = True,
    timestamp_col: Optional[int] = None,
    timestamp_key: Optional[str] = None,
    eval_all_rows: bool = True,
) -> Dict:
    """
    اختبار النموذج على كل أصل/عملة على حدة، بدعم أهداف مستمرة وفئوية معاً.

    target_specs: TargetSpec أو نصوص ('close' / 'high,low') أو None/'all'. الأهداف التي
                  يُخرجها النموذج ولم تُطلب تُتجاهل بالكامل.
    n_display:    عدد **آخر** العينات المعروضة لكل هدف في كل أصل (0 = بلا جدول عينات) — عرض فقط: كل المقاييس المجمّعة
                  تُحسب على كل صفوف الاختبار.
    eval_all_rows: True (الافتراضي) = تُقيَّم **كل** العملات وكل صفوفها (لا حدّ أدنى لعدد صفوف العملة). False = السلوك
                  القديم: تُتخطّى العملات التي صفوفها < min_samples_per_asset (20) — كانت تُسقط صامتاً العملات القصيرة
                  فيبدو الاختبار أصغر ممّا هو. يُطبع ملخص تغطية (صفوف الاختبار مقابل المُقيَّمة) دائماً حين verbose.
    timestamp_key / timestamp_col: مصدر التاريخ (مفتاح داخل بيانات الأصل، أو رقم عمود
                  في last_candles). إن لم يُحدَّد يُبحث عن مفاتيح شائعة (timestamps/times/dates...).

    Returns:
        dict: {'per_asset_results', 'aggregated_results', 'verification_summary',
               'asset_summary', 'target_specs'}
    """
    all_results: List[Dict] = []
    verification_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    specs: Optional[List['TargetSpec']] = None
    n_assets = len(test_dict)
    min_eff = 1 if eval_all_rows else int(min_samples_per_asset)
    skipped_assets: List = []
    total_rows = int(sum(len(v['base_params']) for v in test_dict.values()))

    if verbose:
        print(f"\n{'=' * 100}")
        print(f"🧪 اختبار {n_assets} عملة (v4.2) — يُعرض آخر {n_display} عينة لكل هدف")
        print("=" * 100)

    for idx, (asset_name, test_data) in enumerate(test_dict.items(), 1):
        n_samples = len(test_data['base_params'])
        if n_samples < min_eff:
            skipped_assets.append((asset_name, n_samples))
            continue

        X_inputs = tuple([test_data[f'X_{tf_name}'] for tf_name in timeframes])
        if specs is None:
            specs = _resolve_for_model(target_specs, model, X_inputs)
            if verbose:
                print(f"🎯 الأهداف المُقيَّمة: {[s.name for s in specs]}")

        base_params = test_data['base_params']
        last_candles = test_data.get('last_candles')
        y_true = {f'y_{s.name}': test_data['y'][s.name] for s in specs if s.name in test_data['y']}

        try:
            ts = _align_timestamps(_find_timestamps(test_data, timestamp_key, timestamp_col),
                                   n_samples, n_samples)

            asset_results = predict_with_evaluation_v4(
                model=model,
                X_inputs=X_inputs,
                target_specs=specs,
                base_params=base_params,
                last_candles=last_candles,
                y_true=y_true,
                verbose=False,
                n_display=0,
                batch_size=batch_size,
                direction_tolerance=range_frac,
                run_verification=run_verification,
            )

            verification = asset_results.pop('_verification', None)
            if verification:
                for target_name, rep in verification.items():
                    verification_rows.append({'asset': asset_name, 'n_samples': n_samples, **rep})

            if verbose:
                print(f"\n{'─' * 100}")
                print(f"[{idx}/{n_assets}] 🪙 {asset_name}  ({n_samples:,} عينة)")
                _print_v4_report(asset_results, specs, n_samples, n_display, bool(y_true),
                                 timestamps=ts, asset=None, legend=False)

            row = {'asset': asset_name, 'n': n_samples}
            for s in specs:
                r = asset_results.get(s.name)
                if not isinstance(r, dict):
                    continue
                if s.kind == 'continuous' and 'error_abs' in r:
                    row[f'{s.name}_MAE'] = _nanmean(r['error_abs'])
                    row[f'{s.name}_win%'] = r.get('win_rate', float('nan'))
                elif s.kind == 'categorical' and 'accuracy' in r:
                    row[f'{s.name}_acc%'] = r['accuracy']
            summary_rows.append(row)

            asset_results['asset'] = asset_name
            asset_results['n_samples'] = n_samples
            all_results.append(asset_results)

        except Exception as e:
            print(f"   ❌ خطأ في {asset_name}: {e}")
            import traceback
            if verbose:
                traceback.print_exc()
            continue

    if specs is None:
        specs = resolve_targets(target_specs)

    if not all_results:
        print("\n⚠️ لم يتم اختبار أي عملة بنجاح")
        return {'per_asset_results': pd.DataFrame(), 'aggregated_results': {},
                'verification_summary': pd.DataFrame(), 'asset_summary': pd.DataFrame(),
                'asset_ci': pd.DataFrame(), 'coverage': {'test_rows': total_rows, 'evaluated_rows': 0,
                                                         'skipped_assets': skipped_assets},
                'target_specs': specs}

    results_df = pd.DataFrame(all_results)
    aggregated = calculate_aggregated_metrics_v4(results_df, specs)
    _rows_per_asset = results_df['n_samples'].to_numpy()
    coverage = {
        'test_rows': total_rows, 'evaluated_rows': int(_rows_per_asset.sum()),
        'test_assets': n_assets, 'evaluated_assets': int(len(results_df)),
        'skipped_assets': skipped_assets, 'eval_all_rows': bool(eval_all_rows),
        'rows_per_asset_min': int(_rows_per_asset.min()), 'rows_per_asset_median': float(np.median(_rows_per_asset)),
        'rows_per_asset_max': int(_rows_per_asset.max()),
    }
    asset_ci = per_asset_ci_table(all_results, specs)
    verification_df = pd.DataFrame(verification_rows) if verification_rows else pd.DataFrame()
    asset_summary = pd.DataFrame(summary_rows)

    if verbose:
        if run_verification and not verification_df.empty:
            print_verification_summary(verification_df)

        print(f"\n{'=' * 100}")
        print(f"📐 التغطية: {coverage['evaluated_rows']:,} من {coverage['test_rows']:,} صف اختبار | "
              f"{coverage['evaluated_assets']} من {coverage['test_assets']} عملة | صفوف/عملة: "
              f"min={coverage['rows_per_asset_min']}، وسيط={coverage['rows_per_asset_median']:.0f}، max={coverage['rows_per_asset_max']}"
              f"{'' if eval_all_rows else f' (eval_all_rows=False: تُتخطّى العملات < {min_samples_per_asset} صف)'}")
        if skipped_assets:
            print(f"   ⚠️ تُخطّيت {len(skipped_assets)} عملة بصفوف قليلة: {skipped_assets[:8]}{' …' if len(skipped_assets) > 8 else ''}")
        if coverage['rows_per_asset_median'] < 50:
            print("   ℹ️ وسيط صفوف العملة صغير: حجم قسم الاختبار نفسه (train_pct/val_pct + فجوة العزل) — لا سقف في chicks؛ "
                  "الدقة لكل عملة ضجيج (انظر فاصل Wilson) والمعتمد هو المجمَّع.")
        if len(asset_summary) > 1:
            print(f"\n{'=' * 100}")
            print("📋 ملخص الأداء لكل أصل (MAE بوحدة سعر الأصل، win% = صحة الاتجاه)")
            print("=" * 100)
            save_or_print(asset_summary.round(4), 'per_asset_summary')
        if not asset_ci.empty:
            print(f"\n{'=' * 100}")
            print("📋 دقة كل عملة × هدف مع n وفاصل Wilson 95% (الجدول كاملاً في analysis_outputs/ إن كبر)")
            print("=" * 100)
            save_or_print(asset_ci.round(2), 'per_asset_ci')

        overall = _overall_metrics_table(all_results, specs)
        if not overall.empty:
            print(f"\n{'=' * 100}")
            print("🧮 الإجمالي لكل هدف عبر كل الأصول (مجمّع بالعينات)")
            print("=" * 100)
            print(overall.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return {
        'per_asset_results': results_df,
        'aggregated_results': aggregated,
        'verification_summary': verification_df,
        'asset_summary': asset_summary,
        'asset_ci': asset_ci,
        'coverage': coverage,
        'target_specs': specs,
    }


def calculate_aggregated_metrics_v4(results_df: pd.DataFrame, target_specs) -> Dict:
    """حساب metrics مجمّعة عبر كل الأصول، لكل هدف بحسب نوعه (آمنة مع NaN)."""
    target_specs = resolve_targets(target_specs)
    aggregated = {
        'total_assets': len(results_df),
        'total_samples': int(results_df['n_samples'].sum()),
        'avg_samples_per_asset': float(results_df['n_samples'].mean()),
    }

    weights = (results_df['n_samples'] / results_df['n_samples'].sum()).to_numpy()

    for spec in target_specs:
        maes, accs, w_used = [], [], []
        for pos, (_, row) in enumerate(results_df.iterrows()):
            r = row[spec.name]
            if spec.kind == 'continuous' and 'error_abs' in r:
                maes.append(_nanmean(r['error_abs']))
                w_used.append(weights[pos])
            elif spec.kind == 'categorical' and 'accuracy' in r:
                accs.append(r['accuracy'])

        if maes:
            m = np.asarray(maes, dtype=np.float64)
            w = np.asarray(w_used, dtype=np.float64)
            ok = np.isfinite(m)
            if ok.any():
                aggregated[f'{spec.name}_weighted_mae'] = float(np.average(m[ok], weights=w[ok]))
                aggregated[f'{spec.name}_mean_mae'] = float(m[ok].mean())
        if accs:
            aggregated[f'{spec.name}_mean_accuracy'] = float(np.mean(accs))

    return aggregated


# دالة قديمة تبقى للتوافق الخلفي (تقبل الآن نصوصاً أو TargetSpec)
def test_all_assets(model, test_dict, timeframes, targets, verbose=True, n_display=1,
                     batch_size=256, range_frac=0.2):
    """(نسخة سابقة - محفوظة للتوافق)"""
    specs = resolve_targets(targets)
    result = test_all_assets_v4(model, test_dict, timeframes, specs, verbose, n_display,
                                 batch_size, range_frac, run_verification=False)
    result['detailed_predictions'] = None
    return result
