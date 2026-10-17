"""
PURPOSE:  build_nig_timenet_v2: the model architecture builder (+ DEFAULT_HEAD_TYPES).
TAGS:     build_nig_timenet_v2, DEFAULT_HEAD_TYPES, encoder, multi-timeframe, anti-memorization options, model builder
PITFALLS: Every anti-memorization option defaults to the OLD behaviour exactly, so older checkpoints still load.
          Executed into the one shared model namespace by model/_loader.py (never imported on its own): names from
          other modules resolve at call time.

## 8) بناء النموذج
"""
# احتياطي build_nig_timenet_v2 الداخلي فقط، لاستدعاء مباشر بلا head_types ولا
# عبر MODEL_CONFIG (الذي يحدّد فعلياً تصنيفاً+انحداراً معاً — انظر MODEL_CONFIG أدناه).
DEFAULT_HEAD_TYPES = {"high": ["nig_regression"], "low": ["nig_regression"], "close": ["nig_regression"]}


def build_nig_timenet_v2(
    seq_len, n_features,
    d_model=128, num_layers=4, num_heads=4, num_kv_heads=2,
    patch_len=4, stride=2, kernel_sizes=(3, 5, 9, 17), max_rel_pos=16,
    dropout=0.1, attn_dropout=0.0, causal=False, window=None,
    use_instance_norm=True, use_decomposition=True, use_linear_path=True,
    price_targets=("high", "low", "close"), head_types=None,
    enforce_order=True, head_hidden=128,
    nu_min=0.1, alpha_min=2.0, beta_min=0.01, unc_max=20.0,
    class_head_hidden=64, n_classes=3,
    norm_eps=1e-4, input_clip=None, stats_mode="full", input_noise_std=0.0, feature_dropout=0.0,
    linear_path_l2=1e-4, level_passthrough=False, level_norm=None,
    encoder="transformer", level_film=False, architecture="fused", fusion="concat", level_hidden=64,
    n_coins=0, coin_emb_dim=4, coin_emb_l2=1e-3,
    name="nig_timenet_v2",
):
    """
    seq_len/n_features: عددان = نافذة إطار زمني واحد (البناء القديم حرفياً: نفس أسماء الطبقات والأوزان، فتُحمَّل نقاط
    الحفظ القديمة). أو قاموسان ``{فريم: قيمة}`` (n_features قد يكون عدداً واحداً لكل الفريمات) = **متعدّد الفريمات**:
    فرع مُرمِّز لكل فريم بنفس المعمارية وبأوزان منفصلة (كل خيارات مقاومة الحفظ تسري على كل فرع)، وتُدمَج تضمينات الفروع
    المُجمَّعة (h بعد readout_fc والمسار الخطّي) بالتسلسل ``branch_concat`` ثم الجذع (trunk_norm/trunk_drop) والرؤوس كما هي.
    أسماء طبقات كل فرع تنتهي بـ ``_{فريم}`` (instance_norm_4h…)، ومُدخلاته ``Input(name=فريم)`` بترتيب القاموس، فيصلح
    التغذية بقاموس ``{فريم: X}`` أو قائمة/tuple بالترتيب نفسه (tuple(X_tf) كما تفعل أدوات التقييم). قاموس بفريم واحد = فريم واحد.

    unc_max: حدّ أعلى لـ y_*_aleatoric/y_*_epistemic (بوحدة y_*_reg) — يمنع انفجار √(β/(α−1)) لعيّنة خارج التوزيع؛ القيم
    الطبيعية (< ~0.5) لا تتغيّر. None = بلا قصّ. (NIGUncertainty: السبب والتفاصيل.)

    head_types: {هدف: [أنواع رؤوس]} — عبر build_model_fn/MODEL_CONFIG (نقطة
    الدخول الفعلية، القسم ٩) يتضمّن تصنيفاً+انحداراً معاً افتراضياً. استدعاء
    هذه الدالة مباشرة بلا head_types يقع على DEFAULT_HEAD_TYPES (انحدار NIG
    فقط) بدلاً من ذلك. لإضافة/إزالة رأس لهدف: عدّل القاموس نفسه، مثلاً:
        head_types = dict(DEFAULT_HEAD_TYPES)
        head_types['close'] = ['nig_regression', 'binary_classification']

    خيارات مقاومة الحفظ (كلها افتراضياً = السلوك القديم؛ ANTI_MEMORIZATION_CONFIG في القسم ٩ يفعّلها معاً):
      norm_eps / input_clip / stats_mode: حرّاس InstanceNorm (راجع docstring الطبقة).
      feature_dropout: يُسقط **قناة ميزة كاملة** (كل خطواتها الزمنية) لكل عيّنة أثناء التدريب
                       (SpatialDropout1D) — يمنع الاعتماد على ميزة واحدة تصلح معرِّفاً للعيّنة.
      input_noise_std: ضجيج غاوسي على z المُطبَّع أثناء التدريب فقط — كل حقبة يرى النموذج نسخة مختلفة
                       قليلاً من نفس العيّنة فلا يستطيع حفظ قيمها الدقيقة.
      linear_path_l2: عقوبة L2 على المسار الخطّي (Flatten(z) → d_model: seq_len×n_features مُدخلاً).
      level_passthrough: InstanceNorm يمحو **مستوى** كل ميزة داخل النافذة (z = شكلها فقط)، فالفرق بين
                       عملة متقلّبة وأخرى هادئة (إشارة H003 نفسها) لا يصل للمحوّل إلا عبر مسار stats الجانبي.
                       True يضيف المدخل الخام بعد SymLog كقنوات إضافية لكل رمز وللمسار الخطّي — المحوّل
                       والمسار الخطّي يريان الشكل والمستوى معاً.
      level_norm: None | "batch". فروق المستوى بين العملات صغيرة عددياً (NATR 0.03 مقابل 0.05) ولا شيء يقيّسها
                  عبر العملات: z يمحوها، ومسار stats/المستوى يمرّرها خاماً بمقياس ضئيل فتكاد لا تحرّك التدرّج.
                  "batch" يضع BatchNormalization على stats وعلى قنوات المستوى — تقييس بإحصاءات التدريب
                  المتحرّكة (في الاستدلال ثابتة، مستقلّة عن تركيب الدفعة)، كما يفعل StandardScaler للمرجع الخطّي.
    الثلاثة الأولى تعمل على z قبل التفكيك والمسار الخطّي معاً؛ لا أثر لها في الاستدلال (training=False).

    بدائل معمارية (كلها افتراضياً = القديم؛ مُقاسة في docs/research/anti_memorization_pr7.md §٧):
      encoder: "transformer" (رُقَع + كتل GQA) | "tcn" (التفافات سببية متوسّعة 1,2,4,…، بواقٍ) | "gru".
      level_film: متجه مستوى العيّنة (stats + آخر خطوة بعد SymLog، مُطبَّعان عبر الدفعة، + تضمين العملة إن وُجد)
                  يُعدِّل مخرج كل كتلة بـ FiLM: x·(1+γ)+β — المستوى يغيّر **كيف تُقرأ** النافذة لا يُضاف إليها فقط.
      architecture: "fused" (دمج مبكر، الحالي) | "two_tower": برج زمني على z وحده (بلا stats ولا قنوات مستوى)
                  وبرج مقطعي MLP على متجه المستوى، يُدمجان متأخّراً (fusion: "concat" | "gate").
      n_coins > 0: مُدخل ثانٍ coin_id (عدد صحيح) → Embedding(n_coins, coin_emb_dim) يبدأ صفراً مع L2 —
                  عملة لم تُرَ في التدريب تبقى صفراً (= «عملة متوسّطة»). ⚠️ معرِّف عملة صريح = قناة حفظ محتملة
                  بالتعريف؛ لذلك L2 وdropout عليه، ويُقاس بالتسميات المخلوطة قبل أي اعتماد.
    """
    head_types = head_types or DEFAULT_HEAD_TYPES
    head_cfg = dict(
        dropout=dropout, class_head_hidden=class_head_hidden, n_classes=n_classes,
        head_hidden=head_hidden, nu_min=nu_min, alpha_min=alpha_min, beta_min=beta_min,
    )

    if encoder not in ("transformer", "tcn", "gru"):
        raise ValueError(f"encoder: 'transformer' | 'tcn' | 'gru' — لا {encoder!r}")
    if architecture not in ("fused", "two_tower") or fusion not in ("concat", "gate"):
        raise ValueError("architecture: 'fused' | 'two_tower'، fusion: 'concat' | 'gate'")
    two_tower = architecture == "two_tower"
    # ── الفريمات: عددان = فريم واحد (البناء القديم حرفياً)؛ قاموسان {فريم: قيمة} = فرع مُرمِّز لكل فريم ──
    tfs = list(seq_len) if isinstance(seq_len, dict) else None
    if tfs is not None and len(tfs) == 1:
        seq_len = seq_len[tfs[0]]
        n_features = n_features[tfs[0]] if isinstance(n_features, dict) else n_features
        tfs = None
    if tfs is not None:
        if isinstance(n_features, dict) and set(n_features) != set(tfs):
            raise ValueError(f"seq_len وn_features بفريمات مختلفة: {tfs} ≠ {list(n_features)}")
        feats = {tf: int(n_features[tf] if isinstance(n_features, dict) else n_features) for tf in tfs}
        inputs_by_tf = {tf: layers.Input(shape=(int(seq_len[tf]), feats[tf]), name=str(tf)) for tf in tfs}
    else:
        inputs = layers.Input(shape=(seq_len, n_features), name="input_sequence")
    coin_in = layers.Input(shape=(), dtype="int32", name="coin_id") if n_coins else None

    def _make_coin_vec():
        vec = layers.Embedding(n_coins, coin_emb_dim, embeddings_initializer="zeros",
                               embeddings_regularizer=regularizers.l2(coin_emb_l2), name="coin_embedding")(coin_in)
        return layers.Dropout(dropout, name="coin_emb_drop")(vec)

    def _branch(inputs, T, sfx="", coin_vec=None):
        """مُرمِّز فريم واحد حتى تضمينه المُجمَّع h (بعد readout_fc والمسار الخطّي). sfx يُلحَق بكل اسم طبقة."""
        nm = lambda s: s + sfx  # noqa: E731

        if use_instance_norm:
            z, stats = InstanceNorm(eps=norm_eps, clip=input_clip, stats_mode=stats_mode, name=nm("instance_norm"))(inputs)
            if stats_mode == "none":
                stats = None
        else:
            z, stats = inputs, None
        if feature_dropout > 0:
            z = layers.SpatialDropout1D(feature_dropout, name=nm("feature_dropout"))(z)
        if input_noise_std > 0:
            z = layers.GaussianNoise(input_noise_std, name=nm("input_noise"))(z)

        if use_decomposition:
            trend, seasonal = CausalMultiScaleDecomp(kernel_sizes, name=nm("decomp"))(z)
            tok_in = layers.Concatenate(axis=-1, name=nm("trend_seasonal"))([trend, seasonal])
        else:
            tok_in = z
        lvl = None
        if level_passthrough and not two_tower:
            lvl = SymLog(name=nm("level_symlog"))(inputs)
            if level_norm == "batch":
                lvl = layers.BatchNormalization(axis=-1, name=nm("level_bn"))(lvl)
            if feature_dropout > 0:
                lvl = layers.SpatialDropout1D(feature_dropout, name=nm("level_feature_dropout"))(lvl)
            if input_noise_std > 0:
                lvl = layers.GaussianNoise(input_noise_std, name=nm("level_noise"))(lvl)
            tok_in = layers.Concatenate(axis=-1, name=nm("shape_and_level"))([tok_in, lvl])

        if stats is not None and level_norm == "batch":
            stats = layers.BatchNormalization(name=nm("stats_bn"))(stats)

        # ── متجه مستوى العيّنة (مقطعي): للبرج الثاني ولتكييف FiLM ──
        if n_coins and coin_vec is None:
            coin_vec = _make_coin_vec()
        level_vec = None
        if level_film or two_tower:
            last_lvl = LastToken(name=nm("last_level"))(SymLog(name=nm("level_vec_symlog"))(inputs))
            last_lvl = layers.BatchNormalization(name=nm("last_level_bn"))(last_lvl)
            lv_parts = [last_lvl] + ([stats] if stats is not None else []) + ([coin_vec] if coin_vec is not None else [])
            level_vec = layers.Concatenate(name=nm("level_vec_concat"))(lv_parts) if len(lv_parts) > 1 else lv_parts[0]
            if input_noise_std > 0:
                level_vec = layers.GaussianNoise(input_noise_std, name=nm("level_vec_noise"))(level_vec)
            level_vec = layers.Dense(level_hidden, activation="gelu", name=nm("level_embed"))(level_vec)
            level_vec = layers.Dropout(dropout, name=nm("level_embed_drop"))(level_vec)

        # ── المُرمِّز الزمني ──
        if encoder == "transformer":
            x = PatchEmbedding(d_model, patch_len, stride, name=nm("patch_embed"))(tok_in)
        else:
            x = layers.Dense(d_model, name=nm("step_proj"))(tok_in)
        n_blocks = num_layers if encoder != "tcn" else max(1, int(np.ceil(np.log2(max(T, 2)))) - 1)
        for i in range(n_blocks):
            if encoder == "transformer":
                x = TransformerBlock(d_model, num_heads, num_kv_heads, max_rel_pos, causal, window,
                                      dropout, attn_dropout, num_layers_for_init=num_layers,
                                      name=nm(f"block_{i + 1}"))(x)
            elif encoder == "tcn":   # مجال الاستقبال بعد n كتلة: 1 + 2·(2^n − 1) ≥ seq_len
                y = layers.Conv1D(d_model, 3, dilation_rate=2 ** i, padding="causal", activation="gelu",
                                  name=nm(f"tcn_conv_{i + 1}"))(RMSNorm(name=nm(f"tcn_norm_{i + 1}"))(x))
                x = layers.Add(name=nm(f"tcn_res_{i + 1}"))([x, layers.Dropout(dropout, name=nm(f"tcn_drop_{i + 1}"))(y)])
            else:
                y = layers.GRU(d_model, return_sequences=True, name=nm(f"gru_{i + 1}"))(RMSNorm(name=nm(f"gru_norm_{i + 1}"))(x))
                x = layers.Add(name=nm(f"gru_res_{i + 1}"))([x, layers.Dropout(dropout, name=nm(f"gru_drop_{i + 1}"))(y)])
            if level_film:
                x = FiLM(d_model, name=nm(f"film_{i + 1}"))([x, level_vec])
        x = RMSNorm(name=nm("final_norm"))(x)

        parts = [LastToken(name=nm("last_token"))(x), AttentionPool(name=nm("attn_pool"))(x)]
        if stats is not None and not two_tower:
            parts.append(layers.Dense(d_model, activation="gelu", name=nm("stats_proj"))(stats))
        if coin_vec is not None and not (two_tower or level_film):
            parts.append(coin_vec)
        h = layers.Concatenate(name=nm("readout_concat"))(parts)
        h = layers.Dense(d_model, activation="gelu", name=nm("readout_fc"))(h)
        if use_linear_path:
            lin_in = z if lvl is None else layers.Concatenate(axis=-1, name=nm("linear_shape_and_level"))([z, lvl])
            lin = layers.Dense(d_model, kernel_regularizer=regularizers.l2(linear_path_l2), name=nm("linear_path"))(
                layers.Flatten(name=nm("flatten_window"))(lin_in))
            h = layers.Add(name=nm("add_linear_path"))([h, lin])
        if two_tower:   # دمج متأخّر: البرج الزمني h (على z وحده) + البرج المقطعي على متجه المستوى
            h_level = layers.Dense(d_model, activation="gelu", name=nm("level_tower"))(level_vec)
            if fusion == "gate":
                h = GatedFusion(d_model, name=nm("tower_fusion"))([h, h_level])
            else:
                h = layers.Dense(d_model, activation="gelu", name=nm("tower_fusion"))(
                    layers.Concatenate(name=nm("towers_concat"))([h, h_level]))
        return h

    if tfs is None:
        h = _branch(inputs, seq_len)
    else:
        shared_coin = _make_coin_vec() if n_coins else None      # تضمين العملة واحد يُشارَك بين الفروع
        h = layers.Concatenate(name="branch_concat")(
            [_branch(inputs_by_tf[tf], int(seq_len[tf]), f"_{tf}", shared_coin) for tf in tfs])

    h = RMSNorm(name="trunk_norm")(h)
    h = layers.Dropout(dropout, name="trunk_drop")(h)

    # -- رؤوس nig_regression: تُبنى أولاً بمعزل عن التسمية النهائية، لأن
    #    enforce_order يحتاج القيم الخام الثلاث معاً قبل الحسم --
    nig_targets = [t for t in price_targets if "nig_regression" in head_types.get(t, [])]
    raw_nig = {}
    for t in nig_targets:
        nig = NIGHead(head_hidden, dropout, nu_min, alpha_min, beta_min, name=f"nig_{t}")
        raw_nig[t] = nig(h)  # (mu_raw, nu, alpha, beta)

    apply_order = enforce_order and {"high", "low", "close"}.issubset(nig_targets)
    if apply_order:
        mu_h, mu_c, mu_l = OrderedMeans(name="ordered_means")(
            [raw_nig["close"][0], raw_nig["high"][0], raw_nig["low"][0]])
        mus = {"high": mu_h, "close": mu_c, "low": mu_l}
    else:
        mus = {t: raw_nig[t][0] for t in nig_targets}

    outputs = {}
    for t in nig_targets:
        _, nu, alpha, beta = raw_nig[t]
        epi, ale = NIGUncertainty(unc_max=unc_max, name=f"unc_{t}")([nu, alpha, beta])
        conf = ConfidenceHead(name=f"conf_{t}")([h, nu, alpha, beta])
        outputs.update({
            f"y_{t}": mus[t], f"y_{t}_nu": nu, f"y_{t}_alpha": alpha, f"y_{t}_beta": beta,
            f"y_{t}_epistemic": epi, f"y_{t}_aleatoric": ale, f"y_{t}_confidence": conf,
        })

    # -- كل أنواع الرؤوس الأخرى (تصنيف، ...): مستقلّة لكل هدف، عبر السجلّ --
    for t in price_targets:
        for head_type in head_types.get(t, []):
            if head_type == "nig_regression":
                continue
            outputs.update(build_head_outputs(head_type, h, t, head_cfg))

    model_inputs = list(inputs_by_tf.values()) if tfs is not None else [inputs]
    return Model(model_inputs + [coin_in] if n_coins else (model_inputs if tfs is not None else inputs),
                 outputs, name=name)
