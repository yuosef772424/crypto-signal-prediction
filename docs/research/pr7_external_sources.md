# PR7 — تحقّق من المصادر الخارجية (اقتباسات حرفية)

> **الغرض**: التحقّق من الادّعاءات المستخدمة في PR7 عن الحفظ/overfitting، dropout، الضجيج على
> المدخلات، label smoothing، حجم النموذج، طريقة التحقّق، التضمينات، الانتباه بين الأصول،
> معماريات GRU/TCN/Transformer، تجميع البذور، والتعلّم المستمر.
>
> **الجمهور**: مطوّر يبني على PR7 ويحتاج إلى أصل كل رقم وكل ادّعاء.
>
> **قواعد هذا الملف**: كل ما هو بين علامتي تنصيص `" "` منقول حرفياً من المصدر بلغته الأصلية.
> الأرقام منقولة كما هي ولم تُعَد صياغتها. ما لم تُفتح صفحته مكتوب فيه **"لم تُفتح"** بلا تخمين.
>
> **طريقة الجلب**: أداة `WebFetch` كانت محجوبة على جميع النطاقات في هذه البيئة
> (`Unable to verify if domain ... is safe to fetch`)، فتمّ استخدام المتصفّح المدمج
> (تحميل الصفحة ثم استخراج نصّها) و`WebSearch`. تاريخ الجلب: 2026-09-27.
>
> **نطاق محجوب**: `reddit.com` متعذّر في هذه البيئة على مستويين — المتصفّح يرفض فتحه
> (`not allowed due to safety restrictions`) وزاحف `WebSearch` مستبعَد منه (`HTTP 400`)؛
> ولذلك القسم 14 مكتوب فيه **"لم تُفتح"** بلا أي محتوى مُخمَّن.
>
> **حصيلة التحقّق** في آخر الملف تختصر حالة كل مصدر والادّعاءات التي لم تُؤكَّد.

---

## 1. Jane Street Market Prediction — Yirun's Solution (1st place)

- **الرابط**: https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s
- **الكاتب**: فريق VECTOR — Yirun Zhang (`gogo827jz`)، Mingjie Wang (`xiaowangiiiii`)، Colton Smith (`coltonfsmith`)، yuanzhe zhou (`yuanzhezhou`). كاتب النصّ هو Yirun Zhang (مُعلَّم TOPIC AUTHOR).
- **التاريخ**: "Solution Writeup · 1st place · Oct 18, 2021"
- **الحالة**: فُتحت بالكامل (النصّ + 48 تعليقاً).

### الأرقام كما هي

- "The single-model AE-MLP scores 6022.202 on the private leaderboard, which is still 1st place!"
- "5-fold 31-gap purged group time-series split"
- "Remove first 85 days for training since they have different feature variance"
- "Train the model with 3 different random seeds and take the average to reduce prediction variance"
- تقييم التصويت على الكتابة: "354" — وعدد التعليقات: "48 Comments"

### الحفظ / overfitting

- "I have realised that this training may cause label leakage because the autoencoder has seen part of the data in the validation set in each CV split and it can generate label-leakage features to overfit."
- "Train autoencoder and MLP together in each CV split to prevent data leakage"
- "Add Gaussian noise layer before encoder for data augmentation and to prevent overfitting"
- من ردّ الكاتب في التعليقات: "I only remember that I was criticised for publishing the neural network starter notebook which uses the vanilla group CV split (not purged time-series CV) and it potentially contains time-leakage and overfitting problems. Actually, he was right on this, so I added some explanation later."
- من ردّ الكاتب على تعليق آخر: "Is this one your overfitting classification model? Its score looks not that 'overfitting' because lots of overfitting models score 10000+ in the previous LB."

### الضجيج على المدخلات

- "Add Gaussian noise layer before encoder for data augmentation and to prevent overfitting"
- **موضع الضجيج**: قبل المُرمِّز (encoder)، أي على المدخلات الأصلية — لا على الطبقات الداخلية.
- لا تُذكر قيمة عدديّة لانحراف الضجيج في نصّ الكتابة نفسه (القيمة في الدفتر — انظر القسم 2).
- من تعليق مشارك آخر (Andy، المركز 15): "In my case, a swap noise DAE gave me a better result and added varieties to my ensemble." — أي نوع ضجيج مختلف (swap noise) لا Gaussian.

### dropout

- "Batch Normalisation and Dropout are used for MLP"
- لا تُذكر أي نسبة عدديّة لـ dropout في نصّ الكتابة (النِّسب في الدفتر — انظر القسم 2).

### label smoothing

- **لا ذكر لـ label smoothing في نصّ هذه الكتابة إطلاقاً.** (المعامل موجود في الدفتر — انظر القسم 2.)

### حجم النموذج

- لا تُذكر أي أرقام لعدد الطبقات أو الوحدات أو المعاملات في نصّ الكتابة. الوصف نوعيّ فقط:
  "Use autoencoder to create new features, concatenating with the original features as the input to the downstream MLP model"
- "Use swish activation function instead of ReLU to prevent 'dead neuron' and smooth the gradient"
- عن اختيار المعماريّة، ردّ الكاتب: "I use hyperopt to search the global hyperparameter set. It improves the score a lot."
- "Use Hyperopt to find the optimal hyperparameter set"
- تعليق مشارك آخر (Andy): "Wow, did not expect your solution was this simple." وردّ الكاتب:
  "I have various more complex solutions but not submitted. Not sure how they perform."

### طريقة التحقّق (purged / grouped CV)

- "5-fold 31-gap purged group time-series split"
- "Train autoencoder and MLP together in each CV split to prevent data leakage"
- "Only use the models (with different seeds) trained in the last two CV splits since they have seen more data"
- "Only monitor the BCE loss of MLP instead of the overall loss for early stopping"
- "Remove first 85 days for training since they have different feature variance"
- "Forward-fill the missing values"

### التضمينات (embeddings)

- **لا ذكر لطبقات embedding.** أقرب ما في النصّ هو تمثيل المُرمِّز التلقائي كميزات مُولَّدة:
  "Use autoencoder to create new features, concatenating with the original features as the input to the downstream MLP model"
- "Add target information to autoencoder (supervised learning) to force it to generate more relevant features, and to create a shortcut for backpropagation of gradient"
- من تعليق François BOYER: "Also when you say you trained autoencoder and MLP together in the same model : did you also train them on exactly the same data ? And using the autoencoder embedding as input. I was afraid to train them on the same data because I didn't want the autoencoder to provide too \"optimistic\" / overfitted embedding during training" — سؤال مشارك، وردّ الكاتب كان إحالة إلى الشكل في الدفتر فقط.

### الانتباه بين الأصول (cross-sectional / asset attention)

- **لا ذكر إطلاقاً.** لا attention ولا معالجة مقطعيّة بين الأصول في هذا الحلّ.

### GRU / TCN / Transformer

- **لا ذكر لأيٍّ منها.** المعماريّة هي مُرمِّز تلقائي مُشرَف + MLP فقط (autoencoder + MLP).
  الحلّ النهائي للفريق: "In this competition, I have used a supervised autoencoder MLP approach and my teammates use XGBoost. Our final submission is a simple blend of these two models."

### تجميع البذور (seed averaging)

- "Train the model with 3 different random seeds and take the average to reduce prediction variance"
- "During inference, the mean of all predicted actions is taken as the final probability"
- "Only use the models (with different seeds) trained in the last two CV splits since they have seen more data"

### التعلّم المستمر

- **لا ذكر لتعلّم مستمر أو online learning.** الأهداف تُحوَّل إلى تصنيف متعدّد التسميات:
  "Transfer all resp targets (resp, resp_1, resp_2, resp_3, resp_4) to action for multi-label classification"
- "Use the mean of the absolute values of all resp targets as sample weights for training so that the model can focus on capturing samples with large absolute resp."

---
## 2. Jane Street: Supervised Autoencoder MLP (الدفتر — مصدر الأرقام)

- **الرابط**: https://www.kaggle.com/code/gogo827jz/jane-street-supervised-autoencoder-mlp
- **الكاتب**: "YIRUN ZHANG"
- **التاريخ**: ما ظهر حرفياً على الصفحة هو "YIRUN ZHANG · 5Y AGO · 65,981 VIEWS" و"Version 2 of 2".
  لم يُعرض تاريخ مطلق، فلا أُسنِد له تاريخاً محدّداً. (الكتابة المرتبطة به في القسم 1 مؤرَّخة "Oct 18, 2021".)
- **الحالة**: فُتح بالكامل. الكود مستخرج من إطار `kaggleusercontent.com/.../__results__.html`
  لأن الصفحة الأمّ لا تحمل الكود في DOM الخاصّ بها (`preCount: 0`).
- **الترخيص المعلن على الصفحة**: "This Notebook has been released under the Apache 2.0 open source license."
- **بيانات تشغيل أخرى كما هي**: "14m 25s · GPU P100" — "865.5 second run - successful" —
  "Copy & Edit 944" — تصويت "1017" — "Comments (16)" — "Copied from private notebook (+11,-172)"

### تعريف النموذج — الكود الحرفي

```python
def create_ae_mlp(num_columns, num_labels, hidden_units, dropout_rates, ls = 1e-2, lr = 1e-3):

    inp = tf.keras.layers.Input(shape = (num_columns, ))
    x0 = tf.keras.layers.BatchNormalization()(inp)

    encoder = tf.keras.layers.GaussianNoise(dropout_rates[0])(x0)
    encoder = tf.keras.layers.Dense(hidden_units[0])(encoder)
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.layers.Activation('swish')(encoder)

    decoder = tf.keras.layers.Dropout(dropout_rates[1])(encoder)
    decoder = tf.keras.layers.Dense(num_columns, name = 'decoder')(decoder)

    x_ae = tf.keras.layers.Dense(hidden_units[1])(decoder)
    x_ae = tf.keras.layers.BatchNormalization()(x_ae)
    x_ae = tf.keras.layers.Activation('swish')(x_ae)
    x_ae = tf.keras.layers.Dropout(dropout_rates[2])(x_ae)

    out_ae = tf.keras.layers.Dense(num_labels, activation = 'sigmoid', name = 'ae_action')(x_ae)

    x = tf.keras.layers.Concatenate()([x0, encoder])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rates[3])(x)

    for i in range(2, len(hidden_units)):
        x = tf.keras.layers.Dense(hidden_units[i])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('swish')(x)
        x = tf.keras.layers.Dropout(dropout_rates[i + 2])(x)

    out = tf.keras.layers.Dense(num_labels, activation = 'sigmoid', name = 'action')(x)

    model = tf.keras.models.Model(inputs = inp, outputs = [decoder, out_ae, out])
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = lr),
                  loss = {'decoder': tf.keras.losses.MeanSquaredError(),
                          'ae_action': tf.keras.losses.BinaryCrossentropy(label_smoothing = ls),
                          'action': tf.keras.losses.BinaryCrossentropy(label_smoothing = ls),
                         },
                  metrics = {'decoder': tf.keras.metrics.MeanAbsoluteError(name = 'MAE'),
                             'ae_action': tf.keras.metrics.AUC(name = 'AUC'),
                             'action': tf.keras.metrics.AUC(name = 'AUC'),
                            },
                 )

    return model
```

### المعاملات الفعليّة المستخدمة — الكود الحرفي

```python
params = {'num_columns': len(features),
          'num_labels': 5,
          'hidden_units': [96, 96, 896, 448, 448, 256],
          'dropout_rates': [0.03527936123679956, 0.038424974585075086, 0.42409238408801436, 0.10431484318345882, 0.49230389137187497, 0.32024444956111164, 0.2716856145683449, 0.4379233941604448],
          'ls': 0,
          'lr':1e-3,
         }
```

```python
n_splits = 5
group_gap = 31
```

### الأرقام كما هي — مفكوكة على مواضعها في الكود

| الموضع في الكود | القيمة الحرفية |
|---|---|
| `GaussianNoise(dropout_rates[0])` على `x0` (بعد BatchNorm للمدخلات، قبل المُرمِّز) | `0.03527936123679956` |
| `Dropout(dropout_rates[1])` قبل `Dense(..., name='decoder')` | `0.038424974585075086` |
| `Dropout(dropout_rates[2])` في فرع `x_ae` | `0.42409238408801436` |
| `Dropout(dropout_rates[3])` بعد `Concatenate([x0, encoder])` — أي على مدخل الـMLP | `0.10431484318345882` |
| `Dropout(dropout_rates[4])` بعد `Dense(hidden_units[2])` | `0.49230389137187497` |
| `Dropout(dropout_rates[5])` بعد `Dense(hidden_units[3])` | `0.32024444956111164` |
| `Dropout(dropout_rates[6])` بعد `Dense(hidden_units[4])` | `0.2716856145683449` |
| `Dropout(dropout_rates[7])` بعد `Dense(hidden_units[5])` | `0.4379233941604448` |
| `hidden_units[0]` — `Dense` المُرمِّز | `96` |
| `hidden_units[1]` — `Dense` فرع `x_ae` | `96` |
| `hidden_units[2:]` — طبقات الـMLP | `896, 448, 448, 256` |
| `label_smoothing = ls` في كلتا خسارتَي `BinaryCrossentropy` | الافتراضي في توقيع الدالة `ls = 1e-2`، **والقيمة الممرَّرة فعلياً `'ls': 0`** |
| `lr` | `1e-3` |
| `batch_size` | `4096` |
| `epochs` | `100` |
| `num_labels` | `5` |
| `tf.random.set_seed(42)` | `42` |

### الضجيج على المدخلات — ملاحظة دقيقة

- الطبقة هي `tf.keras.layers.GaussianNoise(0.03527936123679956)` وتُطبَّق على `x0` أي
  **بعد** `BatchNormalization()(inp)` مباشرةً و**قبل** أوّل `Dense` في المُرمِّز.
- في Keras وسيط `GaussianNoise` هو `stddev` (انحراف معياري، لا تباين). وبما أن المدخل مُطبَّع
  بـBatchNorm (انحراف ≈ 1)، فإن `0.0353` تعني ضجيجاً بنحو 3.5% من الانحراف المعياري للميزة.
- **اسم الوسيط مُضلِّل**: القيمة تُمرَّر عبر `dropout_rates[0]` لكنها ليست معدّل dropout — إنما stddev.
  من يقرأ القائمة سطحياً قد يخلط بين الاثنين.

### label smoothing — ملاحظة دقيقة

- الافتراضي في التوقيع: `ls = 1e-2` أي 0.01.
- **لكن `params` تمرّر `'ls': 0`** — أي **label smoothing مُعطَّل فعلياً في هذا الدفتر المنشور**،
  مع بقاء البنية التحتيّة له قائمة. المعاملات وُجدت بـHyperopt ("Use Hyperopt to find the optimal
  hyperparameter set" في القسم 1)، فالصفر نتيجة بحث لا سهو.
- أي ادّعاء بأن حلّ المركز الأوّل "استخدم label smoothing بقيمة 0.01" غير مدعوم بهذا الدفتر:
  0.01 هو الافتراضي المُتخطَّى، والقيمة الفعليّة صفر.

### طريقة التحقّق (purged / grouped CV)

```python
gkf = PurgedGroupTimeSeriesSplit(n_splits = n_splits, group_gap = group_gap)
for fold, (tr, te) in enumerate(gkf.split(train['action'].values, train['action'].values, train['date'].values)):
```

- `n_splits = 5`، `group_gap = 31`، والتجميع على `train['date'].values`.
- إيقاف مبكر حرفياً:
  `EarlyStopping(monitor = 'val_action_AUC', min_delta = 1e-4, patience = 10, mode = 'max', baseline = None, restore_best_weights = True, verbose = 0)`
- المراقَب هو `val_action_AUC` أي رأس الـMLP وحده — مطابق لبند الكتابة
  "Only monitor the BCE loss of MLP instead of the overall loss for early stopping".
- **لم تُفتح**: جسم صنف `PurgedGroupTimeSeriesSplit` نفسه في خليّة مطويّة (`In [3]` — "Show hidden code")
  ولا يظهر مصدرها في `__results__.html`. لم أقرأ تعريف الصنف ولا أخمّنه.

### المعالجة المسبقة — الكود الحرفي

```python
train = train.query('date > 85').reset_index(drop = True)
train = train.query('weight > 0').reset_index(drop = True)
train[features] = train[features].fillna(method = 'ffill').fillna(0)
train['action'] = ((train['resp_1'] > 0) & (train['resp_2'] > 0) & (train['resp_3'] > 0) & (train['resp_4'] > 0) & (train['resp'] > 0)).astype('int')
```

```python
resp_cols = ['resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']
sw = np.mean(np.abs(train[resp_cols].values), axis = 1)
```

### النتائج كما هي

```
Fold 0 ROC AUC:	 0.527747392654419
Fold 1 ROC AUC:	 0.5351583361625671
Fold 2 ROC AUC:	 0.5391918420791626
Fold 3 ROC AUC:	 0.5394068956375122
Fold 4 ROC AUC:	 0.5481765866279602
Weighted Average CV Score: 0.5427706055343151
```

- **حجم الإشارة**: أعلى طيّة `0.5481765866279602` وأدناها `0.527747392654419`. أي أن حلّ
  المركز الأوّل في هذه المسابقة يعمل على AUC بين 0.528 و0.548.
- أوزان المتوسّط المرجّح، تعليقاً حرفياً في الكود:
  "# [0.0625, 0.0625, 0.125, 0.25, 0.5] for 5 fold" مع مرجع
  "# weighted average as per Donate et al.'s formula" و"# https://doi.org/10.1016/j.neucom.2012.02.053"

### التضمينات / الانتباه بين الأصول / GRU / TCN / Transformer / التعلّم المستمر

- **لا ذكر لأيٍّ منها في الكود.** لا `Embedding`، ولا `Attention`، ولا `GRU`/`LSTM`/`Conv1D`،
  ولا أي حلقة تحديث تدريجي. البنية بالكامل `Dense` + `BatchNormalization` + `Dropout` +
  `GaussianNoise` + `Activation('swish')`.
- تجميع البذور: الدفتر المنشور يثبّت بذرة واحدة `tf.random.set_seed(42)`. تجميع الثلاث بذور
  المذكور في الكتابة (القسم 1) ليس في هذا الدفتر.

---
## 3. Ubiquant Market Prediction — [1st Place Solution] - Our Betting Strategy

- **الرابط**: https://www.kaggle.com/competitions/ubiquant-market-prediction/writeups/k-i-y-1st-place-solution-our-betting-strategy
- **الكاتب**: "yuuniee, Karbitrage, kimimgo" (المعرّفات: `yuuniekiri`، `jihoonchung`، `imgyukim`)
- **التاريخ**: "Solution Writeup · 1st place · Jul 20, 2022"
- **الحالة**: فُتحت بالكامل. "0 Comments" — تصويت "198".

### الأرقام كما هي

- "B. Feature Engineering: 300 + 100"
- "(CV: 0.141 -> 0.154, LB: 0.141 -> 0.149 based on LGBM single model)"
- "C. Data Sampling: (train.csv + supplemental_train.csv)[2400000:]"
- "we used the last 2400k rows of it"
- "as a result of testing in memory (RAM 13GB), the Data Row was stable up to approximately 2500k, but in order to pursue more stability, an additional 100k was dropped. :("
- "best_corr = corr.iloc[3:103, 0].to_list()" — أي أعلى 100 ميزة بعد تخطّي أوّل 3.
- "The ensemble method : Average of (LGBM x 5 Folds) + (TABNET x 5 Folds)"
- "training set: (time_id >= 0) and (time_id <= 1000) test set: (time_id >= 1001) and (time_id <= 1202)"
- درجة الإرسال الثاني: "this submission resulted in a silver medal score(0.115796)"
- "there were a little more about 150 features"

### الحفظ / overfitting

- "To elaborate a bit, this has the advantage of being able to include more data in a variety of ways, while at the same time risking overfitting for future references."
- "So, to reduce the risk of overfitting, we used an early stop for validation and a method of limiting the number of training (num_boost_round or epoch) to a certain value or less."
- "Some Custom MLP models were also candidates, and there were models with a significant ensemble effect even on the LB basis, but as a result, they were excluded because they were not stable in CV."
- "it was considerably overfitted with unlimited training (of course, there is an early stop for validation). However, this submission resulted in a silver medal score(0.115796). In this part, I think there was an effect from the exclusion of supplemental data or excessive CV and LB overfitting."
- "Then, in time series data training, I think that the method such as early stop at the time when the CV score decreases to a few percent or less is also a factor to consider, but it is only a guess."
- "LGBM is a powerful model whose performance has been proven in many competitions. It was also the most stable (especially the consistency of CV and LB) and excellent in the experiment on the competition data. (at least for me)"
- "Also, I think we were lucky enough that the market conditions made our model smile."

### dropout

- **لا ذكر لـ dropout إطلاقاً** في هذه الكتابة. التنظيم المذكور هو حصر عدد جولات/حِقب التدريب
  والإيقاف المبكر: "limiting the number of training (num_boost_round or epoch) to a certain value or less".

### الضجيج على المدخلات

- **لا ذكر إطلاقاً.**

### label smoothing

- **لا ذكر إطلاقاً.** الخسائر المذكورة حرفياً:
  "For loss_fn, rmse and mse are used, respectively, and Pearson Corr. is commonly used for eval_metric."

### حجم النموذج

- لا تُذكر أعداد طبقات أو وحدات أو معاملات. النماذج المذكورة اسماً فقط:
  "A. Used Models: LGBM, TABNET"
- "Sorry if you were expecting a special model, this time it's LGBM and TABNET. :D"
- "In addition, candidate models to be ensembled in LGBM were found and tested. Among them, TABNET was selected with the best ensemble effect while being relatively stable."
- القيد الحقيقي كان الذاكرة لا الحجم: "this was the largest compromise within the limits allowed by
  Kernel Resources (especially RAM), and I guess that there would be an additional score increase if
  more useful features were included."
- "the added features caused a huge memory increase, and we had to trade off [more features] VS [more data]."

### طريقة التحقّق (purged / grouped CV)

- "D. Cross Validation for FE and Parameter Tuning: PurgedGroupTimeSeries, TimeSerieseSplit"
- "E. Cross Validation for Training : KFold"
- وفي العنوان المفصّل للبند نفسه: "E. Cross Validation for Training : KFold, GroupKFold"
  (تعارض داخلي في نصّ المصدر بين الملخّص والتفصيل — منقول كما هو دون ترجيح.)
- "As already described above, various CVs were used to measure the performance of FE, and the most effective FE was selected from all of these CVs. This course also includes Hyper Parameter Tuning."
- "There was another trivial strategy for this choice, which is a self-test data set, which looks like this: training set: (time_id >= 0) and (time_id <= 1000) test set: (time_id >= 1001) and (time_id <= 1202)"
- "There were various training methods, but as a result of testing in the above environment, [limited training KFold] was selected as the method that showed the most stable and excellent results."
- "For this, various probabilistic measures were performed on Score gains on PurgedGroup and TimeSeriesSplit CV, and we finally decided that additional features have a probabilistic advantage in score improvement."
- **ملاحظة مهمّة**: purged CV يُستخدم لاختيار الميزات وضبط المعاملات، أما التدريب النهائي فبـKFold
  عادي — أي فصل صريح بين CV للقياس وCV للتدريب.

### التضمينات (embeddings)

- **لا ذكر لطبقات embedding.** الميزات المضافة كلّها إحصائيّة مقطعيّة (انظر البند التالي).

### الانتباه بين الأصول (cross-sectional / asset attention)

- **لا ذكر لأي attention.** لكن يوجد **مُكافئ يدويّ للمعالجة المقطعيّة**: الميزات المئة المضافة هي
  متوسّطات مقطعيّة على مستوى `time_id`، أي معلومة من الأصول الأخرى في اللحظة نفسها:
  "[ The average value at each time_id for the top 100 features by obtaining and sorting the corr. of 300 features and each target ]"
- الكود الحرفي:

```python
features = [f'f_{i}' for i in range(300)]

corr = train_df[features[:] + ['target']].corr()['target'].reset_index()
corr['target'] = abs(corr['target'])
corr.sort_values('target', ascending = False, inplace = True)
best_corr = corr.iloc[3:103, 0].to_list()

time_id_mean_features = []
for col in tqdm(best_corr):
   mapper = train_df.groupby(['time_id'])[col].mean().to_dict()
   train_df[f'time_id_{col}'] = train_df['time_id'].map(mapper)
   train_df[f'time_id_{col}'] = train_df[f'time_id_{col}'].astype(np.float16)
   time_id_mean_features.append(f'time_id_{col}')

features += time_id_mean_features
```

- وحساب الميزة نفسها على مجموعة الاختبار (ردّ الكاتب على @agenlu @hdynamics @ygygyv):
  "How to make a feature from test? > Take the average of each feature in the test set by referring to the list of features stored in best_corr in the code above."

```python
for col in best_corr:
  test_df['time_id'] = test_df['row_id'].str[0:4].astype(np.int64)
  mapper = test_df.groupby(['time_id'])[col].mean().to_dict()
  test_df[f'time_id_{col}'] = test_df['time_id'].map(mapper)
```

### GRU / TCN / Transformer

- **لا ذكر لأيٍّ منها.** النموذجان: LGBM (أشجار) وTABNET (شبكة جدوليّة). ونماذج MLP المخصّصة
  استُبعدت: "Some Custom MLP models were also candidates ... they were excluded because they were not stable in CV."

### تجميع البذور (seed averaging)

- **لا تجميع بذور صريح.** التجميع على الطيّات والنماذج:
  "The ensemble method : Average of (LGBM x 5 Folds) + (TABNET x 5 Folds)"

### التعلّم المستمر

- **لا ذكر لتعلّم مستمر أو online learning.**

---
## 4. Optiver — Trading at the Close — 1st place solution (hyd)

- **الرابط**: https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution
- **الكاتب**: "hyd" (المعرّف `hydantess`)
- **التاريخ**: "Solution Writeup · 1st place · Oct 14, 2024"
- **الحالة**: فُتحت بالكامل (النصّ + 70 تعليقاً). تصويت "337".
- **ملاحظة**: الكود غير منشور — "Sorry, I don't plan to share my code." (ردّ الكاتب مرّتين).
  فالأرقام أدناه كلّها من النصّ والتعليقات، لا من كود.

### الأرقام كما هي

- "My final model(CV/Private LB of 5.8117/5.4030) was a combination of CatBoost (5.8240/5.4165), GRU (5.8481/5.4259), and Transformer (5.8619/5.4296), with respective weights of 0.5, 0.3, 0.2 searched from validation set. And these models share same 300 features."
- جدول النتائج الكامل كما هو (الأعمدة: model name / validation set w/o PP / validation set w/ PP / test set w/o OL w/ PP / test set w/ OL one time w/ PP / test set w/ OL five times w/ PP):

| model name | validation set w/o PP | validation set w/ PP | test set w/o OL w/ PP | test set w/ OL one time w/ PP | test set w/ OL five times w/ PP |
|---|---|---|---|---|---|
| CatBoost | 5.8287 | 5.8240 | 5.4523 | 5.4291 | 5.4165 |
| GRU | 5.8519 | 5.8481 | 5.4690 | 5.4368 | 5.4259 |
| Transformer | 5.8614 | 5.8619 | 5.4678 | 5.4493 | 5.4296 |
| GRU + Transformer | 5.8233 | 5.8220 | 5.4550 | 5.4252 | 5.4109 |
| CatBoost + GRU + Transformer | 5.8142 | 5.8117 | 5.4438 | 5.4157 | 5.4030*(overtime) |

- "train on first 400 days and choose last 81 days as my holdout validation set"
- "My models have 300 features in the end."
- "I just choose top 300 features by CatBoost model's feature importance."
- "I retrain my model every 12 days, 5 times in total."
- "So there are 4 online training updates in total. I estimate that the best score would be around 5.400 if not overtime."
- "GRU input tensor's shape is (batch_size, 55 time steps, dense_feature_dim), followed by 4 layers GRU, output tensor's shape is (batch_size, 55 time steps)."
- "Transformer input tensor's shape is (batch_size, 200 stocks, dense_feature_dim), followed by 4 layers transformer encoder layers, output tensor's shape is (batch_size, 200 stocks)."
- "I think most teams can only use up to 200 features when training GBDT if online training strategy is adopted."
- من الكود: `pl.when(pl.col('seconds_in_bucket') < 300).then(0).when(pl.col('seconds_in_bucket') < 480).then(1).otherwise(2)` — العتبات 300 و480.
- من الكود: `rolling_mean(100, min_periods=1)` — النافذة 100.
- ردّ الكاتب على سؤال "Why the feature number you choose is 300, not 200 or 400?":
  "based on CV score, 300 is better than 200. 400 will have memory issue."
- من تعليق Geremie Yeo (المركز 14): "The weighted postprocessing improved our selected sub from 5.4457 to 5.4405, good for 11th place."
  و"Our competition selected submission scored 5.4457 without any postprocessing / Adding the weighted mean postprocessing, improved it from 5.4457 to 5.4405"

### الحفظ / overfitting

- **لا يستخدم الكاتب كلمة overfitting إطلاقاً في نصّ الحلّ.** البديل عنده هو محاذاة CV مع LB:
  "The CV score aligns with leaderboard score very well which makes me believe that this competition wouldn't shake too much. So I just focus on improving CV in most of time."
- أقرب ما يتعلّق بالتسريب: "Yes, it's seq2seq model but not bidirection for avoiding label leak."
  — أي أن الاتجاه الأحاديّ في GRU اختيار لمنع تسريب التسمية، لا لسبب سعوي.
- ما لم ينجح، حرفياً: "What not worked for me / ensemble with 1dCNN or MLP. / multi-days input instead of singe day input when applying GRU models / larger transformer, e.g. deberta / predict target bucket mean by GBDT"
  — **"larger transformer, e.g. deberta" فشل** — دليل مباشر على أن تكبير النموذج لم يُفِد هنا.

### dropout

- **لا ذكر لـ dropout إطلاقاً**، ولا أي نسبة.

### الضجيج على المدخلات

- **لا ذكر إطلاقاً.**

### label smoothing

- **لا ذكر إطلاقاً.** (المهمّة انحدار على هدف مستمرّ، والمقياس يبدو MAE بقيم ≈5.4.)

### حجم النموذج

- "followed by 4 layers GRU" — 4 طبقات.
- "followed by 4 layers transformer encoder layers" — 4 طبقات.
- لا تُذكر أبعاد مخفيّة ولا عدد رؤوس انتباه ولا عدد معاملات.
- نوع الـTransformer، ردّ الكاتب على سؤال "you mean Transformer models that are specific for tabular data, namely FT-Transformer or TabTransformer?":
  "No, It's torch.transformerencoder ~"
  — أي `torch.nn.TransformerEncoder` القياسي، لا معماريّة جدوليّة متخصّصة.
- **الأكبر لم يكن أفضل**: "larger transformer, e.g. deberta" مدرَج تحت "What not worked for me".
- عن نموذج مستوى ثانٍ (stacking)، ردّ الكاتب: "It's very time-consuming to train a second-level model and doesn't help much in fact. So i just use weighted sum."

### طريقة التحقّق (purged / grouped CV)

- "My validation strategy is pretty simple, train on first 400 days and choose last 81 days as my holdout validation set."
- **لا purged CV ولا grouped CV ولا k-fold إطلاقاً** — تقسيم زمني واحد (holdout) فقط.
  هذا يخالف حلّ Jane Street (القسم 1) الذي استخدم 5-fold purged group split.
- "The CV score aligns with leaderboard score very well which makes me believe that this competition wouldn't shake too much. So I just focus on improving CV in most of time."
- أوزان التجميع اختيرت على مجموعة التحقّق نفسها: "with respective weights of 0.5, 0.3, 0.2 searched from validation set."

### التضمينات (embeddings)

- **لا ذكر لطبقات embedding** ولا لتضمين `stock_id`. المدخل في كلا النموذجين كثيف:
  "dense_feature_dim" في وصف كلٍّ من GRU والTransformer.

### الانتباه بين الأصول (cross-sectional / asset attention) — **المصدر الأقوى في هذا الملف**

- "Transformer input tensor's shape is (batch_size, 200 stocks, dense_feature_dim), followed by 4 layers transformer encoder layers, output tensor's shape is (batch_size, 200 stocks)."
  — **بُعد التسلسل هو الأصول (200 سهماً) لا الزمن.** أي انتباه مقطعيّ بين الأصول في اللحظة نفسها.
- "A small trick that turns output into zero mean is helpful."

```python
out = out - out.mean(1, keepdim=True)
```

  — إزاحة المتوسّط على بُعد الأصول (`dim=1`)، أي تحييد مقطعيّ داخل النموذج.
- سؤال WindClimber (المركز 23): "If I understand correctly, the GRU model is designed to capture time-series dynamics, while the Transformer is for the cross-sectional dynamics, and thus the data feeding process is different for these two. Please feel free to correct me if my interpretation is off."
  **وردّ الكاتب حرفياً**: "You are right!"
- وردّ الكاتب على سؤال wen6Lev57q4 (لماذا لا يُخرِج الـTransformer مثل GRU):
  "I want transformer to learn info across different stocks, GRU to learn sequence info. You can combine these two modules in one model of course."
- **الأثر الرقمي للجمع بين المحورين** (من الجدول أعلاه): GRU وحده 5.8519 والTransformer وحده 5.8614،
  و"GRU + Transformer" معاً 5.8233 — أي أن جمع المحور الزمني بالمحور المقطعي أفضل من أيٍّ منهما منفرداً
  (وهذا رقم من الجدول لا استنتاج، إذ 5.8233 < 5.8519 و< 5.8614 على مقياس أقل-أفضل).
- المعالجة اللاحقة كذلك مقطعيّة: "Subtract weighted-mean is better than average-mean since metric already told."

```python
test_df['stock_weights'] = test_df['stock_id'].map(stock_weights)
test_df['target'] = test_df['target'] - (test_df['target'] * test_df['stock_weights']).sum() / test_df['stock_weights'].sum()
```

- وميزات مقطعيّة يدويّة أيضاً (rank features grouped by seconds_in_bucket):

```python
 *[(pl.col(col).mean() / pl.col(col)).over(['date_id', 'seconds_in_bucket']).cast(pl.Float32).alias('{}_seconds_in_bucket_group_mean_ratio'.format(col)) for col in base_features],
 *[(pl.col(col).rank(descending=True,method='ordinal') / pl.col(col).count()).over(['date_id', 'seconds_in_bucket']).cast(pl.Float32).alias('{}_seconds_in_bucket_group_rank'.format(col)) for col in base_features],
```

- وميزات التجميع على `seconds_in_bucket_group`:

```python
pl.when(pl.col('seconds_in_bucket') < 300).then(0).when(pl.col('seconds_in_bucket') < 480).then(1).otherwise(2).cast(pl.Float32).alias('seconds_in_bucket_group'),

 *[(pl.col(col).first() / pl.col(col)).over(['date_id', 'seconds_in_bucket_group', 'stock_id']).cast(pl.Float32).alias('{}_group_first_ratio'.format(col)) for col in base_features],
 *[(pl.col(col).rolling_mean(100, min_periods=1) / pl.col(col)).over(['date_id', 'seconds_in_bucket_group', 'stock_id']).cast(pl.Float32).alias('{}_group_expanding_mean{}'.format(col, 100)) for col in base_features]
```

- ملاحظة: `polars`، بتوضيح الكاتب: "This is polars. / polars over == pandas groupby().transform()"

### GRU / TCN / Transformer

- **GRU**: "GRU input tensor's shape is (batch_size, 55 time steps, dense_feature_dim), followed by 4 layers GRU, output tensor's shape is (batch_size, 55 time steps)."
- استخدام المخرج: ردّ الكاتب على سؤال lele (المركز 45) عن سبب كون المخرج بطول 55 لا 1:
  "You're right. Will take the last timestamp when inference."
- اتجاه واحد لا اتجاهان: "Yes, it's seq2seq model but not bidirection for avoiding label leak."
- تصفير الحالة: ردّ الكاتب على "gru should be trained and states reseted every day?" →
  "Just zero inited at the beginning of every day."
- مدخل يوم واحد أفضل من عدّة أيام: "multi-days input instead of singe day input when applying GRU models"
  مدرَج تحت "What not worked for me".
- **Transformer**: 4 طبقات `torch.transformerencoder` على بُعد الأصول (انظر البند السابق).
- **TCN**: **لا ذكر لـ TCN.** وأقرب ما يُذكر: "ensemble with 1dCNN or MLP" مدرَج تحت
  "What not worked for me" — أي أن الالتفاف أحاديّ البُعد والـMLP **لم ينجحا** هنا.

### تجميع البذور (seed averaging)

- **لا ذكر لتجميع بذور.** التجميع بأوزان بين ثلاث معماريّات مختلفة:
  "with respective weights of 0.5, 0.3, 0.2 searched from validation set"
- ورفض للـstacking: "It's very time-consuming to train a second-level model and doesn't help much in fact. So i just use weighted sum."

### التعلّم المستمر — **المصدر الأقوى في هذا الملف**

- "Besides, online learning(OL) and post-processing(PP) also play an important role in my final submission."
- "I retrain my model every 12 days, 5 times in total."
- "Actually, my best submission is overtime at last update. I just skip online training if total inference time meets certain value. So there are 4 online training updates in total. I estimate that the best score would be around 5.400 if not overtime."
- تعريف الكاتب للمصطلح، ردّاً على Ziyi Dong: "It means retrain models with more test data in prediction phase."
- من أين يأتي الهدف في مرحلة التنبّؤ، ردّاً على wen6Lev57q4:
  "We can receive historical target info in inference phase, you can refer to this demo."
- **آليّة الإعادة تختلف بين النماذج**: سؤال Ayman Allawi: "Every 12 day you re-train the Catboost from scratch and fine-tune the two NNs am I right?"
  **وردّ الكاتب**: "Right. ~~~~"
  — أي CatBoost يُعاد تدريبه من الصفر، والشبكتان العصبيّتان تُصقَلان (fine-tune).
- **الأثر الرقمي للتعلّم المستمر** (من الجدول أعلاه، مقياس أقل-أفضل، على مجموعة الاختبار مع PP):
  - CatBoost: 5.4523 بلا OL → 5.4291 بـOL مرّة → 5.4165 بـOL خمس مرّات
  - GRU: 5.4690 → 5.4368 → 5.4259
  - Transformer: 5.4678 → 5.4493 → 5.4296
  - المجموع الثلاثي: 5.4438 → 5.4157 → 5.4030*(overtime)
- القيد كان الذاكنة ووقت الاستدلال: "Feature selection is important because we have to avoid memory error issue and run as many rounds of online training as possible."
  و"Because it requires double memory consumption when concat historical data with online data."
- حيلة تحميل البيانات المذكورة لحلّ ذلك: "The data loading trick can greatly increase this. For achieving this, you should save training data one file per day and also loading day by day."

```python
def load_numpy_data(meta_data, features):
    res = np.empty((len(meta_data), len(features)), dtype=np.float32)
    all_date_id = sorted(meta_data['date_id'].unique())
    data_index = 0
    for date_id in tqdm(all_date_id):
        tmp = h5py.File( '/path/to/{}.h5'.format(date_id), 'r')
        tmp = np.array(tmp['data']['features'], dtype=np.float32)
        res[data_index:data_index+len(tmp),:] = tmp
        data_index += len(tmp)
    return res
```

### ملاحظة على البند "4 sample weight"

- يظهر في النصّ عنوان "4 sample weight" بعد وصف النموذجين، **بلا أي محتوى بعده** — بند مبتور
  في المصدر الأصلي نفسه. لا أخمّن ما كان مقصوداً به.

---
## 5. Optiver Realized Volatility Prediction — 1st Place Solution - Nearest Neighbors

- **الرابط**: https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970
  (الصفحة تُحوَّل إلى `.../writeups/nyanp-1st-place-solution-nearest-neighbors` حسب الاقتباس المرجعي على الصفحة.)
- **الكاتب**: "nyanp" (المعرّف `nyanpn`)
- **التاريخ**: "Solution Writeup · 1st place · Jan 11, 2022" — وعلى رأس النصّ: "2022-01-11 Updated the title."
  والاقتباس المرجعي على الصفحة يذكر سنة "2021".
- **الحالة**: فُتحت بالكامل (النصّ + 143 تعليقاً). تصويت "501".

### الأرقام كما هي

- "Nearest neighbor aggregation features (boost from 0.21 to 0.19)"
- "I used a 4-fold time-series CV, with 10% of data used for validation for each fold."
- "I created about 600 features in total, 360 of which were Nearest Neighbor features, and most of my score improvement was based on these NN features."
- "So, I used NearestNeighbor with various distance metrics to find the similar N time-ids and calculate the average of features like RV and stock size (N=2,3,5,10,20,40)."
- "my NN model was a bit unstable in training, so I trained 10 models with different seeds and cherry-picked the top 5 models with the best validation scores for prediction"
- "It took 15s/epoch x 50epocs = 750sec to train singe NN."
- معاملات t-SNE من الكود: `TSNE(n_components=1, perplexity=400, random_state=0, n_iter=2000)`
- `n_max = 40` و`NearestNeighbors(n_neighbors=n_max, p=1)`
- "by 2020/1/1~2021/3/31, the stock market was open for 443 days, and there were 3833 time-ids during that period, which means that about 8.6 time-ids were recorded per day."
- "the time from 10:30~15:00 can be used for competition data. If they divided this time period into 30-minute intervals, up to 9 time-ids will be recorded per day."
- "correct direction of time-id order using known stock (id61 = AMZN)" — من تعليق الكود.
- من التعليقات (أرقام أطراف أخرى، لا الكاتب):
  - Jonathan Mallia (المركز 1922): "The NN features are the real deal, they have improve my single model from a local CV of 0.180396 to 0.177138."
  - Trushant Kalyanpur (المركز 18): "I can confirm that 0.19x can be broken without recovering time id. Our single model was public LB 0.18951 without time id recovery."
  - Damian (المركز 318): "even with knowledge of future time_id, this solution couldn't meet the hosts' expectation of a **good ** model at 0.15"

### الحفظ / overfitting

- **لا يستخدم الكاتب كلمة overfitting في نصّ الحلّ نفسه.** التنظيم عنده على مستوى التقييم لا النموذج:
  "Time series CV was especially important. I used a 4-fold time-series CV, with 10% of data used for validation for each fold. This allowed me to get a good enough (though not perfect) correlation between CV and LB throughout the competition."
- التحصين ضدّ الانزياح التوزيعي: "Now that we know the order of the time-id, we can detect features that are changing over time. By performing adversarial validation, I noticed that the features aggregated from trade.order_count and book.total_volume changed remarkably over time. Therefore, instead of using their raw features, I converted these features into ranks within the same time-id."
- "I also applied np.log1p to features that have large skew, as they may degrade the predictions if large outliers come during the 2nd stage."
- "These have little improvement on the LB scores, but I believe they help to reduce the risk of shakedown in private LB."
- "Detection of covariate shifts. Using methods such as adversarial validation enables us to find features that change over time."
- محاكاة الفجوة الزمنيّة صراحةً، من ردّ الكاتب على Jean-Michel Nairac:
  "There will probably be a 6-month gap between train-private tests. I created the same situation between train-validation to make sure that the NN features have no negative impact on the score in that situation. I think the model is robust to such long gaps because short-term NN features below 10-neighbors are dominant in the model."
- من تعليق khyeh (المركز 18): "I particularly enjoy reading this part, since we've been struggled to do a proper cv and need to verify on LB directly which might lead to overfitting in the end."

### dropout

- **لا ذكر لـ dropout إطلاقاً**، ولا أي نسبة.

### الضجيج على المدخلات

- **لا ذكر لضجيج على المدخلات.** والمُرمِّز التلقائي المُزيل للضجيج مدرَج تحت ما لم يُجرَّب:
  "What I thought might work, but didn't have time for ... Auto-encoder"
- ومن التعليقات، Rayan-aay (المركز 1162): "I already did test to add an Autoencoder to denoise the data, but it doesn't increase the score overall."
- ومن Carlo Lepelaars (المركز 425): "Very curious to see the result of denoising auto-encoder solutions that were so successful in the Jane Street competition."

### label smoothing

- **لا ذكر إطلاقاً.** (المهمّة انحدار على التقلّب المُتحقِّق.)

### حجم النموذج

- "I used three simple blends for prediction: LightGBM, 1D-CNN, and MLP."
- "The 1D-CNN is a simplified version of the architecture used in MoA 2nd place solution. (This CNN worked surprisingly well in both the recent MLB and Optiver competition.)"
- **أصغر من الأصل صراحةً**، من ردّ الكاتب على Tonghui Li (المركز 22):
  "It took 15s/epoch x 50epocs = 750sec to train singe NN. Perhaps because my CNN is much smaller than the original MoA's CNN."
- لا تُذكر أعداد طبقات أو قنوات أو معاملات.
- TabNet استُبعد لسبب زمني: "TabNet (not bad, but too long training time)" تحت "What didn't work".
- "I did not use the pretrained model because I wanted to use the test data for feature calculation, thus all the model training was done within a single notebook."

### طريقة التحقّق (purged / grouped CV)

- "Time-series cross-validation by reverse engineering of time-id order"
- "Time series cross-validation. Now that we know the correct order of the timestamps, we can construct the validation sets as if they were normal time-series data."
- "Time series CV was especially important. I used a 4-fold time-series CV, with 10% of data used for validation for each fold. This allowed me to get a good enough (though not perfect) correlation between CV and LB throughout the competition."
- **ليست purged ولا grouped CV بالاسم** — بل CV زمنيّ 4 طيّات بعد استعادة ترتيب الزمن بالهندسة العكسيّة.
  والملاحظة الجوهريّة أن ترتيب الزمن لم يكن معطى، فبُذل جهد لاستعادته **لأجل صحّة التحقّق أوّلاً**:
  "The prices in the competition data are normalized, but as someone pointed out in the discussion, you can use \"tick size\" to recover the real prices before normalization. Furthermore, by compressing the time-id x stock-id price matrix to one dimension using t-SNE, I was able to recover the order of the time-id with sufficient accuracy."
- "The correctness of the recovery of the time-id order for the training data can be easily verified by comparing it with the real market data. By doing so, we know that the training data is for the period between 2020/1/1~2021/3/31."
- "On the other hand, for the test data, I did not use the time-id information directly in the features and models because there is no guarantee that t-SNE will be able to sort the time-id order correctly."
- الكود الحرفي لاستعادة الترتيب:

```python
def calc_price_from_tick(df):
    tick = sorted(np.diff(sorted(np.unique(df.values.flatten()))))[0]
    return 0.01 / tick
```

```python
    # t-SNE to recovering time-id order
    clf = TSNE(
        n_components=1,
        perplexity=400,
        random_state=0,
        n_iter=2000
    )
    compressed = clf.fit_transform(
        pd.DataFrame(minmax_scale(df_prices.fillna(df_prices.mean())))
    )

    order = np.argsort(compressed[:, 0])
    ordered = df_prices.reindex(order).reset_index(drop=True)

    # correct direction of time-id order using known stock (id61 = AMZN)
    if ordered[61].iloc[0] > ordered[61].iloc[-1]:
        ordered = ordered.reindex(ordered.index[::-1])\
            .reset_index(drop=True)
```

### التضمينات (embeddings)

- **لا ذكر لطبقات embedding.**
- لكن من تعليق Michael Poluektov (المركز 7) تحذير يمسّ استخدام `stock_id` كميزة فئويّة:
  "Aggregating features by stock_id is a common tactic in top scoring public notebooks, but by doing that we are passing some future information to our model, same with using stock_id as a categorical feature, or doing any kind of stock_id clustering on test data"
- وتحذير أدقّ في التعليق نفسه: "Even aggregating by time_id can cause a leak, since seconds_in_bucket are not synchronised, part of the target variable of the stocks with more book updates can be influenced by an event in stocks with fewer book updates (if there are no book updates at the beginning of a 10 minute window, the whole window for that particular stock is shifted to start at the next book update)"

### الانتباه بين الأصول (cross-sectional / asset attention)

- **لا attention ولا معماريّة مقطعيّة.** لكن **المعالجة المقطعيّة موجودة كهندسة ميزات صريحة**:
  "In addition to the time-id, I also calculated the aggregation between similar stock-ids. Furthermore, by combining these ideas, I calculated features such as \"the average tau of 20 similar stocks with similar volatility in 5 closest time-ids\"."
- "If we generalize further, we can improve the prediction accuracy by using not only the next time-id, but also the information of time-ids that are \"similar\" in some distance metric. For example, the RV of the same stock when the market had similar price, volatility, and trading volume would be useful for predicting the RV at a certain time-id."
- الكود الحرفي لميزات الجوار:

```python
target_feature = 'book.log_return1.realized_volatility'
n_max = 40

# make neighbors
pivot = df.pivot('time_id', 'stock_id', 'price')
pivot = pivot.fillna(pivot.mean())
pivot = pd.DataFrame(minmax_scale(pivot))

nn = NearestNeighbors(n_neighbors=n_max, p=1)
nn.fit(pivot)
neighbors = nn.kneighbors(pivot)

# aggregate

def make_nn_feature(df, neighbors, f_col, n=5, agg=np.mean, postfix=''):
    pivot_aggs = pd.DataFrame(agg(neighbors[1:n,:,:], axis=0), 
                              columns=feature_pivot.columns, 
                              index=feature_pivot.index)
    dst = pivot_aggs.unstack().reset_index()
    dst.columns = ['stock_id', 'time_id', f'{f_col}_cluster{n}{postfix}_{agg.__name__}']
    return dst

feature_pivot = df.pivot('time_id', 'stock_id', target_feature)
feature_pivot = feature_pivot.fillna(feature_pivot.mean())

neighbor_features = np.zeros((n_max, *feature_pivot.shape))

for i in range(n):
    neighbor_features[i, :, :] += feature_pivot.values[neighbors[:, i], :]

for n in [2, 3, 5, 10, 20, 40]:
    dst = make_nn_feature(df, neighbors, feature_pivot, n)
    df = pd.merge(df, dst, on=['stock_id', 'time_id'], how='left')
```

- ترتيب مقطعيّ داخل نفس `time_id` كمعالجة للانزياح: "I converted these features into ranks within the same time-id."
- **تحذير قابليّة النقل الذي يخصّ التقييم** (بقلم الكاتب نفسه):
  "I don't think we can use future information in a real Optiver's scenario, but the idea of using Nearest Neighbor to aggregate nearby features can probably be used in a real model."
- وردّ الكاتب على سؤال Vu: "While this is 100% legal against the competition rules, unfortunately it is probably not what the host wanted us to do. On the other hand, it has happened repeatedly in Kaggle that in order to win the competition, we need to use methods that would probably not work in a real-world scenario. I think this mismatch should be reduced in the design of the competition (this is also why I love the time-series code competition). I believe that time/stock NN features are useful even if future information is not available (e.g. you can use information from past periods of similar volatility), so I still believe that the host can get some meaningful insights from my solution."
- وتعليق Lucas Morin (المركز 2832): "This is working because the test set is presented at once, including data from the future. It probably won't work so well in a real life setting."

### GRU / TCN / Transformer

- **لا Transformer ولا GRU ولا TCN.** المعماريّات: LightGBM + 1D-CNN + MLP.
- والنماذج المتكرّرة مدرَجة تحت ما لم يُجرَّب:
  "What I thought might work, but didn't have time for ... Ensembles with LSTMs and RNNs that do not create features"

### تجميع البذور (seed averaging) — **المصدر الأقوى في هذا الملف لهذا البند**

- "My NN model was a bit unstable in training, so I trained 10 models with different seeds and cherry-picked the top 5 models with the best validation scores for prediction (oops, I forgot to use seed=3047 :) )."
- **ملاحظة منهجيّة مهمّة**: هذا **ليس** تجميع بذور نظيفاً — إنه انتقاء (`cherry-picked`) لأفضل 5 من 10
  **على أساس درجات مجموعة التحقّق نفسها**. أي أن التحقّق استُخدم للاختيار، وهو مسار تسريب خفيف
  يرفع درجة التحقّق أعلى من الأداء الحقيقي المتوقَّع. الكاتب يذكر الأمر بلا تعليق على هذا الجانب.
- السبب المعلن هو التباين لا التحيّز: "My NN model was a bit unstable in training".

### التعلّم المستمر

- **لا تعلّم مستمر.** بل عكسه صراحةً — تدريب كامل داخل دفتر الإرسال:
  "I did not use the pretrained model because I wanted to use the test data for feature calculation, thus all the model training was done within a single notebook."
- والتنبّؤ دُفعيّ لا تسلسليّ، من ردّ الكاتب: "If I understand correctly, this will not be the case since the private test data will be processed in batch."

### ما لم ينجح — حرفياً

- "What didn't work / a lot of domain-specific features like beta coefficient / TabNet (not bad, but too long training time) / training NN on residual (target - book.log_return.realized_volatility) / dimensionality reduction features"
- "What I thought might work, but didn't have time for / 300-sec model (split the training data into the first and second halves, and create a model that predicts the RV of the second half from the first half and use it as meta-feature or data augmentation) / Ensembles with LSTMs and RNNs that do not create features / Auto-encoder"
- ردّ الكاتب على Carlo Lepelaars عن ميزات الـquarticity: "I tried features in that article, but unfortunately none of them worked."
- من تعليق AmbrosM: "I trained my LightGBM model on a multiplicative residual (target / book.log_return.realized_volatility). It gave a slight (i.e. insignificant) performance improvement over training on the target directly."
- من تعليق LongYin/杰少 عن نموذج 300 ثانية: "This helps."

### إسناد الفكرة

- من ردّ الكاتب: "I forgot to write something important: the idea that t-SNE can restore the order of time-ids came from this notebook. https://www.kaggle.com/stassl/recovering-time-id-order?scriptVersionId=71499310"
- وردّ Stas Sl: "I ended up using another dimensionality reduction method, because for some reason 1d TSNE didn't work for me very well."

---
## 6. Jane Street Real-Time Market Data Forecasting — لوحة المتصدّرين والحلول المنشورة

- **الرابط**: https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/leaderboard
- **الحالة**: فُتحت بالكامل.
- **وصف المسابقة على الصفحة**: "JANE STREET GROUP · FEATURED CODE COMPETITION · A YEAR AGO" —
  "Predict financial market responders using real-world data."
- **حالة اللوحة**: "The private leaderboard is calculated with all of the future data." و
  "This competition has completed. This leaderboard reflects the final standings."

### المراكز 1–5 كما هي (Private LB — "Prize Winners")

| # | Team | Score |
|---|---|---|
| 1 | ms capital | 0.013890 |
| 2 | Patrick Yam | 0.013273 |
| 3 | shorturl.at/LKhAD | 0.013163 |
| 4 | Haoze Hou | 0.011683 |
| 5 | hyd | 0.011449 |

- للسياق، المراكز 6–10 كما هي: 6 Thomas Dueholm Hansen 0.010675 / 7 leo 0.010480 /
  8 Evgeniia Grigoreva 0.010434 / 9 HAO LI 0.010417 / 10 ponythewhite 0.010293.
- **ملاحظة**: `hyd` صاحب المركز 5 هنا هو نفس كاتب حلّ المركز الأوّل في Optiver (القسم 4).

### روابط writeups للمراكز 1–5: **غير موجودة**

- المسار `/competitions/jane-street-real-time-market-data-forecasting/writeups` يُرجِع
  "We can't find that page."
- فحصت صفحة النقاش مرتَّبةً بالتصويت (`?sort=votes`، الصفحتان 1 و2). **لا يوجد أي منشور
  حلٍّ من أصحاب المراكز 1–5.** أعلى حلٍّ منشور هو المركز 8.
- الحلول المنشورة الظاهرة على الصفحتين، بمراكزها كما أعلنها كُتّابها:
  - "[Private LB 8th] solution" — Evgeniia Grigoreva — 295 تصويتاً —
    https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/writeups/evgeniia-grigoreva-private-lb-8th-solution
  - "[Public LB 17th] Solution" — snehal — 57 تصويتاً —
    https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/556541
  - "[Public LB 12th] Competition Wrap-up: Great Journey and Thank you all!" — SLi — 41 تصويتاً
  - "[Public LB 13th] Our Journey to 0.0096" — Maciej Zawadzki — 36 تصويتاً
  - "[Public LB 26th] TabM, AutoencoderMLP with online training & GBDT offline models" — I2nfinit3y — 33 تصويتاً
- منشور الحصيلة الرسمي من Kaggle لا يربط أي حلٍّ للفائزين — نصّه حرفياً:
  "Recap of Competition - Congratulations to the Winners!" بقلم "ELIZABETH PARK · A YEAR AGO · KAGGLE STAFF"، وفيه:
  "This competition ended with 25,703 registrations and 3,643 participants on 2,781 teams. We had 4,294 submissions from 102 countries. For 1,308 users (including 90 in the top 100!), this was their first competition."
  و"The top potential winning teams will be contacted via email for the next steps. We look forward to learning more about their winning solutions."
  و"We highly encourage you to post a solution write-up about your approach and solution in the forums (see instructions)."
  و"We've cleaned the leaderboard and disqualified some teams that have violated the rules."
- **تحذير على نتيجة بحث مضلِّلة**: بحثٌ على الويب عن "ms capital" يُرجِع مستودع
  `github.com/diegoarangureen/mscapital-market-forecasting` مع درجات مثل "0.110 (rank 197/206)".
  هذا **ليس** حلّ فريق `ms capital` صاحب المركز الأوّل — تشابه أسماء فقط، والأرقام لا تنتمي إلى
  هذه المسابقة. لم أفتح المستودع ولا أعتمده، وأسجّله هنا فقط لئلّا يُستشهَد به خطأً.
- **لم تُفتح**: لا توجد صفحات writeups للمراكز 1–5 لتُفتَح. ما لم يُنشَر لا أخمّنه.

---

## 6-ب. البديل الموثَّق: [Private LB 8th] solution — أعلى حلٍّ منشور في المسابقة نفسها

- **الرابط**: https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/writeups/evgeniia-grigoreva-private-lb-8th-solution
- **الكاتب**: "Evgeniia Grigoreva" (المعرّف `eivolkova`)
- **التاريخ**: "Solution Writeup · 8th place · Jul 15, 2025"
- **الحالة**: فُتحت بالكامل (النصّ + 52 تعليقاً). تصويت "295".
- **الكود منشور**: "Link to the code https://github.com/evgeniavolkova/kagglejanestreet" و
  "Link to the submission notebook https://www.kaggle.com/code/eivolkova/public-6th-place?scriptVersionId=217330222"

### الأرقام كما هي

- "I used a time-series CV with two folds. The validation size was set to 200 dates, as in the public dataset."
- "the model from the first fold was tested on the last 200 dates with a 200-day gap to simulate the private dataset scenario"
- "Fold 0: date_ids from 1298 to 1498." و"Fold 1: date_ids from 1499 to 1698."
- "I used data starting from date_id = 700, as this is when the number of time_ids stabilizes at 968."
- "I also selected 16 features that showed a high correlation with the target"
- "Rolling statistics: Rolling averages and standard deviations over the last 1000 time_ids for each symbol."
- "Adding these features resulted in an improvement of about +0.002 on CV."
- "The second model worked better than the first model on CV (+0.001)"
- "Adding auxiliary targets improved both CV and LB scores by about +0.001."
- "Models were trained using a batch size of one day, with a learning rate of 0.0005."
- "I ran both models on 3 seeds and took a simple unweighted average of predictions from those 6 models. This resulted in an LB score of 0.0112 (vs best single model LB 0.0105)."
- "I perform one forward pass to update the model weights with a learning rate of 0.0003. This approach significantly improved the model's performance on CV (+0.008)."
- "my tests suggested that it could provide a +0.001 improvement in the score" (عن إعادة التدريب الكامل)
- "it takes 0.06 seconds to run one inference step (time_id), 0.02 of which are spent on data processing. Updating model weights once per date_id takes 3.6 seconds."
- جدول الدرجات كما هو (الأعمدة: CV fold 0 / CV fold 1v / Fold 1 with 200 days gap / CV avg):

| | CV fold 0 | CV fold 1v | Fold 1 with 200 days gap | CV avg |
|---|---|---|---|---|
| GRU 1 without both auxiliary targets and online learning | 0.0161 | 0.0062 | 0.0011 | 0.0112 |
| GRU 1 without auxiliary targets | 0.0235 | 0.0148 | 0.0136 | 0.0190 |
| GRU 1 | 0.0249 | 0.0153 | 0.0147 | 0.0201 |
| GRU 2 | 0.0262 | 0.0166 | 0.0161 | 0.0214 |
| GRU 1 + GRU 2 | 0.0268 | 0.0169 | 0.0163 | 0.0218 |
| GRU 1 3 seeds | 0.0258 | 0.0164 | 0.0152 | 0.0211 |
| GRU 2 3 seeds | 0.0267 | 0.0175 | 0.0163 | 0.0221 |
| GRU 1 + GRU 2 3 seeds | 0.0270 | 0.0175 | 0.0162 | 0.0222 |

### الحفظ / overfitting

- **لا تُستخدم كلمة overfitting في هذا الحلّ.** لكن يوجد **أقوى رقم في هذا الملف على انهيار
  التعميم عبر فجوة زمنيّة**: الصفّ الأوّل من الجدول — GRU بلا أهداف مساعدة وبلا تعلّم مستمر —
  يسجّل `0.0161` على الطيّة 0 و`0.0062` على الطيّة 1 و**`0.0011` على الطيّة 1 مع فجوة 200 يوم**.
  أي أن الأداء يتبخّر تقريباً بمجرّد إدخال فجوة زمنيّة، رغم درجة تحقّق تبدو معقولة بدونها.
- والصفّ نفسه بعد إضافة الهدفين المساعدين والتعلّم المستمر (GRU 1): `0.0249 / 0.0153 / 0.0147` —
  أي أن عمود الفجوة يقترب من عمود الطيّة العاديّة، وهذا هو مقياس المتانة الحقيقي في الجدول.
- "Additionally, the model from the first fold was tested on the last 200 dates with a 200-day gap to simulate the private dataset scenario."
- "I experimented with using the entire dataset, but it did not result in any score improvement."
- "Simple standardization and NaN imputation with zero were applied. Other methods didn't provide any improvement."
- عن التطبيع، ردّ الكاتب على سؤال ZT ("Did you normalize per symbol or was it gloable normalization for the entire column?"): "Global standardization"

### dropout

- "1-layer GRU followed by 2 linear layers with ReLU activation and dropout."
- **لا تُذكر أي نسبة عدديّة لـ dropout في نصّ الحلّ.** النسبة قد تكون في المستودع المرتبط
  (`github.com/evgeniavolkova/kagglejanestreet`) — **لم أفتح المستودع** فلا أنقل منه رقماً.

### الضجيج على المدخلات

- **لا ذكر لضجيج مُضاف على المدخلات.** لكن يوجد ذكر للضجيج **في البيانات نفسها**، وصفاً للأهداف:
  "responder_6 is a 20-day rolling average of some variable, while responder_7 and responder_8 are 120-day and 4-day rolling averages of the same variable, with some added noise"

### label smoothing

- **لا ذكر إطلاقاً.** الخسارة انحداريّة: "The sum of losses (weighted zero-mean R²) for each responder was used to train the model."

### حجم النموذج

- "Time-series GRU with sequence equal to one day. I ended up with two slightly different architectures:
  3-layer GRU / 1-layer GRU followed by 2 linear layers with ReLU activation and dropout."
- **النموذج الأصغر (طبقة GRU واحدة) هو الأفضل**: "The second model worked better than the first model on CV (+0.001), but the first model still contributed to the ensemble, so I kept it."
- لا تُذكر أبعاد مخفيّة ولا عدد معاملات.
- طول التسلسل يوم واحد؛ وعلى سؤال Sumen Zhang عن OOM بطول 968 على RTX 4090 أجاب مشاركٌ آخر
  (Maciej Zawadzki، المركز 14) لا الكاتب: "evgeniia precomputes the features outside of torch. She also uses data with date_id >= 700. So that helps with memory." — وهذا استنتاجه من الكود لا تصريح الكاتب.

### طريقة التحقّق (purged / grouped CV)

- "I used a time-series CV with two folds. The validation size was set to 200 dates, as in the public dataset. It correlated well with the public LB scores."
- "Additionally, the model from the first fold was tested on the last 200 dates with a 200-day gap to simulate the private dataset scenario."
- **ليست purged ولا grouped CV بالاسم** — CV زمنيّ بطيّتين، **مع عمود فجوة (gap) منفصل بـ200 يوم
  كاختبار متانة**. وهذه الفجوة تلعب دور الـpurging وظيفياً وإن لم تُسَمَّ كذلك.
- "For submission, I trained models on data up to the last date_id, using the number of epochs equal to the average optimal number of epochs on CV."
  — أي لا إيقاف مبكر في الإرسال النهائي، بل عدد حِقب ثابت مأخوذ من متوسّط CV.

### التضمينات (embeddings) والانتباه بين الأصول — **أقوى دليل سلبيّ في هذا الملف**

- **الجملة الحرفيّة الكاملة**:
  "MLP, time-series transformers, cross-symbol attention and embeddings didn't work for me."
- أي أن **الانتباه بين الأصول (cross-symbol attention) والتضمينات (embeddings) جُرِّبا وفشلا**
  عند صاحب المركز الثامن في مسابقة مالية بمعطيات متعدّدة الأصول.
- البديل الذي نجح عنده هو **متوسّطات مقطعيّة يدويّة**:
  "Market averages: Averages per date_id and time_id."
- واستُبعدت الميزات الفئويّة: "I used all original features except for three categorical ones (features 09–11)."
- وأُضيف الزمن كميزة عدديّة: "Besides that, I added time_id as a feature."
- **لكن تنبيه مضادّ من المركز 6**: Thomas Dueholm Hansen (6th in this Competition):
  "Thanks for sharing! I did not even consider using GRU, so that was a bit of a blind spot for me. I briefly tried LSTM, but transformers worked much better for me."
  وردّ الكاتب: "I did exactly the opposite :) I hope to see your solution with transformers!"
  وردّ Thomas: "I am not currently planning to make a post about my solution, but I may change my mind if enough information becomes public knowledge."
  — أي أن **فشل الـTransformer ليس عامّاً**: مركزٌ أعلى نجح به. لا مصدر منشور لتفاصيله.

### GRU / TCN / Transformer

- **GRU هو العمود الفقري**: "Time-series GRU with sequence equal to one day."
- **Transformer فشل عند هذا الكاتب**: "time-series transformers ... didn't work for me" —
  **ونجح عند المركز 6** (انظر البند السابق).
- **MLP فشل بلا تعلّم مستمر أفضل، ومع التعلّم المستمر أسوأ**:
  "Interestingly, for an MLP model, the score without online learning was higher than for the GRU, but lower with online learning."
- **لا ذكر لـ TCN إطلاقاً.**
- من تعليق Sergei Fironov: "Interestingly, 3,700 teams participated in this competition. Surely many people have tried GRU. And I'm one of them. But I have a much worse results on LB for pure 3-layers RNN with the same feature enginering. Even though online learning provided about the same boost for me. I still haven't figured out what the secret is."
  — أي أن المعماريّة وحدها لا تفسّر النتيجة.

### الأهداف المساعدة (multi-task) — الكود الحرفي

```python
df = df.with_columns(
    (
        pl.col("responder_8")
        + pl.col("responder_8").shift(-4).over("symbol_id")
    ).fill_null(0.0).alias("responder_9"),
    (
        pl.col("responder_6")
        + pl.col("responder_6").shift(-20).over("symbol_id")
        + pl.col("responder_6").shift(-40).over("symbol_id")
    ).fill_null(0.0).alias("responder_10"),
)
```

- "I used 4 responders as auxiliary targets: responder_7 and responder_8, and two calculated ones"
- "These are approximate rolling averages of the base target over 8 and 60 days, respectively."
- "A separate base model was used for each auxiliary target. The predictions from these models were then passed through a linear layer to produce the final target output, responder_6."
- ولماذا نماذج منفصلة لا نموذج واحد متعدّد المخارج — ردّ الكاتب على 鸽鸽257 (المركز 17):
  "Because it worked better:) I suppose one model is not enough to fully capture different patterns associated with different responders"

### تجميع البذور (seed averaging)

- "I ran both models on 3 seeds and took a simple unweighted average of predictions from those 6 models. This resulted in an LB score of 0.0112 (vs best single model LB 0.0105)."
- **تجميع بسيط بلا أوزان وبلا انتقاء** — بخلاف حلّ Optiver في القسم 5 الذي انتقى أفضل 5 من 10.
- الأثر من الجدول: `GRU 1` وحده CV avg `0.0201` و`GRU 1 3 seeds` يعطي `0.0211`؛
  `GRU 2` وحده `0.0214` و`GRU 2 3 seeds` يعطي `0.0221`؛ والجمع الكامل `GRU 1 + GRU 2 3 seeds` يعطي `0.0222`.

### التعلّم المستمر — **المصدر الأقوى رقمياً في هذا الملف**

- "During inference, when new data with targets becomes available, I perform one forward pass to update the model weights with a learning rate of 0.0003. This approach significantly improved the model's performance on CV (+0.008)."
  — **+0.008 مقابل +0.002 لهندسة الميزات و+0.001 للأهداف المساعدة**: التعلّم المستمر هو
  أكبر مكسب منفرد في هذا الحلّ بفارق أربعة أضعاف عن أي عنصر آخر.
- "Updates are performed only with the responder_6 loss, without auxiliary targets."
- "Updates are applied for the entire dataset provided during submission, including rows with is_scored = False."
- معدّل التعلّم للتحديث `0.0003` أقلّ من معدّل التدريب `0.0005`.
- **تحديث بخطوة واحدة يوميّاً يكفي لسنة كاملة**:
  "I also considered performing a full online retraining on the data up to the start of the private dataset. This would make sense because there is a significant gap between the training data and the private dataset. However, retraining the model would require distributing the training process across multiple inference steps, as the one-minute time limit between dates would not be sufficient. I believe this would have been feasible but I decided not to spend time on it, although my tests suggested that it could provide a +0.001 improvement in the score. Still, I find it amazing that, instead of a full model retraining, performing one-day updates for almost a year is enough, and the model continues to perform well."
- **التفاعل بين المعماريّة والتعلّم المستمر**:
  "Interestingly, for an MLP model, the score without online learning was higher than for the GRU, but lower with online learning."
  — أي أن ترتيب المعماريّات **ينقلب** حسب وجود التعلّم المستمر. مقارنة معماريّات بلا تعلّم مستمر
  تعطي ترتيباً مضلِّلاً.
- التكلفة الزمنيّة: "Updating model weights once per date_id takes 3.6 seconds."

### الأساس التقني كما هو

- "I used PyTorch, but since TensorFlow is said to be faster, I tried switching to it. However, after a few days of experimenting, I couldn't achieve better performance, so I decided to stick with PyTorch."
- "Due to RAM requirements, I switched from Google Colab to vast.ai"
- "I also used WandB to monitor experiments, which helped me keep track of scores and easily revert to an older version of the code if something went wrong."
- "To debug my submission notebook and estimate submission time I used synthetic dataset by @shiyili."

### شهادة من صاحب المركز 4 في التعليقات

- Haoze Hou (4th in this Competition): "Thanks for sharing! Will the engineered features improve the performance in LB? I have tried to do some feature engineering, it has slightly improved the CV but no boost on LB."
- وردّ الكاتب: "Yes, they improved both LB and CV scores significantly"
- **أي تعارض صريح بين المركزين 4 و8 على قيمة هندسة الميزات.** لا حلّ منشوراً للمركز 4 لترجيحه.

---
## 7. G-Research Crypto Forecasting — أفضل ثلاثة حلول منشورة

- **الرابط**: https://www.kaggle.com/competitions/g-research-crypto-forecasting/discussion
- **الحالة**: فُتحت. الحلول المنشورة الموجودة على الصفحة (مرتَّبةً بالتصويت) هي أربعة فقط:

| المركز المعلن | العنوان | الكاتب | الرابط |
|---|---|---|---|
| 2 | 2nd place solution | Nathaniel Maddux | `/writeups/nathaniel-maddux-2nd-place-solution` |
| 3 | 3rd place solution | sugghi | `/writeups/gaba-3rd-place-solution` |
| 7 | 7th place solution | Patrick Yam | `/writeups/patrick-yam-7th-place-solution` |
| 13 | 13th Place (Final) 1st Place (6 Weeks In) Final Solution | Tom Forbes | `/writeups/tom-forbes-13th-place-final-1st-place-6-weeks-in-f` |

- **لا يوجد حلٌّ منشور للمركز الأوّل.** الفريق الأوّل، بتسمية صاحب المركز الثاني حرفياً:
  "Meme Lord Capital" — و"They finished with a big lead on everyone."
- أفضل ثلاثة حلول منشورة إذن: المراكز **2 و3 و7**، وهي المذكورة أدناه. (المركز 13 لم أفتحه —
  خارج نطاق "أفضل 3".)

---

### 7-أ. المركز الثاني — Nathaniel Maddux

- **الرابط**: https://www.kaggle.com/competitions/g-research-crypto-forecasting/writeups/nathaniel-maddux-2nd-place-solution
- **الكاتب**: "Nathaniel Maddux" (المعرّف `nathanrm`)
- **التاريخ**: "Solution Writeup · 2nd place · Nov 10, 2022"
- **الحالة**: فُتحت بالكامل (النصّ + 25 تعليقاً). تصويت "86".
- **قيد معلن على الشفافيّة**: "In this writeup I'll strive to provide insight into my methods, but without giving away model details that could be profitable for the host. When a good financial indicator becomes common knowledge, everyone uses it, therefore it loses profitability. As such, I won't be sharing my code or any specific insight on features."

#### الأرقام كما هي

- "In this competition I used 6-fold, walk-forward, grouped cross validation. The group key was the timestamp. In a typical setup, train folds were 40 weeks long, test folds were 40 weeks long, there was a gap of 1 week between test and train folds, and the ends of training folds were incremented by 20 weeks each fold."
- "By my estimate, there was a 60% chance I would drop below 2nd on the 6th update, due to the noisy nature of this kind of data. I also estimate that on a 7th update, Meme Lord Capital would have a 75% chance of maintaining first place."
- "I did not expect true scores to go above 0.1, based on experience and discussions here. Plotting the sorted scores of the \"low\" scoring masters and grandmasters, there was a clear plateau around 0.08, as I recall. That was the score I aimed to beat in my local CV."
- "I only made a total of two submissions in the competition."
- "Good N-fold CV is easy to set up, taking less than 100 lines of code, without using scikit-learn or any timeseries framework."
- "Our kernels were limited to running in 9 hours or less, using up to 16GB of RAM."
- "It takes 25 minutes to import data, generate features, and train the model. When submitting, predictions are generated at a rate of about 50 timestamps per second, not including the overhead of the submission mechanism. Generating predictions for 2 weeks of data takes around 10 minutes, on top of the the 25 minutes to train the model."
- "In 2014-2017, I researched stock price analysis and prediction in my own time."

#### الحفظ / overfitting

- "Because public leaderboard scores were unrealistically high due to probing and overfitting, I had to find a way to interpret them."
- "So I looked at masters and grandmasters with \"low\" scores. I figured these competitors were good enough to submit good kernels, but had been careful to keep the public leaderboard period out of their training data."
- "Of course, I did not use the public leaderboard as a guide in any optimizations. That's always a bad idea (see the paragraph above about CV variance)."
- "The one week gap between train and test data was to prevent CV results from being too rosy. With no gap, a model can cheat at the the beginning of the test period, because the end of the train period is very similar. There was no gap between train and test data in the final submission."
- **تحذير تباين CV** (نصّ كامل حرفي): "(If you don't pay attention to CV variance, you could waste days optimizing your model, only to find that your early decisions were based on noise and incorrect. An easy way to see if your CV scores have too much variance is to look at a plot of CV score vs. a parameter you're tuning. A good plot will usually be smooth, with a knee and a plateau, or maybe a peak or a valley. If the plot looks too noisy to clearly see those things, then try it again, with different seeds, and see if you get different results. If you do, then your CV results are too noisy. You can try using more folds, or running CV several times with different random seeds and averaging.)"
- "Setting up good N-fold cross validation (CV) is essential. We should try to see if our ideas fail in the most real environment we can make. Every time we have a research question, it should (eventually, after a period of creative exploration) be answered in that manner. If we don't have a good CV setup, our decisions will mostly be mere guesses."
- عن التسلسل السببي، من ردّ الكاتب على PsiqueR: "Timeseries learning is essentially supervised learning that respects causality. You avoid doing anything in your algorithm that would depend on data from the future. That's both harder and easier than it sounds. Technically, it's easy because you're avoiding doing something. In practice, you can sometimes take shortcuts and make mistakes that end up causing your algorithm to need future data. Learning how to avoid these pitfalls is one of the main things to learning timeseries forecasting. You want to be as creative as possible, clustering assets by market, etc., without making these mistakes and breaking causality."

#### dropout / الضجيج على المدخلات / label smoothing

- **لا ذكر لأيٍّ منها.** والتنظيم كلّه مُلغى صراحةً:
  "There was no regularization, augmentation, or feature neutralization. I checked with CV whether these things would help, found that they wouldn't, and decided to go with a simpler submission."
  — **هذه الجملة دليل مباشر ضدّ تعميم "التنظيم ضروري دائماً"**: نموذج شجريّ بلا تنظيم ولا
  توسيع بيانات ولا تحييد ميزات حلّ ثانياً، والقرار مبنيّ على CV لا على تفضيل.
- "LightGBM parameters were not tuned exhaustively. Tuning was taking a long time, and CV results were indicating that regularization would have little effect on model performance. So I decided to not push on that wall and instead placed my focus back on feature engineering. CV is essential in making these resource allocation decisions."

#### حجم النموذج

- "The learner was trivial: a LightGBM GBDT regressor with squared loss. There was no ensembling other than the gradient boosting in GBDT. The only parameters I changed from defaults were the number of estimators, number of leaves, and the learning rate."
- لا تُذكر القيم العدديّة لتلك المعاملات الثلاثة.
- "I wanted to train with the entire dataset, because CV had shown me that scores just kept improving with longer training data."

#### طريقة التحقّق (purged / grouped CV) — **أوضح وصف لـgrouped CV في هذا الملف**

- "In this competition I used 6-fold, walk-forward, grouped cross validation. The group key was the timestamp."
- **الطيّات متداخلة عن قصد**، والسبب حرفياً:
  "I chose to overlap my folds so that they could be long, but I could still have 6 folds. With non-overlapping folds, I would have to decide between many short folds or a few long ones."
  و"The advantage of having many folds is that average CV scores will have lower variance. On the other hand, the advantage of having long folds is that the model sees more data, so you get a better picture of how it will perform with the full dataset. Instead of choosing between many short folds or a few long ones, I let them overlap so I could have many long folds."
- "I could have used more than 6 folds, or run CV multiple times, changing seeds. I didn't do these things because CV score variance was decently low, and CV was already almost too slow."
- **الفجوة (أسبوع) هي المُكافئ الوظيفي للـpurging**: "The one week gap between train and test data was to prevent CV results from being too rosy."

#### التضمينات / الانتباه بين الأصول

- **لا ذكر لأيٍّ منهما صراحةً.** وسؤال Lucas Morin عن معالجة هويّة العملة:
  "How do you handle coind id ? (one model for all coin ? with the coin as a feature ? or individual models ?)"
  **وردّ الكاتب**: "Good questions. You too guessed a topic I decided to leave out. I tried all of it. CV and feature importance guided the way."
  — **رفض الإجابة صراحةً.** لا أستخلص موقفاً من هذا الردّ.
- وعلى سؤال آخر عن اختيار الميزات: "That's a good question. You guessed a topic that I decided to leave out. It was more sophisticated."

#### GRU / TCN / Transformer

- **لا ذكر لأيٍّ منها كجزء من الحلّ.** والشبكات العصبيّة مدرَجة تحت "ما لو توفّر وقت أكثر":
  "I'd try more complex models, especially using autoencoders, convolutional NN's, recursive NN's, etc."
- ملاحظة نظريّة من الكاتب في السياق نفسه: "Essentially, I'm thinking of supervised learning problems where the data is nonlinear, high dimensional and has a lot of noise. There are toy datasets that seem like they should be easy to fit, but quickly become challenging as noisy dimensions are added. Neural nets do the best on these toy datasets, with decision forest methods coming in a distant second, and basically every other method fails with even a few dimensions."

#### تجميع البذور / التعلّم المستمر

- **لا تجميع بذور**: "There was no ensembling other than the gradient boosting in GBDT."
  ومع ذلك يوصي بتغيير البذور **لتقليل ضجيج التقييم** لا لتجميع التنبّؤات:
  "You can try using more folds, or running CV several times with different random seeds and averaging."
- **لا تعلّم مستمر.**

#### ما عزاه الكاتب للنجاح — حرفياً

- ردّ الكاتب على سؤال Fritz Cremer ("how much do you think your success can be attributed to model choice/hyperparameter choice/cross validation setup, and how much is feature engineering?"):
  "It was mostly feature engineering, and yes, finding the right features was key."
- "As I developed my model, feature engineering was guided by feature importance. As I made new features, I focused further effort on developing features sets that already performed well (had high importance), or for which transformations led to easy increases in importance."
- "All the way up to the deadline, feature engineering was the best avenue for improvement."
- "I'd revisit some decisions I made based on CV. With more computational resources, CV scores would have less variance, and I might be able to see, for example, that regularization actually improved the model a little."
  — **أي أن استنتاج "لا حاجة للتنظيم" هو نفسه محدود بتباين الـCV، بشهادة الكاتب.**

---

### 7-ب. المركز الثالث — sugghi

- **الرابط**: https://www.kaggle.com/competitions/g-research-crypto-forecasting/writeups/gaba-3rd-place-solution
- **الكاتب**: "sugghi"
- **التاريخ**: "Solution Writeup · 3rd place · May 24, 2022"
- **الحالة**: فُتحت بالكامل (النصّ + 27 تعليقاً). تصويت "66".
- **الكود منشور**: "training: https://www.kaggle.com/code/sugghi/training-3rd-place-solution"
  و"inference: https://www.kaggle.com/code/sugghi/inference-3rd-place-solution"
  (**لم أفتح الدفترين** — النقل أدناه من الكتابة والتعليقات فقط.)

#### الأرقام كما هي

- "Single model of LightGBM (7-fold CV)"
- "I attempted forward fill to prevent missing data as a result of rolling. On the other hand, I thought that forward fill for the entire period might cause a decline in data quality when there is a long blank period, so I set a limit on forward fill."
  (لا تُذكر قيمة الحدّ عدديّاً.)
- "managed to finish the inference within 9 hours"
- "The first successful submission was a week before the end of the competition"
- "it was the third machine learning challenge in my life"

#### الحفظ / overfitting

- **لا تُستخدم كلمة overfitting.** لكن يوجد **إقرار صريح بخطأ منهجيّ في المقارنة**:
  "The selection of the starting date was done by looking at the CV scores. However, this was a mistake in hindsight, since it meant that I was comparing CV scores across different data."
  — أي أن ضبط نطاق البيانات على أساس درجة CV يُبطِل قابليّة المقارنة بين الدرجات.
- وإقرار بأثر الحظّ: "since this is a competition with a strong element of luck, it was only by sheer luck that I was able to finish in third place, but I am glad that my rank was somewhat stable in the last few updates."
- "I would like to continue to improve so that I can achieve results in competitions where the element of luck is small."
- والاتّساق بين التدريب والتقييم كان اعتباراً صريحاً:
  "Since my model deals with the average of the changes of each currency, I considered it undesirable for the existing currencies to differ significantly between the training period and the evaluation period. As I expected that all currencies would have few missing values during the evaluation period of the competition, I decided not to use all of the train data, but to use the data from the period when there were enough currencies present."
- وتباين بين مرحلتَي التدريب والاستدلال في الـforward fill:
  "In the evaluation phase, the code was designed to have forward fill without a limit, but I thought this would not be a problem since there are no long blank periods in the evaluation phase."

#### dropout / الضجيج على المدخلات / label smoothing / حجم النموذج

- **لا ذكر لأيٍّ منها.** النموذج: "Single model of LightGBM (7-fold CV)" بلا معاملات معلنة،
  و"Parameter optimization" مدرَج تحت ما لم يُنجَز: "What I would have worked on if I had more time ... Parameter optimization"
- "Even so, the ensemble could not be performed because of the limited inference time (and lack of coding skill)."

#### طريقة التحقّق (purged / grouped CV)

- "For CV, I used EmbargoCV by @nrcjea001."
  — **EmbargoCV** أي تقسيم مع فترة حَجْر (embargo)، وهو التسمية الأخرى لمنطق purging.
  والمصدر منسوب لمشاركٍ آخر (`@nrcjea001` — وهو نفسه Jean-Michel Nairac الذي سأل عن
  مخاطر المستقبل في القسم 5).
- "Single model of LightGBM (7-fold CV)"

#### التضمينات (embeddings)

- **لا ذكر لطبقات embedding**، لكن **نموذج منفصل لكل عملة بميزات مشتركة**:
  "The model is trained for each coin using a common set of features for all the coins."
  — أي أن هويّة الأصل تُعالَج بفصل النماذج لا بتضمين.

#### الانتباه بين الأصول (cross-sectional) — **معالجة مقطعيّة صريحة بلا attention**

- "The difference between the change of each currency and the change of all currencies is provided as features."
- الدافع حرفياً: "Considering the definition of the forecasting target in this competition, I felt it was necessary to prepare features with information about the entire market. I also thought that some currencies might be affected by the movements of other currencies, so I made it possible to refer to information about other currencies as well."
- التفصيل: "For 'Close', I prepared two features for multiple lag periods: the log of the ratio of the current value to the average during the period, and the log of the ratio of the current value to the value a certain period ago. For these, I took the average for all currencies (Due to missing data, no weighted averaging was performed). In addition, the difference between each currency and the average of all currencies was also prepared as a feature. As a result, this feature seems to have worked well."
- **تقليل جذريّ في المدخلات**: "Only 'Close' is used." و
  "Specifically, I considered 'Close', which is used to calculate the target, to be the most important, so I decided to use only it."
  — أي أن المركز الثالث بُنِي على عمود سعر واحد + معالجة مقطعيّة، بدافع الذاكرة وزمن الاستدلال:
  "Since I thought that memory and inference time would become more demanding with this kind of processing, I reduced the amount of data to be used."

#### الكود الحرفي من التعليقات

- سؤال theDoors عن سطر في كود التدريب:

```python
df[f'log_close/mean_{lag}id{id}'] = np.log( np.array(df[f'Close{id}']) / np.roll(np.append(np.convolve( np.array(df[f'Close_{id}']), np.ones(lag)/lag, mode="valid"), np.ones(lag-1)), lag-1) )
```

- وردّ الكاتب بالمُكافئ الأبسط والسبب:
  "I used such confusing code to avoid using pandas to accelerate inference. The original and simpler code was as follows;"

```python
df[f'log_close/mean_{lag}_id{id}'] = np.log( df[f'Close_{id}'] / df[f'Close_{id}'].rolling(lag).mean() )
```

- **علّة مُبلَّغ عنها في هذا الكود بعينه** (من معلّقَين، لا من الكاتب — ولا ردّ منه):
  Volgoesupndown: "Stumbled upon this, doesn't your numpy version introduce a bug? roll != shift, as roll will not introduce NaNs but instead roll values from the end of the array to the front."
  و T!: "Yes, the first |lag| values will use the last |lag| values 🫤"
  — **أي أن النسخة المُحسَّنة بـnumpy تُدخِل قيماً من نهاية المصفوفة إلى أوّلها**، وهذا تسريب
  من المستقبل في أوّل `lag` صفّاً. لم يُردّ عليه الكاتب. **أسجّله كادّعاء من معلّقَين لا كحقيقة مؤكَّدة.**

#### GRU / TCN / Transformer / تجميع البذور / التعلّم المستمر

- **لا ذكر لأيٍّ منها.**

---

### 7-ج. المركز السابع — Patrick Yam

- **الرابط**: https://www.kaggle.com/competitions/g-research-crypto-forecasting/writeups/patrick-yam-7th-place-solution
- **الكاتب**: "Patrick Yam" (المعرّف `wimwim`)
- **التاريخ**: **لا تاريخ على الصفحة** — الترويسة هي "Solution Writeup · 7th place" بلا تاريخ،
  بخلاف بقيّة الكتابات. الاقتباس المرجعي يذكر سنة "2022" فقط. لا أُسنِد له يوماً.
- **الحالة**: النصّ فُتح (قصير جداً) + 29 تعليقاً. تصويت "74".
- **ملاحظة سياقيّة**: `Patrick Yam` هو نفسه صاحب **المركز الثاني** في مسابقة Jane Street 2024 (القسم 6).

#### **لم تُفتح**: المخطّط المعماري

- وصف النموذج بالكامل في صورة خارجية على `https://i.imgur.com/ECta6Oe.png` (أبعادها 421×601).
  **محاولة قراءتها فشلت** (الموقع يطلب موافقة لكل فعل ولا يسمح بأدوات القراءة).
  **لم تُفتح — ولا أخمّن محتوى المخطّط.** كلّ ما يلي من النصّ والتعليقات فقط.
- الكاتب يؤكّد أن الصورة هي المحتوى: "No, I didn't delete any content. If you can't see the picture, it could be because Imgur is blocked by your internet provider."

#### الأرقام كما هي

- "My approach is 99% on modeling and the only feature I added is the time of day."
- "the final submission is an ensemble of 4 models trained with different sequence lengths"
- "I use 3-fold Grouped CV and the key is timestamps."
- عن التطبيع: "I decided to normalize input using only the data from the input sequence with the same asset (90 data points for each feature) to compute the mean and std."
- وفي ردٍّ آخر: "I use standardization for input scaling. I find that the features scaling could be very different between different assets (e.g. ETH vs BTC) and different timestamps (2020 vs 2021), so I calculate the mean and std for each asset and only use the data from the input sequence (90 data points) to make the model more stable and robust."
- **شكل المُدخل `(90, 14, 9)`** — لكن هذا من **سؤال معلّق** (Vitor): "I was wondering how many training samples in total did you have for the (90, 14, 9) tensors. Did you consider the whole span of years (2018-2021) or only the most recent period?"
  **ولم يردّ الكاتب على هذا السؤال.** أسجّله كرقم ورد في سؤال معلّق مطابقاً لـ90 نقطة التي أكّدها
  الكاتب، لا كتصريح منه بالأبعاد الثلاثة.

#### الحفظ / overfitting

- **لا ذكر لكلمة overfitting.** والمتانة مذكورة في سياق التطبيع فقط:
  "to make the model more stable and robust."
- **التطبيع داخل النافذة لا عبر التاريخ** هو الفكرة المنهجيّة الوحيدة الصريحة هنا، وسببها معلن:
  "I find that the features scaling could be very different between different assets (e.g. ETH vs BTC) and different timestamps (2020 vs 2021)"
  — أي معالجة انزياح التوزيع بالتطبيع المحلّي، بديلاً عن تطبيع عالمي.
  (قارِن بالقسم 6-ب حيث اختار المركز الثامن "Global standardization" — **قراران متعاكسان**.)

#### dropout / الضجيج على المدخلات / label smoothing / حجم النموذج

- **لا ذكر لأيٍّ منها في النصّ أو التعليقات.** أرقام الطبقات والأبعاد (إن وُجدت) في المخطّط
  الذي **لم تُفتح** صورته.

#### طريقة التحقّق

- "I use 3-fold Grouped CV and the key is timestamps."
- **لا ذكر لـpurging ولا embargo ولا فجوة.** (قارِن بالمركز الثاني الذي وضع فجوة أسبوع،
  والمركز الثالث الذي استخدم EmbargoCV — ثلاثة مراكز، ثلاث صرامات مختلفة.)

#### التضمينات / الانتباه بين الأصول / GRU / TCN / Transformer

- **الطبقات المذكورة اسماً** في ردّ الكاتب على سؤال عن النصيحة للمبتدئين:
  "I always start by checking the data shape (i.e. is the data tabular? 1d? or 2d?), then pick some suitable neural network layers (e.g. LSTM/ Conv/ MHA) for it. I would say Transformer is a very strong candidate for data >= 1D, definitely give it a try."
  — `MHA` = Multi-Head Attention. وهذه **توصية عامّة** من الكاتب، لا تصريح بأن حلّه استخدمها.
- **حدّ الـTransformer الذي يذكره الكاتب صراحةً**:
  "I will perform feature engineering only when the model cannot 'see' the information from the input. For example, Transformer can't handle input with a very long sequence length, so we will have to create features to capture long-term information."
  — أي أن هندسة الميزات عنده بديل عن طول تسلسل لا يحتمله الـTransformer، لا غاية بذاتها.
- **لا ذكر لـ GRU ولا TCN.**
- شهادة صاحب المركز الثاني على اختلاف المنهج: Nathaniel Maddux (2nd in this Competition):
  "Wow, that's so different from what I did, really cool."
- وتعليق MarkFromItaly (المركز 245): "I am really amazed when I see solutions without heavy feature engineering that use well designed deep learning models."

#### تجميع البذور

- **ليس تجميع بذور، بل تجميع أطوال تسلسل**:
  "the final submission is an ensemble of 4 models trained with different sequence lengths"

#### التعلّم المستمر

- **لا ذكر إطلاقاً.**

---

---

## 8. Numerai Forum — "Feature Neutralization Increases Bias and Reduces Variance"

- **الرابط**: https://forum.numer.ai/t/feature-neutralization-increases-bias-and-reduces-variance/7486
- **الكاتب**: `by256` (ردٌّ واحد من `andralienware`)
- **التاريخ**: 4 يونيو 2024 — عدد المشاهدات المعروض: `1.4 ألف`، الردود: `2`

### الحفظ / overfitting — الشبكات العصبية تحديداً

> "On the other hand, if you use neural networks which typically overfit due to their high variance, you might use a higher neutralization proportion to balance this out."

> "The results show that as the neutralization proportion increases, bias increases and variance decreases. The sweet spot for balancing bias and variance seems to be around 𝑝 = 0.5 for the model that I used."

الرقم كما ورد: `p = 0.5` — وهو نسبة تحييد (neutralization proportion)، **لا** معدّل dropout ولا نسبة ضجيج.

> "For example, when using random forests which are known to be variance reducing, you could use a lower neutralization proportion as the model is already lower variance."

### طريقة التحقّق (validation)

> "This involved fitting many models on eras that were randomly sampled from the training set, and using the validation set to calculate bootstrap estimates of bias and variance from the resulting predictions."

التقسيم هنا **per-era** وبإعادة معاينة عشوائية للعُصور (eras)، وليس purged/grouped CV باسمه الصريح:

> "Feature neutralization is done on a per-era basis."

### تعريف التحييد وصيغته (نُقلت حرفياً)

> "Feature neutralization is a technique that is commonly used in the Numerai tournament to reduce exposure to features that a model relies heavily on when making a prediction."

> "In the forum and numerai_tools library, you'll see feature neutralization implemented as something like 𝑦pred − 𝑝⁢𝑋⁢(𝑋𝑇⁢𝑋)−1⁢𝑋𝑇⁢𝑦pred"

الكود الوارد في المنشور (pseudo code) كما هو:

```python
# Fit model
model.fit(X_train, y_train)
# Predict on validation set
y_pred = model.predict(X_val)
# Calculate feature exposures and select the most exposed features
exp_feas = calculate_feature_exposure(X_val, y_pred)
# Neutralize the predictions (p is a neutralization proportion between 0 and 1)
pred_neutralized = y_pred - p*LinearRegression.fit(exp_feas, y_pred).predict(exp_feas)
```

### ملاحظة على التنفيذ (نقد الكاتب لمكتبة numerai_tools)

> "I think the implementation as it exists in numerai_tools is quite unfortunate. In practice, you would almost never solve a linear equation by inverting 𝑋𝑇⁢𝑋 as this is slow, can lead to numerical instabilities and the inverse will not exist if the columns of 𝑋 are dependent."

> "In practice, these kinds of equations are solved by optimization and I found a while ago that doing this made feature neutralization twice as fast."

ردّ `andralienware`:

> "@bluesapphire mentioned np.linalg.solve() last time you posted this, but your solution is better exactly because of the case where 𝑋𝑇⁢𝑋 is singular–even the numpy documentation points out this case and refers users to the lstsq function to handle this. There is no reason not to have the pinv solution instead of the lstsq solution."

### ما **لا** يوجد في هذا المصدر

المنشور لا يذكر: dropout، ولا الضجيج على المدخلات (GaussianNoise)، ولا label smoothing، ولا حجم النموذج، ولا التضمينات، ولا الانتباه بين الأصول، ولا GRU/TCN/Transformer، ولا تجميع البذور، ولا التعلّم المستمر. صلته بـ PR7 محصورة في: **الشبكات العصبية عالية التباين وتنزع إلى overfitting**، وفي تقنية التحييد ونسبتها.

---

## 9. Numerai Forum — "AutoEncoder and multitask MLP on new dataset (from Kaggle Jane Street)"

- **الرابط**: https://forum.numer.ai/t/autoencoder-and-multitask-mlp-on-new-dataset-from-kaggle-jane-street/4338
- **الكاتب الأصلي**: `jrai` — والمشاركون: `yxbot`, `hedgingcat` (صاحب حلّ Jane Street الفائز نفسه), `sunkay`, `perfect_fit`, `luee`, `danzell`, `stoicism`, `gbrecht`, `olivepossum`, `maxchu`
- **التاريخ**: 15 أكتوبر 2021 (الردود تمتدّ إلى يناير 2022)
- **المشاهدات المعروضة**: `10.3 ألف` — `67` إعجاباً، `31` منشوراً، `12` مستخدماً

### الضجيج على المدخلات + dropout + منع overfitting (نقل حرفي لوصف Yirun Zhang)

المنشور ينقل وصف صاحب المركز الأول في Jane Street بعد تعديله ليلائم Numerai:

> "Deep Learning Model:
>
> Use autoencoder to create new features, concatenating with the original features as the input to the downstream MLP model
> Train autoencoder and MLP together
> Add target information to autoencoder (supervised learning) to force it to generate more relevant features, and to create a shortcut for backpropagation of gradient
> Add Gaussian noise layer before encoder for data augmentation and to prevent overfitting
> Use swish activation function instead of ReLU to prevent 'dead neuron' and smooth the gradient
> Batch Normalisation and Dropout are used for MLP
> Only monitor the MSE loss of MLP instead of the overall loss for early stopping"

> "The author of the initial code explains 'The idea of using an encoder is to denoise the data.'"

### الأرقام كما هي — hyperparameters (هذه أهمّ فقرة في هذا المصدر)

من الكود المنشور لنسخة Numerai (Keras):

```python
params = {
    "num_columns": len(feature_names),
    "num_labels": len(targets),
    "hidden_units": [96, 96, 896, 448, 448, 256],
    "dropout_rates": [
        0.035,
        0.035,
        0.4,
        0.1,
        0.4,
        0.3,
        0.25,
        0.4,
    ],
    "lr": 1e-4,
}
```

وموضع كل رقم في المعمارية (منقول من `create_architecture`):

```python
encoder = tf.keras.layers.GaussianNoise(dropout_rates[0])(x0)     # 0.035  → سيغما الضجيج الغاوسي على المدخل
decoder = tf.keras.layers.Dropout(dropout_rates[1])(encoder)      # 0.035
x_ae    = tf.keras.layers.Dropout(dropout_rates[2])(x_ae)         # 0.4
x       = tf.keras.layers.Dropout(dropout_rates[3])(x)            # 0.1  ← بعد دمج x0 مع encoder
x       = tf.keras.layers.Dropout(dropout_rates[i + 2])(x)        # 0.4, 0.3, 0.25, 0.4 لطبقات MLP
```

**نقطة جوهرية للتحقّق**: القيمة التي تُطبَّق على **المدخل** هي `dropout_rates[0] = 0.035`، وهي تُمرَّر إلى `GaussianNoise` — أي **سيغما ضجيج = 0.035**، وليست dropout. وأول dropout فعليّ (`dropout_rates[1] = 0.035`) يقع **بعد** المشفّر لا على المدخل. (انظر القسم 10 لعلاقة هذا بادّعاء «dropout 35% على طبقة الإدخال».)

- حجم النموذج كما ورد: `hidden_units = [96, 96, 896, 448, 448, 256]` ومعدّل التعلّم `lr = 1e-4`.
- الأهداف الستّة: `target_nomi_20`, `target_jerome_20`, `target_janet_20`, `target_ben_20`, `target_alan_20`, `target_paul_20`.
- ثلاث دوالّ خسارة، كلّها `MeanSquaredError` على `decoder` و`ae_targets` و`targets`.

### نسخة PyTorch من `olivepossum` (1 يناير 2022) — نفس الأرقام

```python
hidden_units = [96, 96, 896, 448, 448, 256]
dropout_rates = [0.035, 0.035, 0.4, 0.1, 0.4, 0.3, 0.25, 0.4]
lr = 0.0001
```

وصنف الضجيج الغاوسي عندها يوضّح أن الرقم سيغما نسبية:

```python
class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=True):
        ...
        scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
        sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
        x = x + sampled_noise
```

وعن قابلية إعادة إنتاج النتائج، سأل `maxchu`: "Can you reproduce the same validation result as @jrai ?" وأجاب `olivepossum`: "At the moment, not at all. There might be something wrong in my code".

### dropout في مشفّر بديل (DAE) من `yxbot` — الأرقام كما هي

> "I have been using DAE on the legacy dataset, but haven't got around to tailor it for the new dataset"

```python
self.encoder = nn.Sequential(
    nn.Dropout(p=0.1),
    nn.Linear(in_dimension, 256),
    nn.BatchNorm1d(256),
    nn.Hardswish(),
    nn.Dropout(p=0.1),
    ...
```

`embedding_dimension=10`، والطبقات `256 → 128 → 10`، وكل طبقات dropout بقيمة `p=0.1` — بما فيها **الطبقة الأولى على المدخل**. وقال عن الفرق مع نهج Yirun:

> "from a crude first look, seems the only significant difference from Yirun's approach is their use of Gaussian noise, which I shall try in due time."

### طريقة التحقّق (CV) وتسريب البيانات — نقاش صريح

`yxbot` عن نهجه:

> "basically repeated CV on separated eras - I didn't even do timesplit for any of my legacy models - they still do quite ok."

> "for this, it became unsupervised learning and I do it on train+val+test- all in one go."

`jrai` يحذّر:

> "That is a good idea. For this model, the autoencoder could still be pre-trained or fine-tuned on train+val+test as well to leverage the full unsupervised data. Although it would add leakage in validation stats."

**`hedgingcat` — صاحب الحلّ الفائز في Jane Street نفسه (16 أكتوبر 2021)**:

> "The unsupervised autoencoder without label information can be trained on train+valid+test, however, the supervised version in my solution should take extra care due to label leakage. So, I trained it in every fold to prevent leakage."

> "BTW, the highlight in my winning solution is the sample weight training which gives a boost to both public and private scores. But I am not sure if it can be useful in Numerai."

`luee` يحذّر من التسريب الزمني:

> "Depending on your cross-val scheme I would be wary of adding the valid + test data in the autoencoder loop, you don't want to test a model on data in 2018 that uses 2020 data for dimensionality reduction."

### الحفظ (memorization) — تعليل denoising autoencoder

`jrai` (23 ديسمبر 2021) ينقل عن Jeremy Jordan:

> "With this approach, our model isn't able to simply develop a mapping which memorizes the training data because our input and target output are no longer the same. Rather, the model learns a vector field for mapping the input data towards a lower-dimensional manifold ...; if this manifold accurately describes the natural data, we've effectively 'canceled out' the added noise."

> "The autoencoder is trying to do dimensionality reduction (compression), and in that goal it may be doing noise reduction. The jargon to describe this autoencoder is a bottleneck denoising autoencoder."

### اعتراضات على المعمارية (شكوك منشورة — مهمّة للتوازن)

`sunkay`: "The feature vector compressed by the encode is just another version of the original features, I don't understand why it could makes the model better."

`gbrecht` (25 ديسمبر 2021): "I fed my DAE features to the example model and the results were very bad. Maybe my DAE was bad, but it made me ditch the idea of training on reduced features only"

واعتراف `jrai` باحتمال overfitting نتائجه المنشورة:

> "The validation results I posted are after a fair amount of hp tuning, playing with different loss functions, number of layers, etc (so it's also possible the validation results are just wildly overfit). I think the code provided is a very good starting point though."

### ما **لا** يوجد في هذا المصدر

لا ذكر لـ label smoothing، ولا للانتباه بين الأصول (cross-sectional/asset attention)، ولا لـ GRU/TCN/Transformer، ولا لتجميع البذور (seed bagging) صراحةً — أقرب ما ورد هو نية `jrai`: "Ensemble CV folds and multiple models". ولا ذكر للتعلّم المستمر.

### ردود لاحقة (يناير–مارس 2022) — إضافات جوهرية

**على سيغما الضجيج نفسها** — `maxchu` (3 يناير 2022) يشكّك في تكافؤ النسختين:

> "@olivepossum correct me if I am wrong, the additive Gaussian noise seems not the same as the TensorFlow one. The sigma should be absolute rather than relative?"

(أي أن `GaussianNoise` في Keras سيغما **مطلقة**، بينما تنفيذ PyTorch أعلاه يضربها في `x` فتصبح **نسبية** — فرق يغيّر معنى الرقم `0.035` نفسه.)

وأضاف عن بدائل الخسارة:

> "I have also tried DenseNet and 'autofeature' (similar to autoencoder but uses subnetwork to predict masked features), i need to use differential spearman corr to get reasonable results. Using MSE is not very good in my previous network. I have also added a bunch of different losses like maximizing MMC for example prediction, minimizing feature exposure."

**النتائج الحقيقية (live) والاعتراف بالـ overfitting** — `jrai` (27 فبراير 2022)، وهو أهمّ اقتباس في الموضوع كلّه لأغراض PR7:

> "No unfortunately not yet. Here's one account with this MLP + AE architecture plus feature neutralization: Numerai. It started off looking promising. **It's definitely easy to overfit and that's likely what I did**, but I also like the model's metamodel correlation. It may still have some good performance in different regimes."

وسؤال `olivepossum` الذي سبقه:

> "did you manage to get good live results with this architecture? With my 'port' to pytorch I managed to get good val but I guess I have used way too much that data."

**خطأ برمجي في حلقة التدريب المنشورة** — `jefferythewind` (7 مارس 2022):

> "At the beginning of each iteration, you need to call optimizer.zero_grad() at the top. Pytorch by default accumulates gradients, so what you're doing here is adding the gradients to the previous values at each step and back propagating, which isn't usually what we want."

**على طريقة التحقّق** — `jefferythewind` (7 مارس 2022):

> "Just in general of considering and comparing alternate models, we should always look at cross validation scores. This way we compare out-of-sample performance on as much data as possible. After a long journey I'm fairly convinced that this is really the only way to do it. For neural nets I find it always adds a layer of complexity to the algo, since you need to automate things like how long you train for or early stopping. The training time should even be optimized for. ... If we're tuning the model by just looking at validation performance after fitting to training data, we can't tell if the improved performance works on other folds of the data or just validation."

وعن شكل الدفعة (batch) في PyTorch — `maxchu` (8 مارس 2022):

> "In PyTorch, the first dimension needs to be the same, so I will suggest you use the number of eras as batch size (1st dimension), then the second dimension is the number of stock and the last dimension as feature dimension."

---

## 10. البحث في forum.numer.ai عن ادّعاء «dropout بنسبة 35% على طبقة الإدخال»

### الحكم: **الادّعاء غير موجود** — لم أجد أيّ منشور على forum.numer.ai يذكر dropout بنسبة 35% على طبقة الإدخال.

**ما جُرِّب بحثاً** (كلّها فُتحت فعلاً، لا تخميناً):

| الاستعلام | الوسيلة | الناتج |
|---|---|---|
| `forum.numer.ai/search?q=dropout 35%` | بحث المنتدى الداخلي | لا نتيجة مطابقة |
| `forum.numer.ai/search?q=input dropout` | بحث المنتدى الداخلي | أقرب موضوع: "Feature reversing input noise" |
| `forum.numer.ai/search?q=0.35` | بحث المنتدى الداخلي | لا نتيجة مطابقة |
| `forum.numer.ai dropout 35% input layer neural network` | WebSearch مقيَّد بنطاق `forum.numer.ai` | النتيجة الوحيدة من المنتدى هي الموضوع 1416 نفسه، وبلا ذكر لـ35% |
| `numerai forum "0.35" dropout input features neural net architecture` | WebSearch عامّ | الموضوعان 1416 و3145، وبلا ذكر لـ0.35 |

### أقرب ثلاثة أرقام موجودة فعلاً (مرتَّبة بقربها من الادّعاء)

| # | المصدر | الرقم كما هو | ماهيّته الحقيقية |
|---|---|---|---|
| 1 | `mdo` — "Feature reversing input noise" | **`p = 0.25`** (25%) | **قلب إشارة** 25% من الميزات عشوائياً عند كل تكرار على المدخل — ليس dropout |
| 2 | AE-MLP (القسم 9) | **`dropout_rates[0] = 0.035`** | سيغما **ضجيج غاوسي** على المدخل (3.5%) — ولا dropout على المدخل مطلقاً |
| 3 | `yxbot` — DAE (القسم 9) | **`nn.Dropout(p=0.1)`** | dropout فعليّ على المدخل، بنسبة 10% |

**الفرضية الأرجح لأصل الادّعاء**: خلط بين `0.035` (سيغما ضجيج غاوسي = 3.5%) وقراءتها «35%»، أو خلط بين نسبة `mdo` = 25% لقلب الميزات وبين dropout. وفي الحالتين الادّعاء **غير مؤكَّد بمصدره المزعوم**، ولا يصحّ الاستناد إليه في PR7 بصيغته الحالية.

---

### 10-أ. المصدر الأقرب موضوعياً: `mdo` — "Feature reversing input noise"

- **الرابط**: https://forum.numer.ai/t/feature-reversing-input-noise/1416
- **الكاتب**: `mdo` (Michael Oliver — من فريق Numerai) — `76` إعجاباً، `6.5 ألف` مشاهدة، `22` منشوراً
- **التاريخ**: 6 يناير 2021

**الضجيج على المدخلات — الإطار النظري (نقل حرفي):**

> "A powerful way to regularize neural networks is by applying noise during training, whether it be to the inputs, hidden-unit activations, weights, or gradients. An early example of this is additive Gaussian noise applied to the inputs of denoising autoencoders. A more recent example is Dropout, in which multiplicative binomial noise is applied to inputs or hidden unit activations."

> "While training a network to be invariant or robust to these types of noise can be beneficial, such noise lacks structure that we may wish to be invariant/robust to as well."

**الرقم 25% — النصّ الحرفي الذي يجب أن يُنقل عن هذا المصدر:**

> "But I have found that training a network while reversing the sign of a randomly selected 25% of features at each iteration to be quite beneficial during training. The network naturally learns to reduce its maximum feature exposure and tends to spread exposure across many features rather than relying mostly on only a few."

> "The max feature exposures were also generally < 0.2 which is usually difficult to obtain without explicitly applying a penalty on exposure of applying feature neutralization."

**دافع التقنية (انهيار الأداء عند انقلاب إشارة ميزة):**

> "A major concern when modeling the data is taking on too much feature exposure because a feature could unexpectedly reverse the sign of its correlation with the target and over-dependence on that feature would then wreck prediction performance."

**الكود كما نُشر** (`p=0.25` قيمة افتراضية):

```python
## To be used on input to neural network. Make sure input is centered at 0!
class FeatureReversalNoise(nn.Module):
    def __init__(self, p=0.25):
        super(FeatureReversalNoise, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p

    def forward(self, x):
        if self.training:
            binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
            noise = 2*binomial.sample((1,x.shape[1])) - 1
            return x * noise.cuda()
        else:
            return x
```

ومعه نوع ضجيج ابتكره الكاتب على التنشيطات (وثيق الصلة بباب «الضجيج» في PR7):

```python
# This is used to add noise to neural net activations. It differs from the Gaussian Dropout suggested
# in the original Dropout paper in that the scale of the noise is proportional to the activation such
# that activation level equals the variance of noise (times alpha). Kinda like how real neurons have
# Poisson-ish noise
class CoupledGaussianDropout(nn.Module):
    def __init__(self, alpha=1.0):
        super(CoupledGaussianDropout, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        if self.training:
            stddev = torch.sqrt(torch.clamp(torch.abs(x), min=1e-6)).detach()
            epsilon = torch.randn_like(x) * self.alpha
            epsilon = epsilon * stddev
            return x + epsilon
        else:
            return x
```

**نصائح التدريب السبع (نقل حرفي — فيها early stopping وdropout وeras كدفعات):**

> "Make sure your features and targets are centered at 0, by subtracting 0.5 from each. (You should already be doing something like this if you're training neural networks. If not, SHAME!)
> Use early stopping
> Use other kinds of noise in your network as well, e.g. DropOut and/or the CoupledGaussianDropout I invented and put below because I'm feeling generous
> Use eras as mini-batches
> Try different optimizers. I really like Follow The Moving Leader (easily found using Google for your favorite NN framework)
> Experiment with different architectures, standard feedforward and nets with residual connections work IME
> I like training neural networks like making good BBQ: low (learning rate) and slow (many epochs)"

**تجميع النماذج وأرقام التعرّض:**

> "Also rather than choose one of the four models above, I ensembled them all, along with my XGBoost model (⅕ weight each) to produce a final prediction and then reduced maximum feature exposure down to 0.075."

**ملاحظة على تباين المعماريات (مفيدة لبند تجميع البذور):**

> "Interestingly different choices of network architecture can lead to networks that perform similarly on validation, but have very different feature exposure profiles."

**نتيجة حيّة متحفّظة** — سأل `halsmith99` (12 فبراير 2021): "is this still running in NMRO? rd 245 resolution didn't look too good." وردّ `mdo`: "Judging anything based on one round is a bad idea. The recent rounds have also been especially weird."

**تكييف الفكرة على XGBoost** — `senadorancap` (7 يناير 2021) نقل النهج إلى XGBoost عبر دالة `random_slicer` بمعامل `slice_percent` على 310 ميزة، وردّ `mdo`: "Very nice! I was hoping someone would try this with XGBoost as well".

**لماذا التمركز حول الصفر** — سأل `sirbradflies` عن لزومه مع وجود biases، فأجاب `mdo`:

> "It generally helps convergence since you don't have to move biases as much and in this case you really want them to be centered if you're multiplied by -1."

**eras كدفعات صغيرة — كود منشور من `belzebot` (21 فبراير 2021):**

```python
eras = df.era.unique()
np.random.shuffle(eras)
for era in eras:
   dfs = df[df.era == era]
   x = torch.from_numpy(dfs[features].values).float()
   y = torch.from_numpy(dfs.target.values).float()
```

واعتراض `silentj` على الفكرة (17 فبراير 2021):

> "What is the intuition behind eras as mini-batches? Wouldn't we want each step to be in the direction of lower loss values in more eras, as opposed to one era? I've been using quite large batches on shuffled data and it seemed to result in better risk metrics performance over non-shuffled data"

---

### 10-ب. عائد جانبي من البحث: التضمينات (embeddings) على بيانات Numerai

- **الرابط**: https://forum.numer.ai/t/nn-architecture-for-0-03-corr-on-validation-set/3145
- **الكاتب**: `nyuton` — 1 مايو 2021 — `135` إعجاباً، `9.0 ألف` مشاهدة، `53` منشوراً

هذا الموضوع يخصّ بند «التضمينات» في PR7 مباشرةً، وفيه الرأي والرأي المضادّ:

**مؤيّد** — `jacob_stahl` (3 مايو 2021):

> "While neural networks are fantastic at finding non-linear patterns, even a big network will always follow the path of least resistance during training. Meaning, it will it might only find a few strong relationships during training and ignore weaker ones. Training a single neural network to find as many relationships as possible is like kicking water uphill."

> "The feature embeddings in the article are able to extract patterns in from the dataset that would otherwise be drown out by stronger ones more directly correlated with the targets."

**معارض** — `paulito` (3 مايو 2021):

> "I don't think using embedding layers makes a lot of sense for numerai tournament. Embeddings are useful when handling categorical data, sparse data, or otherwise one hot encoded data. If you are dealing with intervall-scaled data, there is not really a need for that and most higher-order interactions will be learned by the model itself. The embedding layer helps most when you want to reduce the dimensionality of a really big matrix. For the tournament I don'nt see the need for that."

**متحفّظ** — `lysk` (2 مايو 2021):

> "From a NN perspective this is a bit surprising. One benefit of using NN is to learn cross feature relationships (non-linear ones). Here the author is forcing the network to learn some relationships before injecting that knowledge into a bigger network, which begs the question why not work with a big network in the first place? (such as a wide&deep architecture)"

> "So far I have not been successful in applying this idea to the tournament."

والرقم الوحيد الذي نقله `lysk` عن مقال المصدر: "(error rate from 37.6% down to 37.2%)".

**والأهمّ لبند الحفظ/overfitting** — صاحب الموضوع عن نتائجه:

> "This goes beyond my wildest dreams. I guess it's a big overfit, but I don't see why."

وعن early stopping: سأل `olivepossum`: "Do you use early stopping agains validation data?" فأجاب `nyuton`: "Sure, I always do! Otherwise it start overfitting soon." وأكّد أن النتائج بلا تحييد: "There is no neutralisation involved here."

---

## 11. Andrej Karpathy — "A Recipe for Training Neural Networks"

- **الرابط**: http://karpathy.github.io/2019/04/25/recipe/
- **الكاتب**: Andrej Karpathy
- **التاريخ**: 25 أبريل 2019

### الفشل الصامت — الادّعاء الأساسي الذي يستند إليه PR7

> "Neural net training fails silently"

> "Everything could be correct syntactically, but the whole thing isn't arranged properly, and it's really hard to tell. The 'possible error surface' is large, logical (as opposed to syntactic), and very tricky to unit test."

> "Therefore, your misconfigured neural net will throw exceptions only if you're lucky; Most of the time it will train but silently work a bit worse."

> "As a result, (and this is reeaally difficult to over-emphasize) a 'fast and furious' approach to training neural networks does not work and only leads to suffering."

> "The qualities that in my experience correlate most strongly to success in deep learning are patience and attention to detail."

### الاستراتيجية المركزية: احفظ أوّلاً ثمّ نظّم (overfit → regularize)

> "The approach I like to take to finding a good model has two stages: first get a model large enough that it can overfit (i.e. focus on training loss) and then regularize it appropriately (give up some training loss to improve the validation loss)."

> "The reason I like these two stages is that if we are not able to reach a low error rate with any model at all that may again indicate some issues, bugs, or misconfiguration."

### اختبارات التحقّق من صحّة المسار (وثيقة الصلة ببند «الحفظ»)

> "**overfit one batch.** Overfit a single batch of only a few examples (e.g. as little as two). To do so we increase the capacity of our model (e.g. add layers or filters) and verify that we can reach the lowest achievable loss (e.g. zero). I also like to visualize in the same plot both the label and the prediction and ensure that they end up aligning perfectly once we reach the minimum loss. If they do not, there is a bug somewhere and we cannot continue to the next stage."

> "**input-indepent baseline.** Train an input-independent baseline, (e.g. easiest is to just set all your inputs to zero). This should perform worse than when you actually plug in your data without zeroing it out. Does it? i.e. does your model learn to extract any information out of the input at all?"

> "**fix random seed.** Always use a fixed random seed to guarantee that when you run the code twice you will get the same outcome. This removes a factor of variation and will help keep you sane."

> "**verify loss @ init.** Verify that your loss starts at the correct loss value. E.g. if you initialize your final layer correctly you should measure -log(1/n_classes) on a softmax at initialization."

> "**init well.** Initialize the final layer weights correctly. E.g. if you are regressing some values that have a mean of 50 then initialize the final bias to 50."

> "**simplify.** Make sure to disable any unnecessary fanciness. As an example, definitely turn off any data augmentation at this stage."

### dropout — الاقتباس الحرفي الوحيد عنه (وفيه تحذير مهمّ)

> "**drop.** Add dropout. Use dropout2d (spatial dropout) for ConvNets. **Use this sparingly/carefully because dropout does not seem to play nice with batch normalization.**"

هذا التحذير يخصّ PR7 مباشرةً، لأن معمارية AE-MLP في القسم 9 تضع `BatchNormalization` و`Dropout` متجاورين في كل طبقة تقريباً.

### بقيّة أدوات التنظيم — بترتيب الكاتب نفسه (الأفضل أوّلاً)

> "**get more data.** First, the by far best and preferred way to regularize a model in any practical setting is to add more real training data. It is a very common mistake to spend a lot engineering cycles trying to squeeze juice out of a small dataset when you could instead be collecting more data."

> "As far as I'm aware adding more data is pretty much the only guaranteed way to monotonically improve the performance of a well-configured neural network almost indefinitely. **The other would be ensembles (if you can afford them), but that tops out after ~5 models.**"

> "**smaller input dimensionality.** Remove features that may contain spurious signal. Any added spurious input is just another opportunity to overfit if your dataset is small."

> "**smaller model size.** In many cases you can use domain knowledge constraints on the network to decrease its size."

> "**decrease the batch size.** Due to the normalization inside batch norm smaller batch sizes somewhat correspond to stronger regularization. This is because the batch empirical mean/std are more approximate versions of the full mean/std so the scale & offset 'wiggles' your batch around more."

> "**weight decay.** Increase the weight decay penalty."

> "**early stopping.** Stop training based on your measured validation loss to catch your model just as it's about to overfit."

> "**try a larger model.** I mention this last and only after early stopping but I've found a few times in the past that larger models will of course overfit much more eventually, but their 'early stopped' performance can often be much better than that of smaller models."

> "**data augment.** The next best thing to real data is half-fake data - try out more aggressive data augmentation."

### حجم النموذج واختيار المعمارية

> "**picking the model.** To reach a good training loss you'll want to choose an appropriate architecture for the data. When it comes to choosing this my #1 advice is: **Don't be a hero.** ... I always advise people to simply find the most related paper and copy paste their simplest architecture that achieves good performance."

> "**complexify only one at a time.** If you have multiple signals to plug into your classifier I would advise that you plug them in one by one and every time ensure that you get a performance boost you'd expect. Don't throw the kitchen sink at your model at the start."

### التجميع (ensembles) — الرقم كما هو

> "**ensembles.** Model ensembles are a pretty much guaranteed way to gain 2% of accuracy on anything. If you can't afford the computation at test time look into distilling your ensemble into a network using dark knowledge."

### معدّل التعلّم والمُحسِّنات — الأرقام كما هي

> "**adam is safe.** In the early stages of setting baselines I like to use Adam with a learning rate of 3e-4. In my experience Adam is much more forgiving to hyperparameters, including a bad learning rate. For ConvNets a well-tuned SGD will almost always slightly outperform Adam, but the optimal learning rate region is much more narrow and problem-specific."

> "**do not trust learning rate decay defaults.** ... E.g. ImageNet would decay by 10 on epoch 30. If you're not training ImageNet then you almost certainly do not want this. If you're not careful your code could secretely be driving your learning rate to zero too early, not allowing your model to converge. In my own work I always disable learning rate decays entirely (I use a constant LR) and tune this all the way at the very end."

### ضبط المعاملات

> "**random over grid search.** For simultaneously tuning multiple hyperparameters it may sound tempting to use grid search to ensure coverage of all settings, but keep in mind that it is best to use random search instead. Intuitively, this is because neural nets are often much more sensitive to some parameters than others."

### التدريب المسبق والتعلّم غير المراقَب (يخصّ بند المشفّر التلقائي في PR7)

> "**pretrain.** It rarely ever hurts to use a pretrained network if you can, even if you have enough data."

> "**stick with supervised learning.** Do not get over-excited about unsupervised pretraining. Unlike what that blog post from 2008 tells you, as far as I know, no version of it has reported strong results in modern computer vision (though NLP seems to be doing pretty well with BERT and friends these days ...)."

### ما **لا** يوجد في هذا المصدر

لا ذكر لـ label smoothing، ولا للانتباه بين الأصول (cross-sectional/asset attention)، ولا لـ GRU/TCN/Transformer كمعماريات (يُذكر RNN عرَضاً فقط: "If you are using RNNs and related sequence models it is more common to use Adam")، ولا لتجميع البذور باسمه (الأقرب: `ensembles` وسقفها "~5 models")، ولا لـ purged/grouped CV، ولا للتعلّم المستمر. المقال عامّ في الرؤية الحاسوبية ولا يخصّ البيانات المالية.

---

## 12. Hugging Face Blog — "Yes, Transformers are Effective for Time Series Forecasting (+ Autoformer)"

- **الرابط**: https://huggingface.co/blog/autoformer
- **الكُتّاب**: Eli Simhayev (`elisim`)، Kashif Rasul (`kashif`)، Niels Rogge (`nielsr`)
- **التاريخ**: 16 يونيو 2023 — `46` تصويتاً

### ⭐ الانتباه بين الأصول / النماذج متعدّدة المتغيّرات — أهمّ نتيجة في هذا المصدر لأغراض PR7

> "As one can observe, the vanilla Transformer which we introduced last year gets the best results here. **Secondly, multivariate models are typically worse than the univariate ones, the reason being the difficulty in estimating the cross-series correlations/relationships. The additional variance added by the estimates often harms the resulting forecasts or the model learns spurious correlations.** Recent papers like CrossFormer (ICLR 23) and CARD try to address this problem in Transformer models."

> "Multivariate models usually perform well when trained on large amounts of data. However, when compared to univariate models, especially on smaller open datasets, the univariate models tend to provide better metrics."

هذا دليل منشور **مضادّ** لفرضية أن نمذجة العلاقات بين الأصول (cross-sectional / multivariate) تُحسِّن التوقّع تلقائياً: الأرقام أدناه تُظهر أن كل نسخة متعدّدة المتغيّرات أسوأ من نظيرتها أحادية المتغيّر.

### الأرقام كما هي — MASE على مجموعة Traffic (جدول الخلاصة)

| Dataset | Transformer (uni.) | Transformer (mv.) | Informer (uni.) | Informer (mv.) | Autoformer (uni.) | DLinear |
|---|---|---|---|---|---|---|
| Traffic | **0.876** | 1.046 | 0.924 | 1.131 | 0.910 | 0.965 |

وجدول المقارنة الأول في المقال (Autoformer أحادي المتغيّر مقابل DLinear):

| Dataset | Autoformer (uni.) MASE | DLinear MASE |
|---|---|---|
| Traffic | 0.910 | 0.965 |
| Exchange-Rate | 1.087 | 1.690 |
| Electricity | 0.751 | 0.831 |

**ملاحظة على `Exchange-Rate`** — وهي أقرب مجموعة في الجدول إلى البيانات المالية: كلا الرقمين **أكبر من 1.0** (`1.087` و`1.690`)، أي أن كلا النموذجين أسوأ من خطّ الأساس الموسمي الساذج الذي يقيس عليه MASE. هذا الرقم وحده حجّة قويّة للحذر من التوقّع العميق على أسعار الصرف، ويستحقّ نقله في PR7 كما هو.

> "The results show that the Autoformer model outperforms the DLinear model on all three datasets."

> "**TL;DR**: A simple linear model, while advantageous in certain cases, has no capacity to incorporate covariates compared to more complex models like transformers in the univariate setting."

### حجم النموذج — الأرقام كما هي

إعدادات ثابتة لكل النماذج في المقارنة:

```python
prediction_length = 24
context_length = prediction_length*2
batch_size = 128
num_batches_per_epoch = 100
epochs = 50
scaling = "std"
```

> "The transformers models are all relatively small with:"

```python
encoder_layers=2
decoder_layers=2
d_model=16
```

وحجم DLinear المقابل: `hidden_dimension=2`، وعدد معاملاته كما طبعه pytorch-lightning: `4.7 K Trainable params` / `0.019 Total estimated model params size (MB)`، والتدريب توقّف عند `max_epochs=50`.

مجموعة البيانات: `862` سلسلة زمنية ساعية لإشغال طرق منطقة خليج سان فرانسيسكو 2015–2016، والتقييم على `7 * 862 = 6034` تنبّؤاً بـ`100` مسار احتمالي لكل سلسلة.

### آلية الانتباه — الارتباط الذاتي بديلاً عن self-attention

> "Moreover, Autoformer introduces an innovative auto-correlation mechanism that replaces the standard self-attention used in the vanilla transformer. This mechanism enables the model to utilize period-based dependencies in the attention, thus improving the overall performance."

> "In the vanilla Time Series Transformer, attention weights are computed in the time domain and point-wise aggregated. On the other hand ... Autoformer computes them in the frequency domain (using fast fourier transform) and aggregates them by time delay."

> "In practice, autocorrelation of the queries and keys for all lags is calculated at once by FFT. By doing so, the autocorrelation mechanism achieves 𝑂(𝐿 log 𝐿) time complexity (where 𝐿 is the input time length), similar to Informer's ProbSparse attention."

الكود كما نُشر (بديل `QK^T`):

```python
def autocorrelation(query_states, key_states):
    query_states_fft = torch.fft.rfft(query_states, dim=1)
    key_states_fft = torch.fft.rfft(key_states, dim=1)
    attn_weights = query_states_fft * torch.conj(key_states_fft)
    attn_weights = torch.fft.irfft(attn_weights, dim=1)
    return attn_weights
```

وعدد التأخيرات المختارة: `top_k = int(autocorrelation_factor * math.log(time_length))`.

### طبقة التفكيك (Decomposition Layer)

> "Autoformer incorporates a decomposition block as an inner operation of the model ... the encoder and decoder use a decomposition block to aggregate the trend-cyclical part and extract the seasonal part from the series progressively."

الصيغة كما وردت: `X_trend = AvgPool(Padding(X))` و`X_seasonal = X − X_trend`، والتنفيذ بـ`nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)`.

> "The concept of inner decomposition has demonstrated its usefulness since the publication of Autoformer. Subsequently, it has been adopted in several other time series papers, such as FEDformer (Zhou, Tian, et al., ICML 2022) and DLinear (Zeng, Ailing, et al., AAAI 2023)".

### تفسير فشل النموذج الخطّي (يخصّ بند «الميزات الزمنية»)

> "The traffic dataset has a distributional shift in the sensor patterns between weekdays and weekends. So what is going on here? Since the DLinear model has no capacity to incorporate covariates, in particular any date-time features, the context window we give it does not have enough information to figure out if the prediction is for the weekend or weekday. Thus, the model will predict the more common of the patterns, namely the weekdays leading to poorer performance on weekends."

### شحّ البيانات الكبيرة — قيد جوهري على التدريب المسبق

> "To summarize, Transformers are definitely far from being outdated when it comes to time-series forcasting! **Yet the availability of large-scale datasets is crucial for maximizing their potential.** Unlike in CV and NLP, the field of time series lacks publicly accessible large-scale datasets. Most existing pre-trained models for time series are trained on small sample sizes from archives like UCR and UEA, which contain only a few thousands or even hundreds of samples."

> "Although these benchmark datasets have been instrumental in the progress of the time series community, their limited sample sizes and lack of generality pose challenges for pre-training deep learning models."

### نقد منشور في تعليقات المقال (مهمّ للتوازن)

`omcandido` (6 فبراير 2025):

> "I think that this comparison is missing the mark. Here you are talking exclusively about date-related features. Indeed, I would expect neural architectures that encode past inputs (RNNs, CNNs, transformers, etc.) to be to learn specific features optimized for the prediction task at hand. As you say, the covariates available to DLinear are suboptimal and it would benefit from a bigger window."

> "In short, transformers are better at creating features, but not necessarily at incorporating covariates. Is there an honest comparison where both transformers and DLinear look at the same dataset with properly engineered features?"

### ما **لا** يوجد في هذا المصدر

لا ذكر لـdropout ولا للضجيج على المدخلات ولا label smoothing ولا purged/grouped CV ولا تجميع البذور ولا التعلّم المستمر ولا GRU/TCN (يُذكر CNN/RNN عرَضاً في تعليق فقط). قيمته في: **أرقام MASE**، وحُكم «متعدّد المتغيّرات أسوأ من أحادي المتغيّر»، وحجم النماذج الصغير، وآلية الارتباط الذاتي.

---

## 13. Hacker News — تعليق `dongobread` على TimeGPT-1 (الشكّ في نماذج التعلّم العميق للسلاسل الزمنية)

- **الرابط**: https://news.ycombinator.com/item?id=37877443
- **الكاتب**: `dongobread` — والنقاش على مقالة "TimeGPT-1"
- **التاريخ**: 14 أكتوبر 2023

### التعليق الأصلي — نقل حرفي كامل

> "As someone who's worked in time series forecasting for a while, I haven't yet found a use case for these 'time series' focused deep learning models."

> "On extremely high dimensional data (I worked at a credit card processor company doing fraud modeling), deep learning dominates, but there's simply no advantage in using a designated 'time series' model that treats time differently than any other feature. **We've tried most time series deep learning models that claim to be SoTA - N-BEATS, N-HiTS, every RNN variant that was popular pre-transformers, and they don't beat an MLP that just uses lagged values as features.** I've talked to several others in the forecasting space and they've found the same result."

> "On mid-dimensional data, LightGBM/Xgboost is by far the best and generally performs at or better than any deep learning model, while requiring much less finetuning and a tiny fraction of the computation time."

> "And on low-dimensional data, (V)ARIMA/ETS/Factor models are still king, since without adequate data, the model needs to be structured with human intuition."

> "As a result I'm extremely skeptical of any of these claims about a generally high performing 'time series' model. Training on time series gives a model very limited understanding of the fundamental structure of how the world works, unlike a language model, so the amount of generalization ability a model will gain is very limited."

**وزن هذا الاقتباس في PR7**: هو الحجّة المقابلة المباشرة لاختيار معمارية زمنية مخصّصة (GRU/TCN/Transformer) على بيانات أسعار: الادّعاء أن **MLP بميزات مُؤخَّرة (lagged features)** يضاهي أو يتجاوز تلك المعماريات، وأن LightGBM/XGBoost أفضل في المدى المتوسّط للأبعاد. وهو تعليق ممارس لا ورقة محكَّمة — يُنقل بوصفه خبرة منشورة، لا دليلاً قاطعاً.

### الردود عليه بأسمائها (كما وردت)

**`waltherg`** (14 أكتوبر 2023) — يسأل عن الحدود الكمّية وعن التوقّع متعدّد الخطوات:

> "Do you have rough measures for what constitutes high/mid/low- dimensional data? And how do you use XGBoost et al for multi-step forecasting, I.e. in scenarios where you want to predict multiple time steps in the future?"

**`isoprophlex`** (ردّاً على السؤال السابق) — نموذج لكل أفق تنبّؤ:

> "Because they're so cheap to train, you can just use n models if you want to predict n steps ahead."

> "The added benefit is that you optimize each regressor towards its own target timestep t+1 ... t+n. **A single loss on the aggregate of all timesteps is often problematic**"

**`aldanor`** — يشير إلى تطوّر في الانحدار متعدّد المخارج:

> "There's been recent advances in joint fitting of multi-output regression forests ('vector leaf')" (رابط إلى توثيق XGBoost عن `multioutput`)

**`jprafael`** — بديل عمليّ مهمّ، وفيه نقطة **تسريب/تقسيم** تخصّ طريقة التحقّق:

> "I've found that it works well to add the prediction horizon as a numerical feature (e.g. # of days), and them replicate each row for many such horizons, **while ensuring that all such rows go to the same training fold**."

**`yeahwhatever10`** — يسأل عن حدود الميزات المؤخَّرة مقابل طول التسلسل في الانتباه:

> "How does lagged features for an MLP compare to longer sequence lengths for attention in Transformers? Are you able to lag 128 time steps in a feed forward network and get good results?"

(لا جواب منشور على هذا السؤال في الخيط.)

**`asavinov`** (14 أكتوبر 2023) — موافقة من تجربة على بوت تداول، مع تحديد الحالات التي قد ينفع فيها المحوّل:

> "I agree that the conventional (numeric) forecasting can hardly benefit from the newest approaches like transformers and LLMs. I made such a conclusion while working on the intelligent trading bot [0] by experimenting with many ML algorithms. Yet, there exist some cases where transformers might provide significant advantages. They could be useful where the (numeric) forecasting is augmented with discrete event analysis and where sequences of events are important. Another use case is where certain patterns are important like those detected in technical analysis. **Yet, for these cases much more data is needed.**"

**`pfalke`** — الرأي المخالف الوحيد ذو الحجّة:

> "Foundational models can work where so far 'needs human intuition' was the state of things. I can picture a time series model with large enough Training corpus being able to deal quite well with typical quirks of seasonalities, shocks, outliers, etc."

> "I fully agree regarding how things have been so far, but I'm excited to see practitioners try out models such as the one presented here — it might just work."

**`jldugger`** — توضيح لمعنى التعليق الأصلي (ردّاً على `recursive4`: "So fraud is consistent with respect to time?"):

> "My read on this was that you can just dump the lagged values as inputs and let the network figure it out just as well as the other, time series specific models do, not that time doesn't matter."

**`StephenAshmore`** (15 أكتوبر 2023) — تأكيد مستقلّ ثانٍ:

> "I agree! I worked on forecasting sales data for years, and we had the same results."

**`dr_dshiv`** — تعليق تشبيهي: "Reminds me a bit how in psychology you have ANOVA, MANOVA, ANCOVA, MANCOVA etc etc but really in the end we are just running regressions—variables are just variables."

**`teeray`** و`kylebenzle` — تعليقان ساخران عن تسويق توقّعات السوق بـGPT، لا قيمة تقنية لهما.

### ما **لا** يوجد في هذا المصدر

لا أرقام قابلة للنقل (لا MASE ولا MSE ولا أي مقياس)، ولا ذكر لـdropout أو الضجيج أو label smoothing أو حجم النموذج أو purged CV أو التضمينات أو الانتباه بين الأصول أو تجميع البذور أو التعلّم المستمر. قيمته كاملةً في **حجّة المقارنة**: MLP بميزات مؤخَّرة، وLightGBM/XGBoost، مقابل معماريات السلاسل الزمنية المخصّصة.

---

## 14. Reddit — r/MachineLearning: **لم تُفتح**

- **الرابط**: https://www.reddit.com/r/MachineLearning/
- **الاستعلامان المطلوبان**: "transformer time series overfitting" و"financial data neural network memorize"

**الحالة: لم تُفتح.** لم يُنقل أيّ محتوى من هذا المصدر، ولم يُخمَّن شيء عنه.

سبب التعذّر، بالحرف كما ظهر:

| الوسيلة | الناتج |
|---|---|
| المتصفّح المدمج (`navigate`) | `https://reddit.com is not allowed due to safety restrictions and cannot be opened in the Browser pane.` |
| `WebSearch` مقيَّداً بنطاق `reddit.com` | `API Error: 400 — The following domains are not accessible to our user agent: ['reddit.com']` |

أي أن النطاق محجوب على مستوى البيئة (المتصفّح) **وعلى مستوى زاحف البحث** (استبعاد وكيل المستخدم)، لا لعطل عارض. لذلك **لا توجد نتائج لهذا البند**، ولا ترتيب لأعلى خمسة نقاشات بالتصويت.

**ما يلزم لإكماله** (خارج قدرة هذه البيئة): فتح الاستعلامين يدوياً في متصفّح المستخدم ونسخ النصّ، أو استخدام واجهة Reddit الرسمية بمفتاح تطبيق. إلى أن يحدث ذلك، **لا يصحّ إسناد أيّ ادّعاء في PR7 إلى r/MachineLearning**.

**تنبيه على أثر هذا النقص**: لا يسقط أي ادّعاء في PR7 بسببه، لأن المصدرين 13 (Hacker News) و15 (Cross Validated) يغطّيان الموضوعين نفسهما — الشكّ في معماريات السلاسل الزمنية، وحفظ الشبكات للبيانات — بمصادر مفتوحة ومُقتبسة حرفياً.

---

## 15. Cross Validated (stats.stackexchange) — "What should I do when my neural network doesn't generalize well?"

- **الرابط**: https://stats.stackexchange.com/questions/365778/what-should-i-do-when-my-neural-network-doesnt-generalize-well
- **السؤال**: طرحه `DeltaIV` في 7 سبتمبر 2018 — `74` تصويتاً، `70k` مشاهدة، آخر تعديل يوليو 2021، وعدد الإجابات `5`
- **الاستعلام المستخدم**: `neural network random labels memorization early stopping` — أعاد **نتيجتين** فقط، كلتاهما إجابتان على هذا السؤال؛ ولذلك فُتح السؤال نفسه وأُخذت أعلى ثلاث إجابات بالتصويت.

### الإجابة 1 — `Djib2011` (7 سبتمبر 2018) — **87 تصويتاً**، مقبولة

**تعريف الحفظ (نقل حرفي — الادّعاء الجوهري في PR7):**

> "High-capacity Machine Learning models have the ability to memorize the training set, which can lead to overfitting."

> "Overfitting is the state where an estimator has begun to learn the training set so well that it has started to model the noise in the training samples (besides all useful relationships)."

**early stopping:**

> "This technique attempts to stop an estimator's training phase prematurely, at the point where it has learned to extract all meaningful relationships from the data, before beginning to model its noise."

**dropout — وترتيبه عند هذا الكاتب:**

> "By far the most effective regularization technique is dropout, meaning that it should be the first you should use. However, you don't need to (and probably shouldn't) place dropout everywhere! The most prone layers to overfitting are the Fully Connected (FC) layers, because they contain the most parameters."

**تحذير مهمّ على التحقّق نفسه (يخصّ منهجية PR7 مباشرةً):**

> "A more practical approach than early stopping is storing the weights of the model that achieve the best performance on the validation set. Be cautious, however, as this is not an unbiased estimate of the performance of your model (just better than the training set). **You can also overfit on the validation set.**"

ويذكر ضمن ما يحدّ من overfitting: `Batch Normalization`، والدفعات الصغيرة في SGD، و"adding small random noise to weights in hidden layers" — وهذا الأخير صنف ثالث من الضجيج (على الأوزان) يختلف عن الضجيج على المدخلات في القسمين 9 و10.

### الإجابة 2 — `DeltaIV` (30 سبتمبر 2018) — **26 تصويتاً**

**⭐ الدليل المرجعي على حفظ التسميات العشوائية — وهو المصدر الأصليّ الذي يجب أن يُسنَد إليه ادّعاء «الحفظ» في PR7:**

> "There is plenty of empirical evidence that deep enough neural networks can memorize random labels on huge datasets (Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals, 'Understanding deep learning requires rethinking generalization'). Thus in principle by getting a big enough NN we can always reduce the training error to extremely small values, limited in practice by numerical accuracy, **no matter how meaningless the task**."

> "Things are quite different for the generalization error. We cannot be sure that for each learning problem, there exists a learnable NN model which can produce a generalization error as low as desired."

**ترتيب العمل الذي يوصي به (ضبط التوقّعات أولاً):**

> "Find a reputable reference which tells you that there exists an architecture which can reach the generalization error you're looking for, on your data set or on the most similar one for which you can find references."

**متى يُضاف التنظيم:**

> "For this reason, and because of the increase in training time, it's often better to introduce the various regularisation techniques one at a time, **after you successfully managed to overfit the training set**. Note that regularisation by itself doesn't necessarily imply your generalisation error will get smaller: the model must have a large enough capacity to achieve good generalisation properties."

**dropout في الشبكات المتكرّرة (يخصّ بند GRU في PR7) — نقل حرفي:**

> "use dropout: if you use LSTMs, use standard dropout only for input and output units of a LSTM layer. For the recurrent units (the gates) use recurrent dropout, as first shown by Yarin Gal in his Ph.D. thesis. However, if you use CNNs, dropout is used less frequently now."

**تضادّ dropout مع batch norm — بمرجعه المحكَّم (يؤكّد تحذير Karpathy في القسم 11):**

> "This could be just a fad, or it could be due to the fact that apparently dropout and batch normalisation don't play nice together (Xiang Li, Shuo Chen, Xiaolin Hu, Jian Yang, 'Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift')."

**حجم الدفعة — والخلاف المنشور حوله:**

> "reduce batch size: smaller batch sizes are usually associated with smaller generalisation error, so this is something to try. However, note that some dispute the usefulness of minibatches: in my experience, they help (as long as you don't have to use crazy small sizes such as m=16), but Elad Hoffer, Itay Hubara, Daniel Soudry 'Train longer, generalize better: closing the generalization gap in large batch training of neural networks' disagree. Note that if you use batch norm (see below), too small minibatches will be quite harmful."

**اختبارات سلامة الإجراء قبل أي كلام عن التعميم** — يذكر صراحةً: `unit tests`، و`dataset checks`، و`randomisation tests`، و`standardize your preprocessing and package versions`، و`keep a logbook of numerical experiments`.

**ملاحظة صدق يستحقّ نقلها:**

> "there's still a lot of alchemy in Deep Learning, and things that you would expect to work fine, sometimes don't, or vice versa something which worked ok many times, suddenly craps out on you for a new data set."

### الإجابة 3 — `shimao` (9 سبتمبر 2018) — **15 تصويتاً**

**رأي مضادّ صريح لترتيب الإجابة الأولى (dropout مقابل batch norm):**

> "Using batch normalization, which is a surprisingly effective regularizer to the point where **I rarely see dropout used anymore**, because it is simply not necessary."

**حجم النموذج — بالأرقام كما وردت:**

> "If possible, prefer fully convolutional architectures to architectures with fully connected layers. Compare VGG-16, which has 100 million parameters in a single fully connected layer, to Resnet-152, which has 10 times the number of layers and still fewer parameters."

**المُحسِّن وأثره على التعميم — بمرجعه:**

> "Prefer SGD to other optimizers such as Rmsprop and Adam. It has been shown to generalize better. ('Improving Generalization Performance by Switching from Adam to SGD' by Nitish Shirish Keskar and Richard Socher)"

ويذكر أيضاً `A small amount of weight decay`، وتقنيتَي `Shake-shake` و`Cutout`، مع تحفّظه الحرفي: "I believe these work better than dropout -- but I'm not sure."

### تعليق جوهري على السؤال نفسه — `Sycorax` (24 يونيو 2022)

> "The answers make good and relevant suggestions, but all of them seem to skip over the possibility that the data loader, network or training loop contains bugs. ... If the model doesn't generalize well due to a programming error, then the suggestions in the answers here won't catch or fix it."

وهو التقاء صريح مع محور مقال Karpathy في القسم 11 ("Neural net training fails silently").

### تعارض مسجَّل بين المصادر (يجب أن يظهر في PR7 كما هو، لا أن يُحسَم)

| المسألة | الموقف الأول | الموقف المقابل |
|---|---|---|
| dropout أم batch norm؟ | `Djib2011`: "By far the most effective regularization technique is dropout" | `shimao`: "I rarely see dropout used anymore, because it is simply not necessary" |
| جمعهما معاً | معمارية AE-MLP في القسم 9 تضعهما متجاورين في كل طبقة | Karpathy (قسم 11) و`DeltaIV` هنا: بينهما تضادّ موثَّق بورقة (Variance Shift) |
| حجم الدفعة الصغير | `Djib2011` و`DeltaIV`: يقلّل خطأ التعميم | Hoffer & Hubara & Soudry (مُقتبَسان في الإجابة 2): يخالفان |

### ما **لا** يوجد في هذا المصدر

لا ذكر لـ label smoothing، ولا للانتباه بين الأصول، ولا لـ TCN/Transformer، ولا لـ purged/grouped CV باسمه، ولا لتجميع البذور، ولا للتعلّم المستمر، ولا لأيّ شيء خاصّ بالبيانات المالية — الموضوع عامّ في تعميم الشبكات العصبية.

---

# حصيلة التحقّق — حالة المصادر الخمسة عشر

| # | المصدر | الحالة | ما يصلح للاستناد إليه في PR7 |
|---|---|---|---|
| 1 | Jane Street — حلّ Yirun (المركز 1) | فُتح | وصف المعمارية ومنطق الضجيج والمشفّر المراقَب |
| 2 | Jane Street — دفتر Supervised AE-MLP | فُتح | معاملات `GaussianNoise` وdropout وlabel smoothing من الكود |
| 3 | Ubiquant — المركز 1 | فُتح | — |
| 4 | Optiver Trading at the Close — المركز 1 | فُتح | — |
| 5 | Optiver Realized Volatility — المركز 1 | فُتح | — |
| 6 | Jane Street Real-Time — اللوحة والحلول | فُتح جزئياً | الحلّ المنشور الأعلى هو `[Private LB 8th]` لا المركز الأول |
| 7 | G-Research Crypto — أفضل الحلول | فُتح | — |
| 8 | Numerai — Feature Neutralization / Bias-Variance | فُتح | «الشبكات العصبية عالية التباين وتنزع إلى overfitting»، ونسبة التحييد `p=0.5` |
| 9 | Numerai — AutoEncoder + multitask MLP | فُتح | **كل الأرقام**: `dropout_rates`, `hidden_units`, `lr`؛ واعتراف صاحبه بسهولة overfitting حيّاً |
| 10 | بحث عن ادّعاء dropout 35% على المدخل | **بُحث ولم يُوجد** | الادّعاء **غير مؤكَّد**؛ البدائل الحقيقية: 25% (قلب إشارة)، 0.035 (سيغما ضجيج)، 0.1 (dropout على مدخل DAE) |
| 11 | Karpathy — A Recipe for Training NNs | فُتح | الفشل الصامت، overfit→regularize، تحذير dropout مع batch norm، سقف التجميع «~5 models» |
| 12 | HF Blog — Autoformer vs DLinear | فُتح | **أرقام MASE**؛ و«متعدّد المتغيّرات أسوأ من أحادي المتغيّر» — دليل مضادّ للانتباه بين الأصول |
| 13 | Hacker News — تعليق `dongobread` | فُتح | حجّة أن MLP بميزات مؤخَّرة يضاهي معماريات السلاسل الزمنية المخصّصة |
| 14 | Reddit r/MachineLearning | **لم تُفتح** | لا شيء — النطاق محجوب على مستوى البيئة والزاحف معاً |
| 15 | Cross Validated — عدم التعميم | فُتح | **المرجع الأصلي لحفظ التسميات العشوائية** (Zhang et al.)، وورقة تضادّ dropout/batch-norm، وتحذير «overfit على مجموعة التحقّق» |

## الادّعاءات التي خرجت من هذا التحقّق **غير مؤكَّدة**

1. **«dropout بنسبة 35% على طبقة الإدخال، بسند من منتدى Numerai»** — لا وجود له (القسم 10). إن كان PR7 يستند إليه فيجب حذفه أو إبداله بأحد الأرقام الثلاثة الموثَّقة مع نسبتها الصحيحة (قلب إشارة / ضجيج غاوسي / dropout).
2. **«الحلّ الأول في Jane Street Real-Time»** — لم يُنشر؛ أعلى حلٍّ منشور في تلك المسابقة هو `[Private LB 8th]` (القسم 6-ب). أي نسبة رقم إلى «المركز الأول» فيها غير قابلة للتحقّق.
3. **أي ادّعاء منسوب إلى r/MachineLearning** — لا سند له في هذا الملف (القسم 14).

## تعارضات يجب أن تُنقل كما هي، لا أن تُحسَم

- **dropout مع BatchNorm**: معمارية AE-MLP (القسم 9) تجمعهما في كل طبقة، بينما Karpathy (11) و`DeltaIV` (15) يوثّقان تضادّهما بورقة *Variance Shift*.
- **dropout أم batch norm أولاً**: `Djib2011` يضع dropout أوّلاً، و`shimao` يكاد يستبعده (القسم 15).
- **حجم الدفعة**: الأصغر أفضل للتعميم عند إجابتَي Cross Validated، وHoffer et al. يخالفون (القسم 15).
- **نمذجة العلاقات بين الأصول**: مرجوّة في PR7، لكن أرقام القسم 12 تُظهر أن كل نسخة متعدّدة المتغيّرات أسوأ من نظيرتها أحادية المتغيّر على Traffic.
