# Vessel Motor Classification

Classifying small boat motors from the sound they make. The recordings come from the
Wolfset collection of outboard-engine audio, and the goal across these notebooks is to
answer one practical question: what's the best way to tell these motors apart from a short
audio clip?

To get there the project pulls on three levers and compares them head to head:

- **Representations** — hand-built acoustic features (MFCCs + spectral rolloff) versus deep
  audio embeddings from Google's YAMNet.
- **Architectures** — logistic regression, k-nearest-neighbours, random forest, and a small
  Keras MLP.
- **Hyperparameters** — swept with cross-validation for every model so the comparison is
  fair rather than one lucky run.

That's also where the repo name comes from. If you're reviewing this, the notebooks are
meant to be read in numerical order; each one hands its output to the next.

## The data

The clips are `.wav` files of individual outboard motors. There are 168 recordings in the
raw folder, of which 122 have exactly one motor running and are the ones we actually use.

Every label lives in the filename. The naming template is `Axxxxx Ttt Nnn`, for example
`A00300T00N00.wav`:

- Characters 2–6 are the **motor code** — five digits, one per motor. A non-zero digit means
  that motor is present. The *position* of the non-zero digit is the class.
- Characters after `T` encode the **transient** condition, and after `N` the **noise**
  condition. We keep these columns around but don't train on them.

So the five classes are just "which motor slot is lit up," and they map to real engines like
this:

| Label | Motor      | Notes                |
|-------|------------|----------------------|
| 1     | 4.5 hp     | combustion           |
| 2     | 18 hp      | combustion           |
| 3     | 8 hp       | combustion           |
| 4     | 3.6 hp     | combustion           |
| 5     | Electric   | electric             |

Step 1 keeps only files where a single motor is present (the first non-zero digit wins), which
is how 168 recordings become 122 labelled examples.

One thing worth flagging up front: the classes are imbalanced. The 4.5 hp motor dominates
(40 of 85 training clips), and the electric motor is the rarest (8). That imbalance is the
reason Step 4 exists, and it's the lens to keep in mind when reading any per-class score.

## How everything is wired up

All seven notebooks were written for Google Colab and read from a fixed Google Drive layout:

```
MyDrive/1:1_Arjan_Walia/Vessels/
├── datasets/
│   ├── 25791978/                    # raw .wav recordings (the Wolfset dump)
│   └── processed_files/
│       ├── wolfset_train.csv        # 70% split, MFCC features + labels
│       └── wolfset_test.csv         # 30% split, held out until evaluation
├── models/                          # the classical .pkl models + the feature scaler
└── artifacts/                       # YAMNet embeddings, trained MLP, metrics, label metadata
```

To run any notebook, the only line you should need to touch is `PROJECT_ROOT` (or the
`dataset_path` / `save_folder` variables in Step 1). Point it at wherever the `Vessels`
folder lives in your Drive and the rest of the paths build off it. The YAMNet notebooks
(Steps 5–7) also `pip install` their own dependencies in the first cell, so they'll run on a
clean Colab runtime; the classical notebooks lean on libraries Colab already ships.

Data flows strictly forward. Step 1 writes the CSVs and the scaler. Steps 2 and 4 read the
train CSV and write models. Step 3 reads the test CSV and the saved models. Steps 5–7 build
their own YAMNet embeddings from the same CSVs. Nothing loops back, so as long as Step 1 has
run once against your copy of the audio, you can jump to whichever track you want to look at.

## Notebook walkthrough

### Step 1 — data processing
Turns raw audio into a tabular dataset. For each `.wav` it loads the first 10 seconds, applies
a pre-emphasis filter (standing in for the paper's 50 Hz notch to knock out electrical hum),
and pulls out **13 MFCCs** plus **spectral rolloff** — 14 features per clip, all mean-pooled
over time. It parses the label out of the filename, does a stratified 70/30 train/test split
(`random_state=42`), fits a `StandardScaler` **on the training set only**, and saves the
scaler alongside `wolfset_train.csv` and `wolfset_test.csv`.

What to check here: the split is stratified so the rare classes survive into both sides, and
the scaler is fit before the test data is ever touched. Both matter for trusting the numbers
downstream. The last few cells are quick sanity prints of the class counts.

### Step 2 — building classifiers (baseline)
Trains three classical models on the MFCC features and tunes each with 5-fold cross-validation:

- **Logistic regression**, sweeping `max_iter` — flat at ~0.81, so iterations aren't the
  bottleneck.
- **KNN**, sweeping `k` from 1 to 14 — best at `k=1` (~0.81), degrading as `k` grows, which is
  what you'd expect on a small, imbalanced set.
- **Random forest**, gridding `n_estimators` × `max_depth` — tops out around 0.81 in the
  deeper-tree corner.

The cross-validation helper reports mean accuracy per configuration. The notebook then refits
the chosen models on the full training set and saves them (`KNNbestmodel.pkl`,
`LogisticRegression_bestmodel.pkl`, `RandomForest_bestmodel.pkl`).

### Step 3 — evaluation on test data
The only notebook that touches `wolfset_test.csv`. It loads the saved classical models, runs
them over the held-out clips, and prints a confusion matrix plus a `classification_report`
for each. Note that it loads the **oversampled** (`_OS`) variants of the models from Step 4,
so run Step 4 before this one if you're reproducing from scratch.

Headline on the held-out set: the logistic-regression model lands at **0.97 accuracy** across
the 37 test clips, with the only real confusion being one 8 hp clip. The random forest trails
well behind it.

Heads-up for reviewers: the random-forest report shows a different support count (25) than the
logistic-regression one (37) in the saved outputs, which means those two cells were last run
against different states of the data. If you re-execute the whole notebook top to bottom the
supports will line up; just don't read too much into that particular saved comparison.

### Step 4 — classifiers with oversampled data
A near-copy of Step 2 with one change that matters: **SMOTE** is applied inside each CV fold,
on the training portion only, to synthetically balance the classes before fitting. Doing it
inside the fold (rather than once up front) is the correct way to avoid leaking synthetic
neighbours into validation. Logistic regression nudges up to ~0.835 in cross-validation. The
refit models are saved with the `_OS` suffix, and these are what Step 3 actually evaluates.

### Step 5 — classifier using YAMNet
Switches representations entirely. Instead of MFCCs, each clip (resampled to 16 kHz, 5 seconds,
peak-normalised) is pushed through **YAMNet** to get a 1024-dim embedding, mean-pooled across
frames. Embeddings are cached to disk as `.npy` files so you only pay the extraction cost once.

On top of the embeddings sits a small MLP: `Dense(256) → Dropout → Dense(64) → Dropout →
softmax`, trained with class weights to fight the imbalance. It sweeps learning rate
(`1e-4, 3e-4, 1e-3`) against epochs (`10, 20, 30`) using stratified 5-fold CV, writes the
per-fold and summary tables to `artifacts/`, and draws a heatmap of mean macro-F1 over the
grid. Best configuration: **learning rate 1e-3, 20 epochs** (mean macro-F1 ≈ 0.72).

### Step 6 — training the best YAMNet model
Self-contained final-training run. It reads the CV summary from Step 5, picks the winning
learning-rate/epochs pair automatically (with a hard-coded fallback of 1e-3 / 20), and trains
the MLP once on **all** the training data. Then it saves everything inference needs:
`yamnet_classifier.keras`, the fitted `yamnet_scaler.joblib`, `label_metadata.json`, the
training config, and the training history. Note it writes to `artifacts/new/` — if you want
Step 7 to read these, either repoint Step 7's `ARTIFACT_DIR` or move the files up to
`artifacts/`.

### Step 7 — YAMNet inference
Loads the saved model, scaler, and label metadata, builds embeddings for the **test** clips
(cached separately from the training embeddings), and produces the final numbers: a
`classification_report`, a metrics JSON, a per-file predictions CSV with confidence scores,
and a confusion matrix. On the held-out set the YAMNet model reaches **0.88 accuracy** and
**0.84 macro-F1**. It's strong on the mid-range motors and, unsurprisingly given only two
electric test clips, shakiest there.

## Results at a glance

Cross-validation (mean accuracy / macro-F1 on the training folds):

| Track                       | Best model                       | CV score        |
|-----------------------------|----------------------------------|-----------------|
| MFCC, no resampling         | KNN (`k=1`) / LogReg             | ~0.81 accuracy  |
| MFCC, SMOTE per fold        | LogReg                           | ~0.835 accuracy |
| YAMNet embeddings + MLP     | lr=1e-3, 20 epochs               | ~0.72 macro-F1  |

Held-out test set:

| Model                                   | Accuracy | Macro-F1 |
|-----------------------------------------|----------|----------|
| Logistic regression on MFCCs (Step 3)   | 0.97     | 0.97     |
| YAMNet + MLP (Step 7)                    | 0.88     | 0.84     |

The interesting takeaway is that the heavyweight representation doesn't win. On a dataset this
small and this clean, 14 hand-picked features plus a linear model beat 1024-dim deep
embeddings plus a neural net. YAMNet generalises well without much tuning, but the simple
pipeline has less to overfit and comes out ahead. With only 25–37 test clips, though, a single
misclassification swings the accuracy by a few points, so treat the gap as suggestive rather
than settled — the honest way to firm it up is more recordings, especially of the electric
and 8 hp motors.

## Reproducing this

1. Put the Wolfset `.wav` files under `datasets/25791978/` in your Drive and update
   `PROJECT_ROOT`.
2. Run **Step 1** to generate the CSV splits and the scaler.
3. For the classical track: run **Step 2**, then **Step 4** (Step 3 loads the `_OS` models),
   then **Step 3** to evaluate.
4. For the YAMNet track: run **Step 5** to sweep, **Step 6** to train the final model, then
   **Step 7** to evaluate. If Step 6 wrote to `artifacts/new/`, point Step 7 at the same
   folder or move the artifacts up one level first.

Runtimes are short — the classical notebooks finish in minutes on CPU, and the YAMNet
notebooks are dominated by the one-time embedding extraction, which is cached after the first
pass.

## A few notes for reviewers

- Every split, scaler, and SMOTE step is seeded and kept train-only, so there's no obvious
  leakage between train and test.
- The label lives in the filename, so if you swap in new recordings they have to follow the
  `Axxxxx Ttt Nnn` convention or Step 1 won't parse them.
- The two tracks share the same train/test split (both read the same CSVs), which is what
  makes the accuracy comparison in the results table a fair one.
- `label_metadata.json` is produced by the YAMNet track but reused for the class names in the
  Step 3 confusion matrices, so run the YAMNet side at least once if you want those labels to
  render.
- The confusion-matrix helper is labelled "normalized" in its title but actually plots raw
  counts — the values are correct, the title is just misleading.
