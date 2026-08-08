# Vessel-Motor-Classification-Representations-Architectures-and-Hyperparameters




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

- **Logistic regression**, sweeping the regularisation strength `C` over
  `[0.1, 0.5, 1, 2, 3, 4, 5, 10, 25, 50, 100]`. Accuracy climbs steadily with `C` and plateaus
  around **0.906** once `C ≥ 25` — the model was being over-regularised, and loosening it is
  what unlocks the jump from the low-0.80s. (An earlier `max_iter` sweep is left in the cell,
  commented out, as a record of what didn't move the needle — iteration count was never the
  bottleneck.)
- **KNN**, sweeping `k` from 1 to 14 — best at `k=1` (~0.81), degrading as `k` grows, which is
  what you'd expect on a small, imbalanced set.
- **Random forest**, gridding `n_estimators` × `max_depth` — tops out around 0.81 in the
  deeper-tree corner.

The cross-validation helper reports mean accuracy per configuration. The notebook then refits
the chosen models on the full training set and saves them: logistic regression is saved with
`C=25` (`LogisticRegression_bestmodel.pkl`), alongside `KNNbestmodel.pkl` and
`RandomForest_bestmodel.pkl`.

### Step 3 — evaluation on test data
The only notebook that touches `wolfset_test.csv`. It loads the saved classical models, runs
them over the held-out clips, and prints a confusion matrix plus a `classification_report`
for each. Three models get scored here:

- **Logistic regression, `C=25`, no resampling** (the Step 2 model) — **0.89 accuracy**,
  macro-F1 0.87 across the 37 test clips.
- **Logistic regression, `C=10`, SMOTE-trained** (the Step 4 `_OS` model) — **0.92 accuracy**,
  macro-F1 0.90. This is the strongest classical result on the test set.
- **Random forest, SMOTE-trained** (`_OS`) — **0.84 accuracy**, macro-F1 0.84.

Because the `_OS` models come from Step 4, run Step 4 before this notebook if you're
reproducing from scratch. One small inconsistency to be aware of when reading the saved
output: in the logistic-regression comparison cell the confusion-matrix plot is drawn from the
plain model's predictions while the printed report just below it comes from the `_OS` model, so
the picture and the numbers in that one cell describe slightly different models. The reports
are the values to trust.

### Step 4 — classifiers with oversampled data
A near-copy of Step 2 with one change that matters: **SMOTE** is applied inside each CV fold,
on the training portion only, to synthetically balance the classes before fitting. Doing it
inside the fold (rather than once up front) is the correct way to avoid leaking synthetic
neighbours into validation. Logistic regression is tuned over the same `C` grid and reaches
**~0.894** in cross-validation (flat once `C ≥ 10`). The refit models are saved with the `_OS`
suffix — logistic regression at `C=10` — and these are what Step 3 evaluates as the oversampled
variants. On the held-out test set that oversampled LR is the one that edges ahead (0.92),
so here the SMOTE step earns its place.

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
and a confusion matrix (whose class names it reads straight from `label_metadata.json`). On the
held-out set the YAMNet model reaches **0.88 accuracy** and **0.84 macro-F1**. It's strong on
the mid-range motors and, unsurprisingly given only two electric test clips, shakiest there.

## Reproducing this

1. Put the Wolfset `.wav` files under `datasets/25791978/` in your Drive and update
   `PROJECT_ROOT`.
2. Run **Step 1** to generate the CSV splits and the scaler.
3. For the classical track: run **Step 2**, then **Step 4** (Step 3 also evaluates the `_OS`
   models Step 4 saves), then **Step 3** to evaluate.
4. For the YAMNet track: run **Step 5** to sweep, **Step 6** to train the final model, then
   **Step 7** to evaluate. If Step 6 wrote to `artifacts/new/`, point Step 7 at the same
   folder or move the artifacts up one level first.

