# Does BERT Really Know Grammar?
## Stress-Testing Attention Probes with Control Tasks

This repository contains the code, experiments, results, figures, and seminar paper for a study of syntactic information in BERT attention.

The project investigates whether high probing accuracy genuinely reflects syntactic information encoded in BERT, or whether part of that performance can be explained by the learning capacity of the probe itself.

Rather than evaluating probes only through raw task accuracy, the study compares performance on real linguistic tasks with corresponding control tasks using **selectivity**.

---

## Research Question

> **To what extent does control-task selectivity support the claim that syntactic dependency information can be recovered from BERT attention, and how does this interpretation change across probe capacities and BERT layers?**

Two syntactic probing tasks are studied:

1. **Dependency Existence**  
   Predict whether two words participate in a dependency relation.

2. **Dependency Type**  
   Predict the grammatical dependency relation between a dependent and its syntactic head.

---

## Motivation

Probing classifiers are commonly used to investigate what linguistic information can be recovered from neural language-model representations.

However, high probe accuracy alone can be difficult to interpret. A sufficiently expressive classifier may learn regularities of the probing task or memorize associations rather than simply reveal information already available in the underlying representation.

Following the control-task framework of Hewitt and Liang, this project evaluates probes using **selectivity**:

\[
\text{Selectivity}
=
\text{Accuracy}_{\text{real}}
-
\text{Accuracy}_{\text{control}}
\]

A probe is more selective when it performs well on the real linguistic task while remaining comparatively weak on the corresponding arbitrary control task.

---

# Experimental Setup

## Language Model

The experiments use:

- `bert-base-uncased`
- 12 Transformer layers
- 12 attention heads per layer
- 144 attention dimensions per word pair

BERT remains frozen throughout the experiments.

No task-specific fine-tuning is performed.

---

## Dataset

The experiments use:

**Universal Dependencies English Web Treebank (UD English EWT), version r2.18**

Each annotated dependency is converted into an ordered pair:

```text
dependent -> syntactic head
```

Root relations are excluded.

For dependency existence, annotated dependencies form the positive examples. Non-dependency word pairs from the same sentence are sampled as negative examples.

Self-pairs and genuine dependencies in either direction are excluded from negative sampling.

The probing classifiers are trained using the training split and evaluated using the test split.

---

## Attention Representation

BERT operates on WordPiece tokens, whereas the dependency treebank contains word-level tokens.

The original words are therefore aligned with their corresponding BERT subword tokens.

If a word is divided into multiple WordPieces, attention values across the relevant subword combinations are averaged separately for every attention head.

BERT-base contains:

```text
12 layers x 12 attention heads = 144 dimensions
```

Each word pair is therefore represented by a 144-dimensional attention vector.

For genuine dependency pairs, attention is measured in the direction:

```text
dependent -> syntactic head
```

The attention features are cached before probe training so that the real and control experiments use identical BERT representations.

---

# Probing Tasks

## Dependency Existence

Dependency existence is formulated as a binary classification problem.

The probe predicts whether a pair of words forms an annotated dependency relation.

- Positive class: genuine dependency pair
- Negative class: sampled non-dependency pair

---

## Dependency Type

Dependency type is evaluated only on genuine dependency pairs.

The classifier predicts the grammatical relation connecting the dependent to its syntactic head.

The 20 most frequent dependency labels in the training data are retained individually.

All remaining labels are mapped to:

```text
other
```

---

# Control Tasks

Control tasks use the same BERT attention features as the linguistic tasks but replace meaningful linguistic targets with arbitrary labels.

This allows probe performance to be compared with what the same classifier can learn without the intended linguistic relationship.

## Dependency-Existence Control Task

Each lowercased word type in the training vocabulary is randomly assigned to one of two arbitrary classes.

A word pair receives a positive control label when both words belong to the same arbitrary class.

Otherwise, it receives a negative control label.

The control vocabulary is generated reproducibly using seed `42`.

---

## Dependency-Type Control Task

Each word type is assigned an arbitrary dependency label.

Instead of sampling labels uniformly, the labels are sampled according to the empirical dependency-label distribution in the training data.

This approximately preserves the class-frequency distribution of the real task while removing the actual linguistic relationship.

The dependent word determines the control label of a pair.

---

# Experiments

The project contains three main experiments.

---

## Experiment 1 — Baseline Selectivity

The first experiment trains a linear probe using the complete 144-dimensional attention representation.

### Results

| Task | Real Accuracy | Control Accuracy | Selectivity |
|---|---:|---:|---:|
| Dependency Existence | 0.8404 | 0.5065 | 0.3339 |
| Dependency Type | 0.7130 | 0.3116 | 0.4014 |

Both tasks show substantial positive selectivity.

The probe performs considerably better on meaningful linguistic labels than on the corresponding arbitrary control labels.

---

## Experiment 2 — Probe Capacity

The second experiment investigates whether increasing classifier capacity also increases selectivity.

Three probe configurations are compared:

- Logistic Regression
- MLP with 8 hidden units
- MLP with 32 hidden units

### Dependency Existence

| Probe | Real Accuracy | Control Accuracy | Selectivity |
|---|---:|---:|---:|
| Linear | 0.8404 | 0.5065 | 0.3339 |
| MLP-8 | 0.8790 | 0.4934 | **0.3856** |
| MLP-32 | **0.8995** | 0.5225 | 0.3771 |

The MLP-32 achieves the highest raw accuracy.

However, the smaller MLP-8 produces the highest selectivity.

---

### Dependency Type

| Probe | Real Accuracy | Control Accuracy | Selectivity |
|---|---:|---:|---:|
| Linear | 0.7130 | 0.3116 | **0.4014** |
| MLP-8 | 0.6707 | 0.3191 | 0.3516 |
| MLP-32 | **0.7743** | 0.4033 | 0.3710 |

For dependency type, the MLP-32 again achieves the highest real-task accuracy.

However, the linear probe obtains the highest selectivity.

---

## Main Observation from Experiment 2

**Higher probing accuracy does not necessarily imply stronger evidence of linguistic encoding.**

Increasing classifier capacity can improve performance on both the linguistic task and the arbitrary control task.

Selectivity therefore provides information that raw accuracy alone does not capture.

---

# Experiment 3 — Layer-Wise Selectivity

The complete 144-dimensional attention representation is divided into 12 groups corresponding to the 12 BERT layers.

Each layer contributes:

```text
12 attention-head dimensions
```

A separate linear probe is trained using each layer individually.

A randomly selected 12-dimensional subset from the full attention representation is also evaluated as a dimensionality-matched baseline.

Layer numbering follows the implementation and ranges from:

```text
0 to 11
```

---

## Dependency Existence

The highest single-layer raw accuracy occurs at layer 3:

```text
Layer 3

Real Accuracy = 0.7699
Selectivity   = 0.2176
```

The highest layer-wise selectivity occurs at layer 7:

```text
Layer 7

Real Accuracy = 0.7687
Selectivity   = 0.3100
```

This is an important result.

The layer with the highest raw accuracy is **not** the layer with the highest selectivity.

### Random 12-Dimensional Baseline

```text
Real Accuracy    = 0.7063
Control Accuracy = 0.4789
Selectivity      = 0.2274
```

Layer 3 achieves higher raw accuracy than the random baseline but slightly lower selectivity.

This further demonstrates why accuracy and selectivity should not be treated as equivalent.

---

## Dependency Type

Layer 7 is the strongest layer for both raw accuracy and selectivity:

```text
Layer 7

Real Accuracy    = 0.4404
Control Accuracy = 0.1762
Selectivity      = 0.2642
```

### Random 12-Dimensional Baseline

```text
Real Accuracy    = 0.2745
Control Accuracy = 0.1596
Selectivity      = 0.1149
```

The middle layers, particularly around layers 5–8, contain stronger syntactic signals than many of the earlier or later layers.

---

# Full Representation vs. Individual Layers

No individual layer matches the complete 144-dimensional attention representation.

## Dependency Existence

```text
Best single-layer accuracy = 0.7699
Full representation        = 0.8404
```

## Dependency Type

```text
Best single-layer accuracy = 0.4404
Full representation        = 0.7130
```

The results suggest that syntactic information is distributed across multiple BERT attention layers rather than being completely localized in one layer.

Layer 7 provides particularly strong selective information, but it should not be interpreted as a universal syntactic layer.

---

# Main Findings

The experiments support four main conclusions.

### 1. Syntactic information is recoverable from BERT attention

Both dependency existence and dependency type produce substantial positive selectivity.

---

### 2. Probe accuracy and selectivity are not equivalent

The classifier with the highest linguistic-task accuracy is not always the classifier with the highest selectivity.

---

### 3. Layer-wise accuracy and selectivity can produce different interpretations

For dependency existence:

```text
Highest accuracy    -> Layer 3
Highest selectivity -> Layer 7
```

For dependency type:

```text
Highest accuracy    -> Layer 7
Highest selectivity -> Layer 7
```

---

### 4. Syntactic information appears to be distributed across BERT

No individual layer achieves the performance of the complete 144-dimensional attention representation.

This suggests that multiple layers contribute complementary syntactic information.

---

# Repository Structure

```text
bert-attention-probe-selectivity/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── build_pairs.py
│   ├── build_pairs_helpers.py
│   ├── build_control_tasks.py
│   ├── capacity_sweep.py
│   ├── capacity_sweep_dryrun.py
│   ├── check_features.py
│   ├── config.py
│   ├── control_tasks.py
│   ├── extract_features.py
│   ├── layerwise.py
│   ├── make_plots.py
│   ├── parse_treebank.py
│   ├── probes.py
│   ├── run_experiment.py
│   ├── run_selectivity.py
│   └── sanity_check.py
│
├── results/
│   ├── tables/
│   │   ├── capacity_sweep_existence.csv
│   │   ├── capacity_sweep_type.csv
│   │   ├── layerwise_existence.csv
│   │   └── layerwise_type.csv
│   │
│   └── figures/
│       ├── capacity_selectivity_existence.png
│       ├── capacity_selectivity_type.png
│       ├── capacity_sweep_existence.png
│       ├── capacity_sweep_type.png
│       ├── layerwise_existence.png
│       ├── layerwise_selectivity_existence.png
│       ├── layerwise_selectivity_type.png
│       └── layerwise_type.png
│
└── paper/
    ├── main.tex
    ├── references.bib
    └── final paper files
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/vinsy2774/bert-attention-probe-selectivity.git
cd bert-attention-probe-selectivity
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# Running the Pipeline

The overall workflow is:

```text
UD English EWT
      |
      v
Dependency Pair Construction
      |
      v
Control Task Construction
      |
      v
BERT Attention Extraction
      |
      v
Probe Training
      |
      +----------------------+
      |                      |
      v                      v
Capacity Sweep          Layer-Wise Analysis
      |                      |
      +-----------+----------+
                  |
                  v
             CSV Results
                  |
                  v
               Figures
```

---

## Build Dependency Pairs

```bash
python src/build_pairs.py
```

---

## Build Control Tasks

```bash
python src/build_control_tasks.py
```

---

## Extract BERT Attention Features

```bash
python src/extract_features.py
```

---

## Run Baseline Selectivity Experiment

```bash
python src/run_selectivity.py
```

---

## Run Probe-Capacity Experiment

```bash
python src/capacity_sweep.py
```

---

## Run Layer-Wise Experiment

```bash
python src/layerwise.py
```

---

## Generate Figures

```bash
python src/make_plots.py
```

Generated tables are stored in:

```text
results/tables/
```

Generated figures are stored in:

```text
results/figures/
```

---

# Reproducibility

The experiments use the following configured seeds:

```text
13
42
97
1000
2024
```

The control-task vocabulary is generated reproducibly.

The random 12-dimensional baseline uses seed:

```text
42
```

BERT attention features are cached before probe training so that linguistic and control experiments operate on identical representations.

---

# Result Files

## Capacity Sweep

```text
results/tables/capacity_sweep_existence.csv
results/tables/capacity_sweep_type.csv
```

## Layer-Wise Analysis

```text
results/tables/layerwise_existence.csv
results/tables/layerwise_type.csv
```

---

# Figures

## Probe Capacity

```text
capacity_sweep_existence.png
capacity_sweep_type.png
capacity_selectivity_existence.png
capacity_selectivity_type.png
```

## Layer-Wise Analysis

```text
layerwise_existence.png
layerwise_type.png
layerwise_selectivity_existence.png
layerwise_selectivity_type.png
```

The selectivity figures are emphasized in the seminar paper because selectivity is the main quantity used to distinguish linguistic-task performance from control-task learning.

---

# Limitations

The study is intentionally limited in scope.

It evaluates:

- one pretrained model: `bert-base-uncased`
- one language: English
- one dependency corpus: UD English EWT
- attention representations rather than hidden states
- one linear probe
- two small MLP probes
- one fixed random 12-dimensional baseline

The findings therefore should not automatically be generalized to other Transformer models, languages, domains, or datasets.

The experiments also demonstrate that syntactic information is **decodable** from BERT attention.

They do not establish that BERT necessarily uses this information causally during its own predictions.

---

# Future Work

An important future experiment is to compare pretrained BERT with an identically structured but randomly initialized model.

This would help distinguish syntactic patterns acquired through pretraining from patterns that may arise from architectural inductive biases.

Other possible extensions include:

- other Transformer architectures
- multilingual models
- dependency treebanks from additional languages
- hidden-state probing
- broader probe-capacity ranges
- alternative control-task formulations
- repeated random 12-dimensional baselines
- comparisons between attention-based and hidden-state representations

---

# Seminar Paper

The accompanying seminar paper is titled:

> **Does BERT Really Know Grammar? Stress-Testing Attention Probes with Control Tasks**

Prepared for:

**Advanced Topics in Computational Text and Media Science**

Department of Computational Linguistics  
M.Sc. Natural Language Processing  
Universität Trier  
Summer Semester 2026

The LaTeX source, bibliography, and final paper files are stored in:

```text
paper/
```

---

# Key References

The study builds on work including:

- Devlin et al. — *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*
- Clark et al. — *What Does BERT Look at? An Analysis of BERT's Attention*
- Tenney et al. — *BERT Rediscovers the Classical NLP Pipeline*
- Hewitt and Liang — *Designing and Interpreting Probes with Control Tasks*
- Rogers et al. — *A Primer in BERTology*
- Belinkov — *Probing Classifiers: Promises, Shortcomings, and Advances*
- Niu et al. — *Does BERT Rediscover a Classical NLP Pipeline?*
- Zhu and Rudzicz — *An Information Theoretic View on Selecting Linguistic Probes*

Complete bibliographic information is available in:

```text
paper/references.bib
```

---

# Author

**Swathi Raya**

M.Sc. Natural Language Processing  
Universität Trier

---

# Repository

https://github.com/vinsy2774/bert-attention-probe-selectivity