# Gaussian geometry visualizations

The research visualizers are optional and require Matplotlib:

```bash
python -m pip install -e '.[viz]'
```

They project high-dimensional means and covariance into a shared two-dimensional diagnostic basis and separately retain every low-rank frame axis. A projected ellipse is a visualization aid, not the complete high-dimensional distribution.

## Query and passage geometry

Render a trained finetuning checkpoint:

```bash
python finetune/visualize_gaussians.py \
  --checkpoint /models/geologic-memory.pt \
  --model-name all-MiniLM-L6-v2 \
  --output-dim 384 \
  --rank 16 \
  --device cpu \
  --query "Where did Alice live before Portland?" \
  --passage "Alice lived in Boston until June 2026." \
  --passage "Alice moved to Portland after joining Northstar Robotics." \
  --output visualizations/finetune/alice_query.png
```

Match `model-name`, `hidden-dim`, `output-dim`, `rank`, decomposition type, and Stiefel iterations to the checkpoint. The command writes:

- a PNG with projected 2σ Gaussians, sorted low-rank eigenvalues, projected frame directions, and orthogonality error;
- an adjacent JSON file with norms, variance summaries, eigenvalue ranges, and `‖VᵀV-I‖F`.

## Periodic finetuning snapshots

Enable snapshots every 500 optimization steps:

```bash
python finetune/train.py \
  --exp_name gaussian-geometry \
  --model_name all-MiniLM-L6-v2 \
  --output_dim 384 \
  --rank 16 \
  --visualize_every 500 \
  --visualization_examples 4 \
  --visualization_dir visualizations/finetune
```

Plotting defaults off. A plotting failure is reported but does not stop training.

## LoCoMo planning geometry

Enable planning capture during the existing LoCoMo evaluation:

```bash
python -m agent.run \
  --path /models/geologic-memory.pt \
  --dataset_path data/locomo10.json \
  --device cpu \
  --output_dim 384 \
  --rank 16 \
  --interval 5 \
  --visualize_planning \
  --visualization_dir visualizations/locomo_planning \
  --visualization_max_plans 50
```

New runs group every diagnostic by conversation sample and question number:

```text
visualizations/locomo_planning/
└── locomo_sample_0/
    └── question_001/
        ├── retrieval_wasserstein.png
        ├── retrieval_wasserstein.json
        ├── reranking.png
        ├── reranking.json
        ├── gaussians.png
        ├── gaussians.json
        └── gaussians_frames.png
```

Question folders use the one-based LoCoMo question number, not a random hash. Each conversation sample has a separate folder, so its JSON data and plot types stay together.

For every captured LoCoMo question, `agent/visualize_planning.py` embeds the raw query, rewritten dense query, distinct decomposed queries, every target entity, reference answer, and final answer. The main Gaussian report shows:

1. all query, target-entity, and answer Gaussians in one shared 2D projection;
2. low-rank eigenvalues for every orthonormal axis;
3. residual diagonal-variance distributions;
4. raw-query frame alignment against every other plotted distribution.

The frame report shows each `Vᵀ` loading matrix and one 1D Gaussian panel per raw-query frame axis. A rank-8 model therefore produces eight directional panels. Every query, target entity, and answer distribution is projected onto the same raw-query axes, making its mean offset and variance directly comparable. The variance uses the complete low-rank-plus-diagonal covariance `D + V diag(λ)Vᵀ`.

Strictly, a `d×8` matrix `V` is one Stiefel frame containing eight orthonormal axes, not eight separate frames. The JSON records the retrieval plan, all source texts and labels, answers, state statistics, Gram matrices, axis alignments, and all directional mean/variance values.

## Wasserstein FAISS retrieval hits

The dense retrieval report uses the values actually returned by the FAISS `IndexFlatL2` search. Both ingested memories and query variants are flattened as:

```text
[μ, sqrt(diag Σ)]
```

Therefore the squared L2 distance is the diagonal/commuting Gaussian Wasserstein proxy:

```text
W₂_proxy² = ||μq - μn||²
          + ||sqrt(diag Σq) - sqrt(diag Σn)||²
```

The report focuses on retrieval behavior rather than arbitrary embedding differences:

1. Wasserstein-proxy distance by FAISS hit rank for the dense query, each hop, and the multi-hop intersection when present;
2. a query-variant by memory-hit distance heatmap;
3. mean and standard-deviation-profile contributions for the strongest unique hits;
4. hit consistency across query variants using reciprocal rank and hit count.

The JSON retains the memory text, node ID, FAISS index, query variant, hit rank, raw FAISS distance, and both distance components. `--visualization_max_candidates` limits captured/plotted hits per query variant. Retrieval behavior is unchanged: the current database still searches all indexed nodes before its later reranking stage.

## Test only the plotting layer

```bash
python -m pytest -q tests/test_visualizations.py
```

## Geometric energy and reranker report

The LoCoMo visualization run also writes the existing reranking report inside each question folder:

```text
reranking.png
reranking.json
```

Limit the displayed candidate count independently from the captured-question count:

```bash
python -m agent.run \
  --path /models/geologic-memory.pt \
  --dataset_path data/locomo10.json \
  --visualize_query_answers \
  --visualization_max_plans 50 \
  --visualization_max_candidates 30
```

The report shows:

1. stage-1 geometric energy split into squared mean energy and weighted covariance Frobenius energy;
2. stage-2 reranker energy split into squared mean energy and diagonal standard-deviation-profile energy;
3. rank movement from stage 1 to stage 2 and then to chronological output position;
4. agreement or disagreement between the two energy functions, with selected memories highlighted.

The adjacent JSON retains every candidate, even when the PNG displays only the top configured candidates. Each record includes memory text, timestamp, both component scores, both ranks, selection rank, and chronological output position.

Current formulas are:

```text
stage 1 = ||μq - μn||² + 0.5 ||Σq - Σn||²F
stage 2 = ||μq - μn||² + ||sqrt(diag Σq) - sqrt(diag Σn)||²
```

The current agent scoring implementation constructs covariance from `VVᵀ + D` and does not apply the learned low-rank `λ` values. The report records this explicitly. If `λ` should affect retrieval, change and evaluate the scorer separately rather than silently interpreting these plots as lambda-weighted energy.
