import json

import numpy as np

from finetune.visualize_msmarco_samples import (
    flatten_groups,
    retrieval_records,
    sample_msmarco_groups,
)


def test_msmarco_sampling_and_flattening(tmp_path):
    shard = tmp_path / "samples.jsonl"
    rows = [
        {
            "query": f"query {index}",
            "positive": f"positive {index}",
            "negatives": [
                f"negative {index} a",
                f"negative {index} b",
                f"negative {index} c",
            ],
        }
        for index in range(6)
    ]
    shard.write_text("\n".join(json.dumps(row) for row in rows))

    groups = sample_msmarco_groups(
        [shard], groups=3, negatives_per_query=2, scan_records=6, seed=9
    )
    records = flatten_groups(groups)

    assert len(groups) == 3
    assert len(records) == 12
    assert [record["role"] for record in records[:4]] == [
        "query", "positive", "negative", "negative"
    ]
    assert all(len(group["negatives"]) == 2 for group in groups)


def test_retrieval_metrics_rank_positive_against_same_group_negatives():
    groups = [{
        "query": "query",
        "positive": "positive",
        "negatives": ["negative a", "negative b"],
    }]
    records = flatten_groups(groups)
    mean_distances = np.zeros((4, 4))
    bures_distances = np.zeros((4, 4))
    mean_distances[0, 1:] = mean_distances[1:, 0] = [0.1, 0.5, 0.8]
    bures_distances[0, 1:] = bures_distances[1:, 0] = [1.0, 2.0, 3.0]

    retrieval, metrics = retrieval_records(
        records, mean_distances, bures_distances
    )

    positive = next(item for item in retrieval if item["role"] == "positive")
    assert positive["mean_squared_distance_rank"] == 1
    assert positive["bures_squared_distance_rank"] == 1
    assert metrics["mean_positive_top1_rate"] == 1.0
    assert metrics["bures_positive_top1_rate"] == 1.0
