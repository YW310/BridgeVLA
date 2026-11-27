import json
from collections import Counter

from finetune.OHT.data.real_episodes import DATASET_PATH, episode_records
from finetune.OHT.data.real_rgbd import FIXED_CAMERAS


def test_real_episode_split_keeps_transitive_replay_lineage(tmp_path):
    cohorts = ("manual", "manual", "manual", "augmentation", "dagger", "manual", "manual", "manual")
    parents = {1: "session-0", 2: "session-1", 3: "session-2", 4: "session-1"}
    receipts = []
    for index, cohort in enumerate(cohorts):
        source = f"session-{index}"
        metadata = tmp_path / DATASET_PATH / "provenance" / f"episode_{index:06d}" / "metadata.json"
        metadata.parent.mkdir(parents=True)
        parent = parents.get(index, "manual")
        reference = {"source": {"episode": parent}}
        metadata.write_text(json.dumps({"replay": reference, "dagger": reference}), encoding="utf-8")
        receipts.append(dict(episode=source, episode_index=index, group=cohort, rows=20,
                             cameras=[dict(camera=c) for c in FIXED_CAMERAS],
                             parquet=f"data/{index}.parquet", parquet_sha256=f"file-{index}",
                             raw_index_sha256=f"raw-{index}"))
    report = dict(status="converted", episodes=len(receipts), groups=Counter(cohorts), receipts=receipts)
    (tmp_path / "conversion_report.json").write_text(json.dumps(report), encoding="utf-8")

    records = episode_records(tmp_path, seed=0)

    relatives = records[:5]
    assert [row["source_parent"] for row in relatives] == [
        "session-0", "session-0", "session-1", "session-2", "session-1",
    ]
    assert {row["source_root"] for row in relatives} == {"session-0"}
    assert len({row["group"] for row in relatives}) == 1
    assert len({row["split"] for row in relatives}) == 1
    assert all(row["group"] != relatives[0]["group"] for row in records[5:])
