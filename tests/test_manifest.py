from dataclasses import fields

from signalign.hand.parallel import _batch_preserving_partitions
from signalign.manifest import FrameRecord


def test_inference_record_has_no_reference_annotation_field() -> None:
    names = {field.name for field in fields(FrameRecord)}
    assert names == {
        "sign",
        "sign_class",
        "source_path",
        "source_frame_id",
        "sequence_index",
    }


def test_parallel_partition_preserves_sequential_batches() -> None:
    records = list(range(1493))
    partitions = _batch_preserving_partitions(records, batch_size=8, workers=4)
    assert [item for partition in partitions for item in partition] == records
    boundaries = [sum(map(len, partitions[:index])) for index in range(1, len(partitions))]
    assert all(boundary % 8 == 0 for boundary in boundaries)
