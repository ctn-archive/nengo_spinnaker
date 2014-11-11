from pacman.model.graph_mapper.slice import Slice
from .. import regions as region_utils


def test_bitfield_based_recording_region_none():
    bbrr = region_utils.BitfieldBasedRecordingRegion(None)
    assert not bbrr.in_dtcm
    assert bbrr.unfilled

    assert bbrr.sizeof(Slice(0, 9)) == 0
    assert bbrr.sizeof(Slice(0, 31)) == 0


def test_bitfield_based_recording_region_normal():
    bbrr = region_utils.BitfieldBasedRecordingRegion(1)

    assert bbrr.sizeof(Slice(0, 9)) == 1
    assert bbrr.sizeof(Slice(0, 32)) == 2
    assert bbrr.sizeof(Slice(32, 64)) == 2
