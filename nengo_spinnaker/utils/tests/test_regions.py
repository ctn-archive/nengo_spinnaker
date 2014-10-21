from .. import regions as region_utils


def test_bitfield_based_recording_region_none():
    bbrr = region_utils.BitfieldBasedRecordingRegion(None)
    assert not bbrr.in_dtcm
    assert bbrr.unfilled

    assert bbrr.sizeof(slice(0, 10)) == 0
    assert bbrr.sizeof(slice(0, 32)) == 0


def test_bitfield_based_recording_region_normal():
    bbrr = region_utils.BitfieldBasedRecordingRegion(1)

    assert bbrr.sizeof(slice(0, 10)) == 1
    assert bbrr.sizeof(slice(0, 33)) == 2
    assert bbrr.sizeof(slice(32, 65)) == 2
