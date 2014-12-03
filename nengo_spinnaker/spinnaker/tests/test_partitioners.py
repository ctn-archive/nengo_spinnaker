from nengo_spinnaker.spinnaker.partitioners import Slice


def test_slice():
    """Test creating and querying slice objects."""
    # Single atom
    sl1 = Slice(0)
    assert sl1.start == 0
    assert sl1.stop == 1
    assert sl1.n_atoms == 1
    assert sl1.as_slice == slice(sl1.start, sl1.stop, 1)

    # Multiple atoms, offset
    sl2 = Slice(10, 20)  # Contains 10 atoms
    assert sl2.start == 10
    assert sl2.stop == 20
    assert sl2.n_atoms == 10
    assert sl2.as_slice == slice(sl2.start, sl2.stop, 1)

    assert "10" in repr(sl2)
    assert "20" in repr(sl2)
    assert "slice" not in repr(sl2)
