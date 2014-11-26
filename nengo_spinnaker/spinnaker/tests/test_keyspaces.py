import pytest

import random

from ..keyspaces import Keyspace


def test_keyspace_add_field():
    # Assert that we can't create a new keyspace with fixed fields beyond its
    # length
    ks = Keyspace(64)
    with pytest.raises(ValueError):
        ks.add_field("out_of_range", start_at = 64)
    with pytest.raises(ValueError):
        ks.add_field("out_of_range", start_at = 128)
    with pytest.raises(ValueError):
        ks.add_field("out_of_range", length = 2, start_at = 63)
    with pytest.raises(ValueError):
        ks.add_field("out_of_range", length = 2, start_at = 128)
    
    # Check fields must have non-zero lengths
    with pytest.raises(ValueError):
        ks.add_field("zero_length", length = 0, start_at = 0)
    
    # Check we can't create overlapping fixed fields
    ks.add_field("obstruction", length = 8, start_at = 8)
    with pytest.raises(ValueError):
        ks.add_field("obstructed", length = 8, start_at = 8)
    with pytest.raises(ValueError):
        ks.add_field("obstructed", length = 16, start_at = 0)
    with pytest.raises(ValueError):
        ks.add_field("obstructed", length = 12, start_at = 0)
    with pytest.raises(ValueError):
        ks.add_field("obstructed", length = 16, start_at = 8)
    with pytest.raises(ValueError):
        ks.add_field("obstructed", length = 12, start_at = 12)
    with pytest.raises(ValueError):
        ks.add_field("obstructed", length = 4, start_at = 10)
    
    # Check we can't create fields with the same name but which otherwise don't
    # clash
    with pytest.raises(ValueError):
        ks.add_field("obstruction", length = 1, start_at = 0)


def test_keyspace_masks():
    # Create a new keyspace and check that appropriate masks can be retrieved
    # from it, filtering by tag.
    ks = Keyspace(64)
    ks.add_field("bottom", length = 32, start_at = 0, tags = "Bottom Both")
    ks.add_field("top", length = 8, start_at = 56, tags = "Top Both")

    assert ks.get_mask() == 0xFF000000FFFFFFFFl
    assert ks.get_mask("Both") == 0xFF000000FFFFFFFFl
    assert ks.get_mask("Bottom") == 0x00000000FFFFFFFFl
    assert ks.get_mask("Top") == 0xFF00000000000000l


def test_keyspace_tag_formats():
    # Test the ability to define tags in different ways
    ks = Keyspace(6)
    ks.add_field("a", length = 1, start_at = 0)
    ks.add_field("b", length = 1, start_at = 1, tags = "B")
    ks.add_field("c", length = 1, start_at = 2, tags = "C C_")
    ks.add_field("d", length = 1, start_at = 3, tags = ["D"])
    ks.add_field("e", length = 1, start_at = 4, tags = ["E", "E_"])

    assert ks.get_mask() == 0x1F
    assert ks.get_mask("A") == 0x00
    assert ks.get_mask("B") == 0x02
    assert ks.get_mask("C") == 0x04
    assert ks.get_mask("C_") == 0x04
    assert ks.get_mask("D") == 0x08
    assert ks.get_mask("E") == 0x10
    assert ks.get_mask("E_") == 0x10


def test_keyspace_keys():
    # Create a new keyspace and check that keys can be filled in and extracted
    ks = Keyspace(32)
    ks.add_field("a", length = 8, start_at = 0, tags = "A All")
    ks.add_field("b", length = 8, start_at = 8, tags = "B All")
    ks.add_field("c", length = 8, start_at = 16, tags = "C All")
    
    # Shouldn't be able to access a field which isn't defined
    with pytest.raises(AttributeError):
        _ = ks.d
    
    # Shouldn't be able to set a field which isn't defined
    with pytest.raises(ValueError):
        _ = ks(d = 123)
    
    # Shouldn't be able to get values before they're assigned
    with pytest.raises(ValueError):
        ks.get_key()
    with pytest.raises(ValueError):
        ks.get_key("All")
    with pytest.raises(ValueError):
        ks.get_key("A")
    with pytest.raises(ValueError):
        ks.get_key("B")
    with pytest.raises(ValueError):
        ks.get_key("C")
    
    # Should just get None for unspecified fields
    assert ks.a is None
    assert ks.b is None
    assert ks.c is None
    
    # Shouldn't be able to set value out of range
    with pytest.raises(ValueError):
      ks_a = ks(a = 0x100)
    with pytest.raises(ValueError):
      ks_a = ks(a = -1)
    
    # Should now be able to get that field but not others
    ks_a = ks(a = 0xAA)
    assert ks_a.a == 0xAA
    assert ks_a.b is None
    assert ks_a.c is None
    assert ks_a.get_key("A") == 0x000000AA
    with pytest.raises(ValueError):
        ks_a.get_key()
    with pytest.raises(ValueError):
        ks_a.get_key("All")
    with pytest.raises(ValueError):
        ks_a.get_key("B")
    with pytest.raises(ValueError):
        ks_a.get_key("C")
    
    # Should not be able to change a field
    with pytest.raises(ValueError):
        ks_a_ = ks_a(a = 0x00)
    with pytest.raises(ValueError):
        ks_a_ = ks_a(a = 0xAA)
    
    # Fill in all fields, everything should now drop out
    ks_abc = ks_a(b = 0xBB, c = 0xCC)
    assert ks_abc.a == 0xAA
    assert ks_abc.b == 0xBB
    assert ks_abc.c == 0xCC
    assert ks_abc.get_key() == 0x00CCBBAA
    assert ks_abc.get_key("All") == 0x00CCBBAA
    assert ks_abc.get_key("A") == 0x000000AA
    assert ks_abc.get_key("B") == 0x0000BB00
    assert ks_abc.get_key("C") == 0x00CC0000
    
    # Test some special-case values
    ks_a0 = ks(a=0)
    assert ks_a0.get_key("A") == 0x00000000
    ks_a1 = ks(a=1)
    assert ks_a1.get_key("A") == 0x00000001
    ks_aFF = ks(a=0xFF)
    assert ks_aFF.get_key("A") == 0x000000FF


def test_keyspace_hierachy():
    ks = Keyspace(8)
    ks.add_field("always", length = 1, start_at = 7)
    ks.add_field("split", length = 1, start_at = 6)
    
    # Add different fields dependent on the split-bit. This checks that creating
    # fields in different branches of the heirachy doesn't result in clashes.
    ks_s0 = ks(split=0)
    ks_s0.add_field("s0_btm", length = 3, start_at = 0)
    ks_s0.add_field("s0_top", length = 3, start_at = 3)
    
    ks_s1 = ks(split=1)
    ks_s1.add_field("s1_btm", length = 3, start_at = 0)
    ks_s1.add_field("s1_top", length = 3, start_at = 3)
    
    # Shouldn't be able to access child-fields of split before it is defined
    with pytest.raises(AttributeError):
        _ = ks.s0_top
    with pytest.raises(AttributeError):
        _ = ks.s0_btm
    with pytest.raises(AttributeError):
        _ = ks.s1_top
    with pytest.raises(AttributeError):
        _ = ks.s1_btm
    
    # Should be able to define a new key from scratch
    ks_s0_defined = ks(always = 1, split = 0, s0_btm = 3, s0_top = 5)
    assert ks_s0_defined.always == 1
    assert ks_s0_defined.split == 0
    assert ks_s0_defined.s0_btm == 3
    assert ks_s0_defined.s0_top == 5
    assert ks_s0_defined.get_key() == 0b10101011
    assert ks_s0_defined.get_mask() == 0b11111111
    
    # Should be able to order child keys before the parent key too in the
    # arguments without causing a crash
    _ = ks(s0_btm = 3, s0_top = 5, always = 1, split = 0)
    
    # Accessing fields from the other side of the split should fail
    with pytest.raises(AttributeError):
        _ = ks_s0_defined.s1_btm
    with pytest.raises(AttributeError):
        _ = ks_s0_defined.s1_top
    
    # Shouldn't have to define all top-level fields in order to get at split
    # fields
    ks_s1_selected = ks(split = 1)
    assert ks_s1_selected.split == 1
    assert ks_s1_selected.always is None
    assert ks_s1_selected.s1_btm is None
    assert ks_s1_selected.s1_top is None
    
    # Accessing fields from the other side of the split should still fail
    with pytest.raises(AttributeError):
        _ = ks_s1_selected.s0_btm
    with pytest.raises(AttributeError):
        _ = ks_s1_selected.s0_top
    
    # Should not be able to set child fields before parent field is set
    with pytest.raises(ValueError):
        _ = ks(s0_btm = 3, s0_top = 5)


def test_auto_length():
    # Tests for automatic length calculation.
    # Test that fields never given any values just become single-bit fields
    ks_never = Keyspace(8)
    ks_never.add_field("never", start_at = 0)
    assert ks_never.get_mask() == 0x01
    
    # Test that we can't start an auto-length field within an existing field.
    with pytest.raises(ValueError):
        ks_never.add_field("obstructed", start_at = 0)
    
    # Make sure that get_key() also forces a fixing of the size
    ks_once = Keyspace(8)
    ks_once.add_field("once", start_at = 0)
    assert ks_once(once = 0x0F).get_key() == 0x0F
    assert ks_once.get_mask() == 0x0F
    
    # The signle-bit value should happily fit only 0 and 1
    assert ks_never(never = 0).never == 0
    assert ks_never(never = 1).never == 1
    with pytest.raises(ValueError):
      _ = ks_never(never = 2)
    
    # Test that fields can be sized automatically but positioned manually
    ks = Keyspace(64)
    ks.add_field("auto_length", start_at = 32)
    
    # Create a load of keys (of which one is at least 32-bits long)
    for val in [0, 1, 0xDEADBEEF, 0x1234]:
      ks_val = ks(auto_length = val)
      assert ks_val.auto_length == val
    
    # The resulting field should be 32 bits long
    assert ks.get_mask() == 0xFFFFFFFF00000000
    
    # The field's length should now be fixed
    with pytest.raises(ValueError):
        _ = ks(auto_length = 0x100000000)
    
    # Test that fields can be created which when auto-lengthed, don't fit
    ks_long = Keyspace(16)
    ks_long.add_field("too_long", start_at = 0)
    assert ks_long(too_long = 0x10000).too_long == 0x10000
    with pytest.raises(ValueError):
        _ = ks_long.get_mask()
    
    # Test that fields can be generated hierarchically
    ks_h = Keyspace(16)
    ks_h.add_field("split", start_at = 8, tags = "TopLevel")
    ks_h_s0 = ks_h(split = 0)
    ks_h_s0.add_field("s0", start_at = 0)
    ks_h_s2 = ks_h(split = 2)
    ks_h_s2.add_field("s2", start_at = 0)
    
    # Test that when working on on a child on one side of the split, the field
    # sizes don't get fixed on the other.
    ks_h_s0_val = ks_h(split = 0, s0 = 0x10)
    assert ks_h_s0_val.get_mask() == 0x031F
    assert ks_h_s0_val.get_key() == 0x0010
    
    # Make sure this side and the split field are now fixed
    with pytest.raises(ValueError):
        _ = ks_h(split = 0, s0 = 0x3F)
    with pytest.raises(ValueError):
        _ = ks_h(split = 4)
    
    # Make sure the other side of the split is still not fixed (this will fail
    # if it has been fixed since its size would be fixed as 0 as no values have
    # been assigned to it).
    assert ks_h(split = 2, s2 = 0x3F).s2 == 0x3F
    
    # Getting the mask for a higher-level field shouldn't cause the field to
    # become fixed in size either
    assert ks_h.get_mask("TopLevel") == 0x0300
    assert ks_h(split = 2, s2 = 0x7F).s2 == 0x7F

def test_auto_start_at():
    # Test automatic positioning of fixed-length fields
    ks = Keyspace(32)
    
    ks.add_field("a", length = 4, tags = "A")
    ks.add_field("b", length = 4, tags = "B")
    ks.add_field("c", length = 4, tags = "C")
    
    # Test that all fields are allocated at once and in order of insertion (even
    # if requested out of order).
    assert ks.get_mask("C") == 0x00000F00
    assert ks.get_mask("B") == 0x000000F0
    assert ks.get_mask("A") == 0x0000000F
    assert ks.get_mask() == 0x00000FFF
    
    # Test positioning when a space is fully obstructed 
    ks.add_field("full_obstruction", length = 4, start_at = 12)
    ks.add_field("d", length = 4, tags = "D")
    assert ks.get_mask("D") == 0x000F0000
    
    # Test positioning when a space is partially obstructed. The first field
    # will end up after the second field since the second field can fill the
    # gap left by the partial obstruction.
    ks.add_field("partial_obstruction", length = 4, start_at = 22)
    ks.add_field("e", length = 4, tags = "E")
    ks.add_field("f", length = 2, tags = "F")
    assert ks.get_mask("E") == 0x3C000000
    assert ks.get_mask("F") == 0x00300000
    
    # Ensure we can define fields which don't fit when placed
    ks.add_field("last_straw", length = 4)
    with pytest.raises(ValueError):
        _ = ks.get_mask()
    
    # Test that auto-placed fields can be created heirachically
    ks_h = Keyspace(32)
    ks_h.add_field("split", length = 4, tags = "Split")
    
    # Ensure that additional top-level fields are placed before all split
    # fields when defined before
    ks_h.add_field("always_before", length = 4, tags = "AlwaysBefore")
    
    ks_h_s0 = ks_h(split = 0)
    ks_h_s0.add_field("s0", length = 4, tags = "S0")
    ks_h_s1 = ks_h(split = 1)
    ks_h_s1.add_field("s1", length = 8, tags = "S1")
    
    # Check that as soon as we access one side of the split, all fields are
    # fixed.
    assert ks_h(split = 0).get_mask("Split") == 0x0000000F
    with pytest.raises(ValueError):
        ks_h_s0.add_field("obstructed", start_at = 8)
    assert ks_h(split = 0).get_mask("AlwaysBefore") == 0x000000F0
    assert ks_h(split = 0).get_mask("S0") == 0x00000F00
    assert ks_h(split = 0).get_mask() == 0x00000FFF
    
    # But ensure fields on the other side of the field are not
    ks_h_s1.add_field("obstruction", length = 4, start_at = 8, tags = "Obstruction")
    assert ks_h(split = 1).get_mask("Split") == 0x0000000F
    assert ks_h(split = 1).get_mask("AlwaysBefore") == 0x000000F0
    assert ks_h(split = 1).get_mask("Obstruction") == 0x00000F00
    assert ks_h(split = 1).get_mask("S1") == 0x000FF000
    assert ks_h(split = 1).get_mask() == 0x000FFFFF


def test_full_auto():
    # Brief test that automatic placement and length assignment can happen
    # simultaneously
    ks = Keyspace(32)
    ks.add_field("a", tags = "A")
    ks.add_field("b", tags = "B")
    
    # Give the fields values to force their sizes
    ks(a = 0xFF, b = 0xFFF)
    
    # Check the masks come out correctly
    assert ks.get_mask("A") == 0x000000FF
    assert ks.get_mask("B") == 0x000FFF00
    assert ks.get_mask() == 0x000FFFFF


def test_repr():
    # Very rough tests to ensure the string representation of a keyspace is
    # reasonably sane.
    ks = Keyspace(128)
    ks.add_field("always")
    ks.add_field("split")
    ks_s0 = ks(split = 0)
    ks_s0.add_field("s0")
    ks_s1 = ks(split = 1)
    ks_s1.add_field("s1")
    
    # Basic info should appear
    assert "128" in repr(ks)
    assert "Keyspace" in repr(ks)
    
    # Global fields should appear, even if undefined
    assert "'always'" in repr(ks)
    assert "'split'" in repr(ks)
    
    # Values should appear when defined
    assert "12345" in repr(ks(always = 12345))
    
    # Fields behind splits should not be shown if the split field is not defined
    assert "'s0'" not in repr(ks)
    assert "'s1'" not in repr(ks)
    
    # Fields should be shown when split is defined accordingly
    assert "'s0'" in repr(ks_s0)
    assert "'s1'" not in repr(ks_s0)
    assert "'s0'" not in repr(ks_s1)
    assert "'s1'" in repr(ks_s1)
