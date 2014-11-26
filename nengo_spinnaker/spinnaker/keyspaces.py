"""A system for defining and representing routing keys.

See the :py:class:`~.spinnaker.keyspaces.Keyspace` class.
"""

from collections import OrderedDict

from math import log


class Keyspace(object):
    """Defines the format & value of routing keys for SpiNNaker.

    Notes
    -----
    This model presumes that routing keys are made up of a number of (unsigned,
    integral) fields. For example, there might be a field which specifies which
    chip a message is destined for, another which specifies the core and
    another which specifies the type of message::

        # Create a new 32-bit Keyspace
        ks = Keyspace(32)

        ks.add_field("chip")
        ks.add_field("core")
        ks.add_field("type")

    Given a keyspace with some fields defined, we can define the value of these
    fields to define a specific key::

        # Returns a new `Keyspace` object with the specified fields set
        # accordingly.
        start_master = ks(chip = 0, core = 1, type = 0x01)

    We can also specify just some keys at a time. This means that the fields
    can be specified at different points in execution where it makes most
    sense. As an example::

        master_core = ks(chip = 0, core = 1)

        # chip = 0, core = 1, type = 0x01 (same as previous example)
        start_master = master_core(type = 0x01)

        # chip = 0, core = 1, type = 0x02
        stop_master = master_core(type = 0x02)

    Fields can also exist in a hierarchical structure. For example, we may have
    a field which defines that a key is destined for an external device which
    expects a different set of fields to those for internal use::

        # Starting a new example...
        ks2 = Keyspace(32)

        ks2.add_field("external")

        ks2_internal = ks2(external = 0)
        ks2_internal.add_field("chip")
        ks2_internal.add_field("core")
        ks2_internal.add_field("type")

        ks2_external = ks2(external = 1)
        ks2_internal.add_field("device_id")
        ks2_internal.add_field("command")

        # Keys can be derived from the top-level object
        example_internal = ks(external = 0, chip = 0, core = 1, type = 0x01)
        example_external = ks(external = 1, device_id = 0xBEEF, command = 0x0)

    Now, whenever the `external` field is '0' we have fields `external`,
    `chip`, `core` and `type`. Whenever the `external` field is '1' we have
    fields `external`, `device_id` and `command`. In this example, the
    `device_id` and `command` fields are free to overlap with the `chip`,
    `core` and `type` fields since they are never present in the same key.

    APIs making use of the `Keyspace` module can use this mechanism to create
    'holes' in their keyspace within which API users can add their fields which
    are guaranteed not to collide with the API's key space. For example, we
    could allow developers to set up their own custom fields for their specific
    devices by exposing the following `Keyspace`::

        ks2(external = 1)

    Note only one `Keyspace` object should ever be directly constructed in an
    application from which all new keyspaces are produced ensuring that all
    fields are defined in a non-conflicting manner.

    Certain fields may have fixed lengths and positions within a routing key,
    for example, those required by a hardware peripheral. Conversely, some
    fields' lengths and positions are unimportant and are only required to be
    long enough to represent the maximum value held by a key. Lets re-write the
    above example but this time lets specify the length and position of the
    fields used by external peripherals while leaving internal fields' lengths
    and positions unspecified. These fields will be automatically assigned a
    free space (working from the least-significant bit upwards) in the keyspace
    large enough for the largest value ever assigned to the field.

        ks3 = Keyspace(32)

        # Top-most bit
        ks3.add_field("external", length = 1, start_at = 31)

        # Length and position to be determined automatically
        ks3_internal = ks3(external = 0)
        ks3_internal.add_field("chip")
        ks3_internal.add_field("core")
        ks3_internal.add_field("type")

        # Manually specified field sizes/positions
        ks3_external = ks3(external = 1)
        ks3_internal.add_field("device_id", length = 16, start_at = 0)
        ks3_internal.add_field("command", length = 4, start_at = 24)

    In order to turn a `Keyspace` whose fields have been given values into an
    actual routing key we can use::

        start_master = ks3(external = 0, chip = 0, core = 1, type = 0x01)
        print(hex(start_master.get_key()))

    We can also generate a mask which selects only those bits used by fields in
    the key::

        print(hex(start_master.get_mask()))

    Generating a routing key requires that all fields involved have fixed
    lengths and positions. As a result, any fields in the key whose length or
    position is undefined will be assigned one. The main side-effect of this is
    that the field lengths chosen will be large enough only for the largest
    value ever assigned to that field up until that point in the program. In
    the above example, that means the fields `chip`, `core` and `type` will
    have been assigned a length of 1. Attempting to define a key with these
    fields larger than 1 will now fail. As a result, users are advised to
    define all field values in one phase of execution and then generate keys in
    a second distinct phase.

    With keys broken down into fields, routing can also be simplified by only
    routing based on only a subset of fields. Continuing our example we need
    only route based on the `external`, `device_id`, `chip` and `core` fields.
    In fact, we can route entirely based on `external`, `device_id` and `chip`
    when we're not on the target chip. If we re-write our keyspace definition
    one final time we can apply tags to these subsets of fields to enable us to
    easily generate keys/masks based only on these fields::

        ks4 = Keyspace(32)

        ks4.add_field("external", length = 1, start_at = 31,
            tags = "routing local_routing")

        ks4_internal = ks4(external = 0)
        ks4_internal.add_field("chip", tags = "routing local_routing")
        ks4_internal.add_field("core", tags = "local_routing")
        ks4_internal.add_field("type")

        ks4_external = ks4(external = 1)
        ks4_internal.add_field("device_id", length = 16, start_at = 0,
            tags = "routing local_routing")
        ks4_internal.add_field("command", length = 4, start_at = 24)

        start_master = ks4(external = 0, chip = 0, core = 1, type = 0x01)

        # Keys/masks for the target chip
        print(hex(start_master.get_key(tag = "local_routing")))
        print(hex(start_master.get_mask(tag = "local_routing")))

        # Keys/masks for other chips
        print(hex(start_master.get_key(tag = "routing")))
        print(hex(start_master.get_mask(tag = "routing")))

        device = ks4(external = 1, device_id = 12)

        # Keys/masks for a device (note that we don't need to define the
        # command field since it does not have the routing tag.
        print(hex(device.get_key(tag = "routing")))
        print(hex(device.get_mask(tag = "routing")))

        # Equivalently:
        print(hex(device.get_key(tag = "local_routing")))
        print(hex(device.get_mask(tag = "local_routing")))
    """

    def __init__(self, length=32, fields=None, field_values=None):
        """Create a new Keyspace.

        Parameters
        ----------
        length : int
            The total number of bits in routing keys.
        fields : dict
            For internal use only. The shared, global field dictionary.
        field_values : dict
            For internal use only. Mapping of field-identifier to value.
        """
        self.length = length

        # Field definitions (globally shared by all descendents of a keyspace).
        # An :py:class:`collections.OrderedDict` which maps human-friendly
        # field-identifiers (e.g. strings) to corresponding
        # :py:class:`~.spinnaker.keyspaces.Keyspace.Field` instances. Insertion
        # ordering is maintained to make auto-allocation of field positions
        # more predictable.
        self.fields = fields if fields is not None else OrderedDict()

        if field_values is not None:
            self.field_values = field_values
        else:
            self.field_values = dict()

    def add_field(self, identifier, length=None, start_at=None, tags=None):
        """Add a new field to the Keyspace.

        If any existing fields' values are set, the newly created field will
        become a child of those fields. This means that this field will exist
        only when the parent fields' values are set as they are currently.

        Parameters
        ----------
        identifier : str
            A identifier for the field. Must be a valid python identifier.
            Field names must be unique and users are encouraged to sensibly
            name-space fields in the `prefix_` style to avoid collisions.
        length : int or None
            The number of bits in the field. If *None* the field will be
            automatically assigned a length long enough for the largest value
            assigned.
        start_at : int or None
            0-based index of least significant bit of the field within the
            keyspace. If *None* the field will be automatically located in free
            space in the keyspace.
        tags : string or collection of strings or None
            A (possibly empty) set of tags used to classify the field.  Tags
            should be valid Python identifiers. If a string, the string must be
            a single tag or a space-separated list of tags. If *None*, an empty
            set of tags is assumed.

        Raises
        ------
        :py:class:`ValueError`
            If any the field overlaps with another one or does not fit within
            the Keyspace. Note that fields with unspecified lengths and
            positions do not undergo such checks until their length and
            position become known.
        """
        if identifier in self.fields:
            raise ValueError(
                "Field with identifier '{}' already exists.".format(
                    identifier))

        # Check for zero-length fields
        if length is not None and length <= 0:
            raise ValueError("Fields must be at least one bit in length.")

        # Check for fields which don't fit in the keyspace
        if start_at is not None and (
            0 <= start_at >= self.length
                or (length is not None and start_at + length > self.length)):
                raise ValueError(
                    "Field doesn't fit within {}-bit keyspace.".format(
                        self.length))

        # Check for fields which occupy the same bits
        if start_at is not None:
            for other_identifier, other_field in self._potential_fields():
                if other_field.start_at is not None:
                    if (start_at + (length or 1)) > other_field.start_at and \
                        (other_field.start_at + (other_field.length or 1)) \
                            > start_at:
                            raise ValueError(
                                "Field '{}' (range {}-{}) "
                                "overlaps field '{}' (range {}-{})".format(
                                    identifier,
                                    start_at,
                                    start_at + (length or 1),
                                    other_identifier,
                                    other_field.start_at,
                                    other_field.start_at + (
                                        other_field.length or 1)))

        if type(tags) is str:
            tags = set(tags.split())
        elif tags is None:
            tags = set()
        else:
            tags = set(tags)

        self.fields[identifier] = Keyspace.Field(
            length, start_at, tags, dict(self.field_values))

    def __call__(self, **field_values):
        """Return a new Keyspace instance with fields assigned values as
        specified in the keyword arguments.

        Returns
        -------
        :py:class:`~.spinnaker.keyspaces.Keyspace`
            A `Keyspace` derived from this one but with the specified fields
            assigned a value.

        Raises
        ------
        :py:class:`ValueError`
            If any field has already been assigned a value or a field is
            specified which is not present given other fields' values.
        """
        # Ensure fields exist
        for identifier in field_values.keys():
            if identifier not in self.fields:
                raise ValueError("Field '{}' not defined.".format(identifier))

        # Accumulate all field values checking to ensure no value is
        # overwritten
        for identifier, value in self.field_values.items():
            if identifier in field_values:
                raise ValueError(
                    "Field '{}' already has value.".format(identifier))
            else:
                field_values[identifier] = value

        # Ensure no fields are specified which are not enabled
        enabled_fields = dict(self._enabled_fields(field_values))
        for identifier in field_values:
            if identifier not in enabled_fields:
                raise ValueError("Field '{}' requires that {}.".format(
                    identifier,
                    ", ".join("'{}' == {}".format(cond_ident, cond_val)
                                 for cond_ident, cond_val
                                 in self.fields[identifier].conditions.items()
                                 if field_values.get(cond_ident, None)
                                 != cond_val)))

        # Ensure values are within range and record maximum observed values
        for identifier, value in field_values.items():
            field = self.fields[identifier]

            if field.length is not None and value >= (1 << field.length):
                raise ValueError(
                    "Value {} too large for {}-bit field '{}'.".format(
                        value, field.length, identifier))
            elif field.length is not None and value < 0:
                raise ValueError("Negative values not permitted in fields.")

            field.max_value = max(field.max_value, value)

        return Keyspace(self.length, self.fields, field_values)

    def __getattr__(self, identifier):
        """Get the value of a field.

        Returns
        -------
        int or None
            The value of the field (or None if the field has not been given a
            value).

        Raises
        ------
        AttributeError
            If the field requested does not exist or is not available given
            current field values.
        """
        self._assert_field_available(identifier)

        return self.field_values.get(identifier, None)

    def get_key(self, tag=None, field=None):
        """Generate a key whose fields are set appropriately and with all other
        bits set to zero.

        Calling this method will cause all defined fields whose length or
        position is not defined to become fixed. As a result, users should only
        call this method after all field values have been assigned to keyspaces
        otherwise fields may be fixed at an inadequate size.

        Parameters
        ----------
        tag : str
            Optionally specifies that the key should only include fields with
            the specified tag.
        field : str
            Optionally specifies that the key should only include the specified
            field.

        Raises
        ------
        :py:class:`ValueError`
            If a field whose length or position was unspecified does not fit
            within the Keyspace.
        """
        assert not (tag is not None and field is not None)

        enabled_field_idents = [
            i for (i, f) in self._enabled_fields()
            if tag is None or tag in f.tags]

        if tag is not None:
            self._assert_tag_exists(tag)

        if field is not None:
            self._assert_field_available(field)
            selected_field_idents = [field]
        else:
            selected_field_idents = [
                i for i in enabled_field_idents
                if tag is None or tag in self.fields[i].tags]

        # Check all fields are present
        defined_fields = self.field_values.keys()
        missing_fields = set(selected_field_idents) - set(defined_fields)
        if missing_fields:
            raise ValueError(
                "Cannot generate key with undefined fields {}.".format(
                    ", ".join(missing_fields)))

        self._assign_field_bits(enabled_field_idents)

        key = 0
        for identifier in selected_field_idents:
            key |= (self.field_values[identifier] <<
                    self.fields[identifier].start_at)

        return key

    def get_mask(self, tag=None, field=None):
        """Get the for all fields which exist in the current keyspace.

        Calling this method will cause all defined fields whose length or
        position is not defined to become fixed. As a result, users should only
        call this method after all field values have been assigned to keyspaces
        otherwise fields may be fixed at an inadequate size.

        Parameters
        ----------
        tag : str
            Optionally specifies that the mask should only include fields with
            the specified tag.
        field : str
            Optionally specifies that the mask should only include the
            specified field.

        Raises
        ------
        :py:class:`ValueError`
            If a field whose length or position was unspecified does not fit
            within the Keyspace.
        """
        assert not (tag is not None and field is not None)

        enabled_field_idents = [
            i for (i, f) in self._enabled_fields()]

        if tag is not None:
            self._assert_tag_exists(tag)

        if field is not None:
            self._assert_field_available(field)
            selected_field_idents = [field]
        else:
            selected_field_idents = [
                i for i in enabled_field_idents
                if (tag is None) or (tag in self.fields[i].tags)]

        self._assign_field_bits(enabled_field_idents)

        mask = 0
        for identifier in selected_field_idents:
            field = self.fields[identifier]
            mask |= ((1 << field.length) - 1) << field.start_at

        return mask

    def __repr__(self):
        """Produce a human-readable representation of this Keyspace and its
        current value.
        """
        enabled_field_idents = [
            i for (i, f) in self._enabled_fields()]

        return "<{}-bit Keyspace {}>".format(
            self.length,
            ", ".join(
                "'{}':{}".format(identifier,
                                 self.field_values.get(identifier, "?"))
                for identifier in enabled_field_idents))

    class Field(object):
        """Internally used class which defines a field.
        """

        def __init__(self, length=None, start_at=None, tags=None,
                     conditions=None, max_value=1):
            """Field definition used internally by
            :py:class:`~.spinnaker.keyspaces.Keyspace`.

            Parameters/Attributes
            ---------------------
            length : int
                The number of bits in the field. *None* if this should be
                determined based on the values assigned to it.
            start_at : int
                0-based index of least significant bit of the field within the
                keyspace.  *None* if this field is to be automatically placed
                into an unused area of the keyspace.
            tags : set
                A (possibly empty) set of tags used to classify the field.
            conditions : dict
                Specifies conditions when this field is valid. If empty, this
                field is always defined. Otherwise, keys in the dictionary
                specify field-identifers and values specify the desired value.
                All listed fields must match the specified values for the
                condition to be met.
            max_value : int
                The largest value ever assigned to this field (used for
                automatically determining field sizes.
            """
            self.length = length
            self.start_at = start_at
            self.tags = tags if tags is not None else set()
            self.conditions = conditions if conditions is not None else set()
            self.max_value = max_value

    def _assert_field_available(self, identifier):
        """Raise a human-readable ValueError if the specified field does not
        exist or is not enabled by the current field values.
        """
        if identifier not in self.fields:
            raise AttributeError(
                "Field '{}' does not exist.".format(identifier))
        elif identifier not in (i for (i, f) in self._enabled_fields()):
            raise AttributeError("Field '{}' requires that {}.".format(
                identifier,
                ", ".join("'{}' == {}".format(cond_ident, cond_val)
                          for cond_ident, cond_val
                          in self.fields[identifier].conditions.items()
                          if self.field_values.get(cond_ident, None)
                          != cond_val)))

    def _assert_tag_exists(self, tag):
        """Raise a human-readable ValueError if the supplied tag is not used by
        any enabled field.
        """

        for identifier, field in self._enabled_fields():
            if tag in field.tags:
                return
        raise ValueError("Tag '{}' does not exist.".format(tag))

    def _enabled_fields(self, field_values=None):
        """Generator of (identifier, field) tuples which iterates over the
        fields which can be set based on the currently specified field values.

        Parameters
        ----------
        field_values : dict or None
            Dictionary of field identifier to value mappings to use in the
            test. If *None*, uses `self.field_values`.
        """
        if field_values is None:
            field_values = self.field_values

        for identifier, field in self.fields.items():
            if field.conditions:
                met = True
                for cond_field, cond_value in field.conditions.items():
                    if field_values.get(cond_field, None) != cond_value:
                        met = False
                        break
                if met:
                    yield (identifier, field)
            else:
                yield (identifier, field)

    def _potential_fields(self, field_values=None):
        """Generator of (identifier, field) tuples iterating over every field
        which could potentially be defined given the currently specified field
        values.

        Parameters
        ----------
        field_values : dict or None
            Dictionary of field identifier to value mappings to use in the
            test. If *None*, uses `self.field_values`.
        """
        if field_values is None:
            field_values = self.field_values

        # The ordered dictionary of fields under a partial ordering of
        # dependency.  Simply accumulate blocked fields which either depend on
        # a field value which has been defined as a non-matching value or
        # depend on fields which have already been blocked.
        blocked = set()
        for identifier, field in self.fields.items():
            if field.conditions:
                met = True
                for cond_field, cond_value in field.conditions.items():
                    if cond_field in blocked or \
                       field_values.get(cond_field, None) != cond_value:
                        met = False
                        break
                if met:
                    yield (identifier, field)
                else:
                    blocked.add(identifier)
            else:
                yield (identifier, field)

    def _assign_field_bits(self, field_identifiers):
        """Assign a position & length to any listed fields which do not have
        one.

        Parameters
        ----------
        fields : collection
            The identifiers of the fields to check.
        """
        # Get the currently allocated bits which are in visible fields
        assigned_bits = 0
        for identifier, field in self._enabled_fields():
            if field.length is not None and field.start_at is not None:
                assigned_bits |= ((1 << field.length) - 1) << field.start_at

        # Iterate through the fields in the order they appear in the
        # OrderedDict since this is sorted by the partial ordering imposed by
        # the hierarchy. This ordering ensures that fields higher up the
        # hierarchy are packed together.
        for identifier, field in self.fields.items():
            if identifier in field_identifiers:
                # Assign lengths based on values
                if field.length is None:
                    field.length = int(log(field.max_value, 2)) + 1

                # Assign the next available free space starting from the least
                # significant bit
                if field.start_at is None:
                    for bit in range(0, self.length - field.length):
                        field_bits = ((1 << field.length) - 1) << bit
                        if not (assigned_bits & field_bits):
                            field.start_at = bit
                            assigned_bits |= field_bits
                            break

                    # Fail
                    if field.start_at is None:
                        field.start_at = self.length

                # Check that the field is within the keyspace
                if field.start_at + field.length > self.length:
                    raise ValueError(
                        "{}-bit field '{}' "
                        "does not fit in keyspace.".format(
                            field.length, identifier))
