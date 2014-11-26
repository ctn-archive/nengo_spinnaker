"""Tools for formating and defining routing keys.
"""

from collections import namedtuple, OrderedDict

from math import log

class Keyspace(object):
    """Defines the format & value of routing keys.
    
    This object is used to define what bit-fields & encodings are contained
    within routing keys and, subsequently, to generate keys/masks based on these
    defintions.
    """
    
    class Field(object):
        def __init__(self, length = None, start_at = None, tags = None,
            conditions = None, max_value = 1):
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
            self.length   = length
            self.start_at = start_at
            self.tags = tags if tags is not None else set()
            self.conditions = conditions if conditions is not None else set()
            self.max_value = max_value
    
    def __init__(self, length = 32, fields = None, field_values = None):
        """Create a new keyspace.
        
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
        # ordering is maintained to make auto-allocation of field positions more
        # predictable.
        self.fields = fields if fields is not None else OrderedDict()
        
        self.field_values = field_values if field_values is not None else dict()
    
    def _enabled_fields(self, field_values):
        """Generator over the list of (identifier, field) tuples which are
        relevent to the currently specified field values.
        
        Parameters
        ----------
        field_values : dict
            Dictionary of field identifier to value mappings to use in the test.
        """
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
    
    def _potential_fields(self, field_values):
        """Generator over the list of (identifier, field) tuples identifiers
        which are not blocked by their dependencies' field values.
        
        Parameters
        ----------
        field_values : dict
            Dictionary of field identifier to value mappings to use in the test.
        """
        # The ordered dictionary of fields is partially ordered by dependency.
        # Simply accumulate blocked fields which either depend on a field value
        # which has been defined as a non-matching value or depend on fields
        # which have already been blocked.
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
        """Assign a position & length to any fields which do not have one.
        
        Parameters
        ----------
        fields : collection
            The identifiers of the fields to check.
        """
        # Get the currently allocated bits which are in visible fields
        assigned_bits = 0
        for identifier, field in self._enabled_fields(self.field_values):
            if field.length is not None and field.start_at is not None:
                assigned_bits |= ((1 << field.length) - 1) << field.start_at
        
        # Iterate through the fields in the order they appear in the OrderedDict
        # since this is sorted by the partial ordering imposed by the hierachy.
        # This ordering ensures that fields higher up the heirachy are packed
        # together.
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
    
    def add_field(self, identifier, length = None, start_at = None, tags = None):
        """Add a new field to the keyspace.
        
        The field defined will only be present for keys under the currently
        defined field values.
        
        Parameters
        ----------
        identifier : str
            A identifier for the field. Must be a valid python identifier. Field
            names must be unique and users are encouraged to sensibly name-space
            fields in the `prefix_` style to avoid collisions.
        length : int
            The number of bits in the field. *None* if this should be
            determined based on the values assigned to it.
        start_at : int
            0-based index of least significant bit of the field within the
            keyspace.  *None* if this field is to be automatically placed
            into an unused area of the keyspace.
        tags : set
            A (possibly empty) set of tags used to classify the field.
        """
        if identifier in self.fields:
            raise ValueError(
                "Field with identifier '{}' already exists.".format(identifier))
        
        # Check for zero-length fields
        if length is not None and length <= 0:
            raise ValueError("Fields must be at least one bit in length.")
        
        # Check for fields which don't fit in the keyspace
        if start_at is not None and (
            0 <= start_at >= self.length
            or (length is not None and start_at + length > self.length)):
            raise ValueError("Field doesn't fit within {}-bit keyspace.".format(
                self.length))
        
        # Check for fields which occupy the same bits
        if start_at is not None:
            for other_identifier, other_field in self._potential_fields(
                self.field_values):
                if other_field.start_at is not None:
                    if (start_at + (length or 1)) > other_field.start_at and \
                        (other_field.start_at + (other_field.length or 1)) > start_at:
                        raise ValueError(
                            "Field '{}' (range {}-{}) "
                            "overlaps field '{}' (range {}-{})".format(
                                identifier,
                                start_at,
                                start_at + (length or 1),
                                other_identifier,
                                other_field.start_at,
                                other_field.start_at + (other_field.length or 1)))
        
        if type(tags) is str:
            tags = set(tags.split())
        elif tags is None:
            tags = set()
        else:
            tags = set(tags)
        
        self.fields[identifier] = Keyspace.Field(
            length, start_at, tags, dict(self.field_values))
    
    def __call__(self, **field_values):
        """Return a new Keyspace instance with the specified fields values
        assigned.
        
        Returns
        -------
        :py:class:`~.spinnaker.keyspaces.Keyspace`
            A keyspace derrived from this one but with the specified fields
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
        
        # Accumulate all field values checking to ensure no value is overwritten
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
            
            if field.length is not None and value >= (1<<field.length):
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
        int
            The value of the field (or None if not defined).
        
        Raises
        ------
        AttributeError
            If the field requested does not exist or is not available given
            current field values.
        """
        enabled_field_idents = [
            i for (i,f) in self._enabled_fields(self.field_values)]
        
        if identifier in enabled_field_idents:
            return self.field_values.get(identifier, None)
        else:
            if identifier in self.fields:
                raise AttributeError("Field '{}' requires that {}.".format(
                    identifier,
                    ", ".join("'{}' == {}".format(cond_ident, cond_val)
                        for cond_ident, cond_val
                            in self.fields[identifier].conditions.items()
                                if self.field_values.get(cond_ident, None)
                                    != cond_val)))
            else:
                raise AttributeError(
                    "Field '{}' does not exist.".format(identifier))
    
    def get_key(self, tag = None):
        """Get the key with fields with the supplied tag set according to this
        keyspace.
        
        Calling this method will cause any fields whose length or position is
        not defined to become fixed. As a result, users should take care to call
        this method after assigning all keys to the variable-sized fields used
        in this key.
        
        Parameters
        ----------
        tag : str
            Optionally specifies that the key should only contain fields with
            the specified tag.
        
        Raises
        ------
        """
        enabled_field_idents = [
            i for (i,f) in self._enabled_fields(self.field_values)
            if tag is None or tag in f.tags]
        selected_field_idents = [
            i for i in self.field_values.keys()
            if tag is None or tag in self.fields[i].tags]
        
        # Check all fields are present
        missing_fields = set(enabled_field_idents) - set(selected_field_idents)
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
    
    def get_mask(self, tag = None):
        """Get the mask for the fields with the given tag and selected by the
        current keyspace values.
        
        Calling this method will cause any fields whose length or position is
        not defined to become fixed. As a result, users should take care to call
        this method after assigning all keys to the variable-sized fields used
        in this key.
        
        Parameters
        ----------
        tag : str
            Optionally specifies that the key should only contain fields with
            the specified tag.
        
        Raises
        ------
        """
        enabled_field_idents = [
            i for (i,f) in self._enabled_fields(self.field_values)]
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
        """Produce a human-readable representation of this keyspace.
        """
        enabled_field_idents = [
            i for (i,f) in self._enabled_fields(self.field_values)]
        
        return "<{}-bit Keyspace {}>".format(
            self.length,
            ", ".join(
                "'{}':{}".format(identifier,
                    self.field_values.get(identifier, "?"))
                for identifier in enabled_field_idents))

