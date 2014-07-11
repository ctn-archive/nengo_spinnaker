"""Tools for managing various key spaces.
"""

from nengo.utils.compat import with_metaclass


def _make_mask_getter(mask):
    def get_mask(self):
        return mask
    return get_mask

def _make_set_checker(field):
    def get_set(self):
        return field in self._field_values
    return get_set


class MetaKeySpace(type):
    """Metaclass for creating keyspace classes.
    """
    def __new__(cls, clsname, bases, dct):
        new_dct = dict([(k, v) for (k, v) in dct.items() if k != "fields" and
                        k != "routing_fields"])
        new_dct['__fields__'] = [f[0] for f in dct['fields']]
        new_dct['__fieldsc__'] = dct['fields']
        new_dct['__field_lengths__'] = dict(dct['fields'])
        new_dct['__routing_fields__'] = dct['routing_fields']
        new_dct['__filter_fields__'] = dct['filter_fields']

        # First ensure that there aren't more than 32 bits assigned
        if sum([f[1] for f in dct['fields']]) > 32:
            raise ValueError("Assigned more than 32-bits to keyspace.")

        # Create properties for each field mask and the routing mask
        b = 32
        r_mask = 0x0
        f_mask = 0x0
        for (name, bits) in dct['fields']:
            mask = sum([1 << n for n in range(bits)]) << (b - bits)
            b -= bits

            if name in dct['routing_fields']:
                r_mask |= mask
            if name in dct['filter_fields']:
                f_mask |= mask

            new_dct['mask_%s' % name] = property(_make_mask_getter(mask))
            new_dct['is_set_%s' % name] = property(_make_set_checker(name))
        new_dct['routing_mask'] = property(_make_mask_getter(r_mask))
        new_dct['filter_mask'] = property(_make_mask_getter(f_mask))

        return super(MetaKeySpace, cls).__new__(cls, clsname, bases, new_dct)


class KeySpace(with_metaclass(MetaKeySpace)):
    fields = []
    routing_fields = []
    filter_fields = []

    def __init__(self, **field_values):
        self._field_values = dict()

        # For each field in the given list of field values
        for (f, v) in field_values.items():
            # Assert that the value given is within range
            v_max = 2**self.__field_lengths__[f] - 1
            if v > v_max:
                raise ValueError("%d is larger than the maximum value for this"
                                 " field '%s' (%d)" % (v, f, v_max))

            # Then save the value for this field
            self._field_values[f] = v

    def __call__(self, **field_values):
        new_field_values = dict(self._field_values)
        new_field_values.update(field_values)
        return type(self)(**new_field_values)

    def _make_key(self, fields, field_values):
        key = 0x0
        b = 32
        for (f, bits) in self.__fieldsc__:

            # Get the value for this field -- try from the dict first, then
            # the local values.  That way the value in the KeySpace takes
            # precedence.  Raise an Exception if there's been an attempt to
            # override the value set in the keyspace.
            if f in self._field_values and f in field_values:
                raise AttributeError("Field '%s' has already been assigned for"
                                     " this keyspace" % f)
            v_ = field_values.get(f, None)
            v = self._field_values.get(f, v_)

            # Get the maximum value, assert value is in range
            v_max = 2**bits - 1
            if v is not None and v > v_max:
                raise ValueError("%d is larger than the maximum value for this"
                                 " field '%s' (%d)" % (v, f, v_max))

            # Add this field to the key
            if v is not None and f in fields:
                key |= v << (b - bits)

            b -= bits
        return key

    def key(self, **field_values):
        return self._make_key(self.__fields__, field_values)

    def routing_key(self, **field_values):
        return self._make_key(self.__routing_fields__, field_values)

    def filter_key(self, **field_values):
        return self._make_key(self.__filter_fields__, field_values)

    def __eq__(self, ks2):
        if not self.__field_lengths__ == ks2.__field_lengths__:
            return False
        if not self.__routing_fields__ == ks2.__routing_fields__:
            return False

        return True


def create_keyspace(name, new_fields, new_routing_fields, new_filter_fields):
    if new_filter_fields is None:
        new_filter_fields = new_routing_fields
    return type(name, (KeySpace, ), {'fields': new_fields,
                                     'routing_fields': new_routing_fields,
                                     'filter_fields': new_filter_fields})
