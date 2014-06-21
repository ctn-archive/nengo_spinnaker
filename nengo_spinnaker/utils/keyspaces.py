"""Tools for managing various key spaces.
"""

from nengo.utils.compat import with_metaclass


def _make_mask_getter(mask):
    def get_mask(self):
        return mask
    return get_mask


def _make_field_getter(fname):
    fname = '_%s' % fname

    def get_field(self):
        return getattr(self, fname)

    return get_field


class MetaKeySpace(type):
    """Metaclass for creating keyspace classes.
    """
    def __new__(cls, clsname, bases, dct):
        new_dct = dict([(k, v) for (k, v) in dct.items() if k != "fields" and
                        k != "routing_fields"])
        new_dct['__fields__'] = [f[0] for f in dct['fields']]
        new_dct['__field_lengths__'] = dict(dct['fields'])
        new_dct['__routing_fields__'] = dct['routing_fields']

        # First ensure that there aren't more than 32 bits assigned
        if sum([f[1] for f in dct['fields']]) > 32:
            raise ValueError("Assigned more than 32-bits to keyspace.")

        # Create properties for each field mask and the routing mask
        b = 32
        r_mask = 0x0
        for (name, bits) in dct['fields']:
            mask = sum([1 << n for n in range(bits)]) << (b - bits)
            b -= bits

            if name in dct['routing_fields']:
                r_mask |= mask

            new_dct['mask_%s' % name] = property(_make_mask_getter(mask))
        new_dct['routing_mask'] = property(_make_mask_getter(r_mask))

        # Create properties for each field
        for (n, _) in dct['fields']:
            new_dct[n] = property(_make_field_getter(n))
            new_dct['_%s' % n] = None

        return super(MetaKeySpace, cls).__new__(cls, clsname, bases, new_dct)


class KeySpace(with_metaclass(MetaKeySpace)):
    fields = []
    routing_fields = []

    def _make_key(self, fields, field_values):
        key = 0x0
        b = 32
        for f in fields:
            bits = self.__field_lengths__[f]
            v = field_values.get(f, None)

            if v is not None:
                key |= v << (b - bits)

            b -= bits
        return key

    def key(self, **field_values):
        return self._make_key(self.__fields__, field_values)

    def routing_key(self, **field_values):
        return self._make_key(self.__routing_fields__, field_values)

    def __eq__(self, ks2):
        if not self.__field_lengths__ == ks2.__field_lengths__:
            return False
        if not self.__routing_fields__ == ks2.__routing_fields__:
            return False

        return True


def create_keyspace(name, new_fields, new_routing_fields):
    return type(name, (KeySpace, ), {'fields': new_fields,
                                     'routing_fields': new_routing_fields})

nengo_default = create_keyspace(
    'NengoDefaultKeySpace',
    [('x', 8), ('y', 8), ('p', 5), ('i', 5), ('d', 6)],
    "xypi"
)
