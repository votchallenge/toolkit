
import inspect
from typing import Type
from collections import Iterable

from vot import VOTException
from vot.utilities import to_number, to_string, to_logical

class AttributeException(VOTException):
    pass

def is_instance_or_subclass(val, class_) -> bool:
    """Return True if ``val`` is either a subclass or instance of ``class_``."""
    try:
        return issubclass(val, class_)
    except TypeError:
        return isinstance(val, class_)

def _get_fields(attrs, field_class, pop=False, ordered=False):
    """Get fields from a class. If ordered=True, fields will sorted by creation index.

    :param attrs: Mapping of class attributes
    :param type field_class: Base field class
    :param bool pop: Remove matching fields
    """
    fields = [
        (field_name, field_value)
        for field_name, field_value in attrs.items()
        if is_instance_or_subclass(field_value, field_class)
    ]
    if pop:
        for field_name, _ in fields:
            del attrs[field_name]
    if ordered:
        fields.sort(key=lambda pair: pair[1]._creation_index)
    return fields

# This function allows Schemas to inherit from non-Schema classes and ensures
#   inheritance according to the MRO
def _get_fields_by_mro(klass, field_class, ordered=False):
    """Collect fields from a class, following its method resolution order. The
    class itself is excluded from the search; only its parents are checked. Get
    fields from ``_declared_fields`` if available, else use ``__dict__``.

    :param type klass: Class whose fields to retrieve
    :param type field_class: Base field class
    """
    mro = inspect.getmro(klass)
    # Loop over mro in reverse to maintain correct order of fields
    return sum(
        (
            _get_fields(
                getattr(base, "_declared_fields", base.__dict__),
                field_class,
                ordered=ordered,
            )
            for base in mro[:0:-1]
        ),
        [],
    )

class Attribute(object):

    def __init__(self, default=None):
        self._default = None if default is None else self.coerce(default)

    def coerce(self, value):
        return value

    def dump(self, value):
        return value

    @property
    def default(self):
        return self._default

class AttributeeMeta(type):

    def __new__(mcs, name, bases, attrs):

        cls_attributes = _get_fields(attrs, Attribute, pop=True)
        klass = super().__new__(mcs, name, bases, attrs)
        inherited_attributes = _get_fields_by_mro(klass, Attribute)

        #cls_attributes += list(klass.opts.include.items())

        # Assign attributes on class
        klass._declared_attributes = dict(inherited_attributes + cls_attributes)

        return klass

class Nested(Attribute):

    def __init__(self, acls: Type["Attributee"]):
        super().__init__(None)

        if not issubclass(acls, Attributee):
            raise AttributeException("Illegal base class {}".format(acls))

        self._acls = acls

    def coerce(self, value):
        return self._acls(**value)

    def dump(self, value: "Attributee"):
        return value.dump()

    @property
    def default(self):
        return self._acls()

class Include(Nested):

    def filter(self, **kwargs):
        attributes = getattr(self._acls, "_declared_attributes", {})
        filtered = dict()
        for aname, afield in attributes.items():
            if isinstance(afield, Include):
                filtered.update(afield.filter(**kwargs))
            elif aname in kwargs:
                filtered[aname] = kwargs[aname]
        return filtered

    @property
    def default(self):
        return None

class Attributee(metaclass=AttributeeMeta):

    def __init__(self, **kwargs):
        attributes = getattr(self.__class__, "_declared_attributes", {})

        unconsumed = set(kwargs.keys())
        unspecified = set(attributes.keys())

        for aname, afield in attributes.items():
            if isinstance(afield, Include):
                iargs = afield.filter(**kwargs)
                super().__setattr__(aname, afield.coerce(iargs))
                unconsumed.difference_update(iargs.keys())
                unspecified.difference_update(iargs.keys())
            else:
                if not aname in kwargs:
                    if not afield.default is None:
                        avalue = afield.default
                    else:
                        continue
                else:
                    avalue = kwargs[aname]
                super().__setattr__(aname, afield.coerce(avalue))
            unconsumed.difference_update([aname])
            unspecified.difference_update([aname])

        if unspecified:
            raise AttributeError("Missing arguments: {}".format(", ".join(unspecified)))

        if unconsumed:
            raise AttributeError("Unsupported arguments: {}".format(", ".join(unconsumed)))

    def __setattr__(self, key, value):
        attributes = getattr(self.__class__, "_declared_attributes", {})
        if key in attributes:
            raise AttributeException("Attribute {} is readonly".format(key))
        super().__setattr__(key, value)

    def dump(self):
        attributes = getattr(self.__class__, "_declared_attributes", {})
        if attributes is None:
            return dict()
    
        serialized = dict()
        for aname, afield in attributes.items():
            if isinstance(afield, Include):
                serialized.update(afield.dump(getattr(self, aname, {})))
            else:
                serialized[aname] = afield.dump(getattr(self, aname, afield.default))
                
        return serialized

class Number(Attribute):

    def __init__(self, conversion, default, val_min=None, val_max=None):
        self._conversion = conversion
        self._val_min = val_min
        self._val_max = val_max
        super().__init__(default)

    def coerce(self, value):
        return to_number(value, max_n=self._val_max, min_n=self._val_min, conversion=self._conversion)

class Integer(Number):

    def __init__(self, val_min=None, val_max=None, default=None):
        super().__init__(int, default, val_max=val_max, val_min=val_min)

class Float(Number):

    def __init__(self, val_min=None, val_max=None, default=None):
        super().__init__(float, default, val_max=val_max, val_min=val_min)

class Boolean(Attribute):

    def __init__(self, default):
        super().__init__(default)

    def coerce(self, value):
        return to_logical(value)

class String(Attribute):

    def __init__(self, default=""):
        super().__init__(default)

    def coerce(self, value):
        return to_string(value)

class List(Attribute):

    def __init__(self, contains, default=None, separator=","):
        super().__init__(default)
        self._separator = separator
        self._contains = contains

    def coerce(self, value):
        if isinstance(value, str):
            value = value.split(self._separator)
        if not isinstance(value, Iterable):
            raise AttributeException("Unable to value convert to list")
        return [self._contains.coerce(x) for x in value]

    def __iter__(self):
        # This is only here to avoid pylint errors for the actual attribute field
        raise NotImplementedError