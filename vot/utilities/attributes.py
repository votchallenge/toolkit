
import inspect
from typing import Type
from collections import Iterable, Mapping

from vot import VOTException
from vot.utilities import to_number, to_string, to_logical, singleton, import_class, class_fullname, class_string

class AttributeException(VOTException):
    pass

class AttributeParseException(AttributeException):
    def __init__(self, cause, key):
        self._keys = []
        if isinstance(cause, AttributeParseException):
            self._keys.extend(cause._keys)
            cause = cause.__cause__ or cause.__context__
        super().__init__(cause)
        self._keys.append(key)
 
    def __str__(self):
        return "Attribute error: {}".format(".".join(self._keys))


@singleton
class Undefined():
    pass

def is_undefined(a):
    if a is None:
        return False
    return a == Undefined()

def is_instance_or_subclass(val, class_) -> bool:
    """Return True if ``val`` is either a subclass or instance of ``class_``."""
    try:
        return issubclass(val, class_)
    except TypeError:
        return isinstance(val, class_)

class ReadonlyMapping(Mapping):

    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

class Attribute(object):

    def __init__(self, default=Undefined()):
        self._default = default if is_undefined(default) else self.coerce(default, {})

    def coerce(self, value, _):
        return value

    def dump(self, value):
        return value

    @property
    def default(self):
        return self._default

    @property
    def required(self):
        return is_undefined(self._default)

class Nested(Attribute):

    def __init__(self, acls: Type["Attributee"], override: Mapping = None, **kwargs):
        if not issubclass(acls, Attributee):
            raise AttributeException("Illegal base class {}".format(acls))

        self._acls = acls
        self._override = dict(override.items() if not override is None else [])
        if "default" not in kwargs:
            self._required = False

            for _, afield in getattr(acls, "_declared_attributes", {}).items():
                if afield.required:
                    self._required = True
            if not self._required:
                kwargs["default"] = {}
        else:
            self._required = False

        super().__init__(**kwargs)

    def coerce(self, value, _):
        if value is None:
            return None
        assert isinstance(value, Mapping)
        kwargs = dict(value.items())
        kwargs.update(self._override)
        return self._acls(**kwargs)

    def dump(self, value: "Attributee"):
        if value is None:
            return None
        return value.dump()

    @property
    def required(self):
        return super().required and self._required

    def __getattr__(self, name):
        # This is only here to avoid pylint errors for the actual attribute field
        return super().__getattr__(name)

    def __setattr__(self, name, value):
        # This is only here to avoid pylint errors for the actual attribute field
        super().__setattr__(name, value)

class AttributeeMeta(type):

    @staticmethod
    def _get_fields(attrs: dict, pop=False):
        """Get fields from a class.
        :param attrs: Mapping of class attributes
        """
        fields = []
        for field_name, field_value in attrs.items():
            if is_instance_or_subclass(field_value, Attribute):
                fields.append((field_name, field_value))
        if pop:
            for field_name, _ in fields:
                del attrs[field_name]

        return fields

    # This function allows Schemas to inherit from non-Schema classes and ensures
    #   inheritance according to the MRO
    @staticmethod
    def _get_fields_by_mro(klass):
        """Collect fields from a class, following its method resolution order. The
        class itself is excluded from the search; only its parents are checked. Get
        fields from ``_declared_attributes`` if available, else use ``__dict__``.

        :param type klass: Class whose fields to retrieve
        """
        mro = inspect.getmro(klass)
        # Loop over mro in reverse to maintain correct order of fields
        return sum(
            (
                AttributeeMeta._get_fields(
                    getattr(base, "_declared_attributes", base.__dict__)
                )
                for base in mro[:0:-1]
            ),
            [],
        )


    def __new__(mcs, name, bases, attrs):

        cls_attributes = AttributeeMeta._get_fields(attrs, pop=True)
        klass = super().__new__(mcs, name, bases, attrs)
        inherited_attributes = AttributeeMeta._get_fields_by_mro(klass)

        # Assign attributes on class
        klass._declared_attributes = dict(inherited_attributes + cls_attributes)

        return klass

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

class Attributee(metaclass=AttributeeMeta):

    def __init__(self, **kwargs):
        attributes = getattr(self.__class__, "_declared_attributes", {})

        unconsumed = set(kwargs.keys())
        unspecified = set(attributes.keys())

        for aname, afield in attributes.items():
            if isinstance(afield, Include):
                iargs = afield.filter(**kwargs)
                super().__setattr__(aname, afield.coerce(iargs, {"parent": self}))
                unconsumed.difference_update(iargs.keys())
                unspecified.difference_update(iargs.keys())
            else:
                if not aname in kwargs:
                    if not afield.required:
                        avalue = afield.default
                        super().__setattr__(aname, avalue)
                    else:
                        continue
                else:
                    avalue = kwargs[aname]
                    try:
                        super().__setattr__(aname, afield.coerce(avalue, {"parent": self}))
                    except AttributeException as ae:
                        raise AttributeParseException(ae, aname)
                    except AttributeError as ae:
                        raise AttributeParseException(ae, aname)
            unconsumed.difference_update([aname])
            unspecified.difference_update([aname])

        if unspecified:
            raise AttributeException("Missing arguments: {}".format(", ".join(unspecified)))

        if unconsumed:
            raise AttributeException("Unsupported arguments: {}".format(", ".join(unconsumed)))

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

    def __init__(self, conversion, val_min=None, val_max=None, **kwargs):
        self._conversion = conversion
        self._val_min = val_min
        self._val_max = val_max
        super().__init__(**kwargs)

    def coerce(self, value, _=None):
        return to_number(value, max_n=self._val_max, min_n=self._val_min, conversion=self._conversion)

class Integer(Number):

    def __init__(self, **kwargs):
        super().__init__(conversion=int, **kwargs)

class Float(Number):

    def __init__(self, **kwargs):
        super().__init__(conversion=float, **kwargs)

class Boolean(Attribute):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def coerce(self, value, _):
        return to_logical(value)

class String(Attribute):

    def __init__(self, transformer=None, **kwargs):
        self._transformer = transformer
        super().__init__(**kwargs)

    def coerce(self, value, ctx):
        if value is None:
            return None
        if self._transformer is None:
            return to_string(value)
        else:
            return self._transformer(to_string(value), ctx)

class List(Attribute):

    def __init__(self, contains, separator=",", **kwargs):
        assert isinstance(contains, Attribute)
        self._separator = separator
        self._contains = contains
        super().__init__(**kwargs)

    def coerce(self, value, context=None):
        if isinstance(value, str):
            value = [v.strip() for v in value.split(self._separator)]
        if isinstance(value, dict):
            value = value.values()
        if not isinstance(value, Iterable):
            raise AttributeException("Unable to convert value to list")
        if context is None:
            context = dict()
        else:
            context = dict(**context)
        return [self._contains.coerce(x, context) for i, x in enumerate(value)]

    def __iter__(self):
        # This is only here to avoid pylint errors for the actual attribute field
        raise NotImplementedError

    def __getitem__(self, key):
        # This is only here to avoid pylint errors for the actual attribute field
        raise NotImplementedError

    def __setitem__(self, key, value):
        # This is only here to avoid pylint errors for the actual attribute field
        raise NotImplementedError

    def dump(self, value):
        return [self._contains.dump(x) for x in value]

class Map(Attribute):

    def __init__(self, contains, container=dict, **kwargs):
        assert isinstance(contains, Attribute)
        self._contains = contains
        self._container = container
        super().__init__(**kwargs)

    def coerce(self, value, context=None):
        if not isinstance(value, Mapping):
            raise AttributeException("Unable to value convert to dict")
        container = self._container()
        if context is None:
            context = dict()
        else:
            context = dict(**context)
        for name, data in value.items():
            context["key"] = name
            container[name] = self._contains.coerce(data, context)
        return ReadonlyMapping(container)

    def __iter__(self):
        # This is only here to avoid pylint errors for the actual attribute field
        raise NotImplementedError

    def __getitem__(self, key):
        # This is only here to avoid pylint errors for the actual attribute field
        raise NotImplementedError

    def __setitem__(self, key, value):
        # This is only here to avoid pylint errors for the actual attribute field
        raise NotImplementedError

    def dump(self, value):
        return {k: self._contains.dump(v) for k, v in value.items()}


def default_object_resolver(typename: str, _, **kwargs) -> Attributee:
    """Default object resovler

    Arguments:
        typename {str} -- String representation of a class that can be imported.
            Should be a subclass of Attributee as it is constructed from kwargs.

    Returns:
        Attributee -- An instance of the class
    """
    clstype = import_class(typename)
    assert issubclass(clstype, Attributee)
    return clstype(**kwargs)

class Object(Attribute):

    def __init__(self, resolver=default_object_resolver, subclass=None, **kwargs):
        super().__init__(**kwargs)
        assert subclass is None or inspect.isclass(subclass)
        self._resolver = resolver
        self._subclass = subclass

    def coerce(self, value, context=None):
        assert isinstance(value, dict)
        class_name = value.get("type", None)
        obj = self._resolver(class_name, context, **{k: v for k, v in value.items() if not k == "type"})
        if not self._subclass is None:
            if not isinstance(obj, self._subclass):
                raise AttributeException("Object is not a subclass of {}".format(self._subclass))
        return obj

    def dump(self, value):
        data = value.dump()
        data["type"] = class_fullname(value)
        return data

    def __getattr__(self, name):
        # This is only here to avoid pylint errors for the actual attribute field
        return super().__getattr__(name)

    def __setattr__(self, name, value):
        # This is only here to avoid pylint errors for the actual attribute field
        super().__setattr__(name, value)

class Callable(Attribute):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def coerce(self, value, context=None):
        if callable(value):
            return value

        assert isinstance(value, str)
        caltype = import_class(value)
        assert callable(caltype)
        caltype.resname = value
        return caltype

    def dump(self, value):
        if hasattr(value, "resname"):
            return value.resname
        if inspect.isclass(value):
            return class_string(value)
        return class_fullname(value)

    def __call__(self):
        # This is only here to avoid pylint errors for the actual attribute field
        raise NotImplementedError
