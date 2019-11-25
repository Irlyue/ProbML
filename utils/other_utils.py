def common_repr(obj, keys, **extras):
    properties = ',\n'.join('  {}={}'.format(key, getattr(obj, key)) for key in keys)
    extras = ',\n'.join('  {}={}'.format(k, v) for k, v in extras.items())
    return '{}(\n{},\n{})'.format(type(obj).__name__, properties, extras)
