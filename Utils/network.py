def get_all_classes(iterable, element_class):
    return filter(lambda x: x.__class__ == element_class, iterable)


def first_class(iterable, element, default=None):
    elements = get_all_classes(iterable, element)
    try:
        return next(elements)
    except:
        return default