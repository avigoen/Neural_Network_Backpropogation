def get_all_classes(iterable, element_class):
    return filter(lambda x: x.__class__ == element_class, iterable)


def first_class(iterable, element, default=None):
    elements = get_all_classes(iterable, element)
    try:
        return next(elements)
    except:
        return default


def get_nearest_dense_layer(iterable, element, dense_class, default=None):
    layers_till_element = []
    for i in iterable:
        if i == element:
            break
        layers_till_element.append(i)

    return first_class(layers_till_element[::-1], dense_class, default)
