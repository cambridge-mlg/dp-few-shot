import torch
from timm.models.layers.norm_act import BatchNormAct2d


def get_full_parameter_names(model, short_parameter_names):
    parameter_list = []
    for name, _ in model.named_parameters():
        for parameter_name in short_parameter_names:
            if parameter_name in name:
                parameter_list.append(name)
    return parameter_list


def get_film_parameter_names(feature_extractor_name, feature_extractor):
    if 'efficientnet' in feature_extractor_name:
        parameter_list = []
        for name, module in feature_extractor.named_modules():
            if isinstance(module, BatchNormAct2d):
                if module.film:
                    parameter_list.append(name + '.weight')
                    parameter_list.append(name + '.bias')
        return parameter_list
    elif 'vit' in feature_extractor_name:
        return get_full_parameter_names(feature_extractor, ['norm', 'norm1', 'norm2'])
    elif 'BiT' in feature_extractor_name:
        return get_full_parameter_names(feature_extractor, ['gn3', 'gn.'])


def enable_film(film_parameter_names, feature_extractor):
    for name, param in feature_extractor.named_parameters():
        if name in film_parameter_names:
            param.requires_grad = True


def get_film_parameters(film_parameter_names, feature_extractor):
    film_params = []
    for name, param in feature_extractor.named_parameters():
        if name in film_parameter_names:
            film_params.append(param.detach().clone())
    return film_params


def get_film_parameter_sizes(film_parameter_names, feature_extractor):
    film_params_sizes = []
    for name, param in feature_extractor.named_parameters():
        if name in film_parameter_names:
            film_params_sizes.append(len(param))
    return film_params_sizes


def set_film_parameters(film_parameter_names, film_parameters, feature_extractor):
    assert len(film_parameter_names) == len(film_parameters)
    with torch.no_grad():
        for name, param in feature_extractor.named_parameters():
            if name in film_parameter_names:
                param.copy_(film_parameters[film_parameter_names.index(name)])


def film_to_dict(film_parameter_names, film_parameters):
    assert len(film_parameter_names) == len(film_parameters)
    film_dict = {}
    for i in range(len(film_parameter_names)):
        film_dict[film_parameter_names[i]] = film_parameters[i]
    return film_dict


def test_film_param_stuff(feature_extractor_name, feature_extractor):
    state_before = feature_extractor.state_dict().__str__()

    film_param_names = get_film_parameter_names(feature_extractor_name, feature_extractor)

    film_params = get_film_parameters(film_param_names, feature_extractor)

    set_film_parameters(film_param_names, film_params,feature_extractor)

    state_after = feature_extractor.state_dict().__str__()

    assert state_before == state_after
