'''
https://gist.github.com/yosakabe/decad61ffe766fe427804e843c9fd746
'''

from os.path import splitext
# import yaml
import json
from collections import OrderedDict

def load_params(param_file=None, verbose=False):
    '''
    To load hyper-parameters in param_file; JSON and YAML format can be accepted. 
    '''
    
    # Provide default values if necessary.
    parameters = {}
    # Example:
    # parameters = {key1: 'value1',
    #               key2: 123,
    #               key3: False}

    if param_file is not None:
        # check file-extentions (JSON or YAML)
        file_type = check_filetype(param_file)
        if verbose:
            print('> Using hyper-parameters')
            print('> The file type of', param_file, 'is checked as', file_type)

        # if file_type == ".yaml":
        #     hyper_p = load_yaml(yamlfile=param_file, isshow=verbose)
        # elif file_type == ".json":
        #     hyper_p = load_json(jsonfile=param_file, isshow=verbose)

        hyper_p = load_json(jsonfile=param_file, isshow=verbose)

        # overwrite parameters
        parameters.update(hyper_p)
        return parameters
    
    else:
        if verbose:
            print('> No additional parameters. Use defaults.')
            print(parameters)
        return parameters


def check_filetype(param_file):
    '''
    To check file extension
    '''
    root, ext = splitext(param_file)
    return ext

# def load_yaml(yamlfile='sample_params.yaml', isshow=False):
#     '''
#     To load param_file given in YAML format.
#     '''
    
#     def construct_odict(loader, node):
#         return OrderedDict(loader.construct_pairs(node))
#     yaml.add_constructor('tag:yaml.org,2002:map', construct_odict)

#     with open(yamlfile) as file:
#         params = yaml.full_load(file)
#         if isshow:
#             print('>> The default hyper-parameters:')
#             print(yaml.dump(params, default_flow_style=False))
#             print('>> The rest of parameters are set as default')
#         return params

def load_json(jsonfile='sample_params.json', isshow=False):
    '''
    To load param_file given in JSON format.
    '''
    params = json.loads(open(jsonfile).read(), object_pairs_hook=OrderedDict)
    if isshow:
        print('>> The default hyper-parameters:')
        for key, value in params.items():
            print('{:25s} - {:12}'.format(key, str(value)))
        print('>> The rest of parameters are set as default')
    return params