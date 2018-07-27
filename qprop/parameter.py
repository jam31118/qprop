from sys import stderr
import re
from re import match
from os import listdir
from os.path import isdir, isfile, join, basename
from numbers import Number

from .core import Qprop20
from .default import default_config, type2castFunction
from .util import is_iterable


class Param(object):
    def __init__(self, name, param_file_name, value=None, type_name=None):
        for arg in [name, value, param_file_name, type_name]:
            if arg is not None: assert type(arg) is str
        self.name, self.value, self.file_name, self.type = name, value, param_file_name, type_name
        if self.value is not None:
            self.value = type2castFunction[self.type](self.value)
        
    @classmethod
    def from_param_entry(cls, entry_string, param_file_name):
        for arg in [entry_string, param_file_name]: assert type(arg) is str
        fields = entry_string.split()
        if len(fields) != default_config['param_file_field_per_entry']:
            err_mesg = "The given entry ({}) seems to be imcompatible with `default_config['param_file_field_per_entry']` ({})"
            raise Exception(err_mesg.format(entry_string, default_config['param_file_field_per_entry']))
        name, type_name, value = fields
        param_obj = cls(name=name, value=value, param_file_name=param_file_name, type_name=type_name)
        return param_obj
        
    def __eq__(self, param_obj):
        both_are_same = True
        both_are_same &= self.name == param_obj.name
        both_are_same &= self.value == param_obj.value
        both_are_same &= self.file_name == param_obj.file_name
        if (param_obj.type is not None) and (self.type is not None):
            both_are_same &= param_obj.type == self.type
        
        return both_are_same
    
    def __repr__(self):
        return "<Param: {0} == {2} ({1}) @ {3}>".format(self.name, self.type, self.value, self.file_name)
    
    def is_same_param(self, param):
        assert isinstance(param, type(self))
        assert (self.name is not None) and (param.name is not None)
        is_same = True
        is_same &= self.name == param.name
        if (self.file_name is not None) and (param.file_name is not None):
            is_same &= self.file_name == param.file_name
        if (self.type is not None) and (param.type is not None):
            is_same &= self.type == param.type
        return is_same


class Param_File(object):
    def __init__(self, param_objects, param_file_path):
        self.param_objects = param_objects
        ## NOTE, since there may be multiple parameter with same name, .. the parameter name shouldn't be gathered in a list without specifying each parameter filename
#        self.param_names = self.get_param_names_from_param_objects(self.param_objects)
        self.file_path = param_file_path
        
    @classmethod
    def from_param_file(cls, param_file_path):
        assert isfile(param_file_path)
        file_name = basename(param_file_path)
        param_objects = []
        with open(param_file_path, mode="r") as f:
            for line in f:
                if cls.is_proper_param_entry(line):
                    param_objects.append(Param.from_param_entry(line, file_name))
        return cls(param_objects, param_file_path)
    
#    @staticmethod
#    def get_param_names_from_param_objects(param_objects):
#        assert is_iterable(param_objects)
#        names = []
#        for param_object in param_objects:
#            assert isinstance(param_object, Param) and hasattr(param_object, "name")
#            names.append(param_object.name)
#        return names

    @staticmethod
    def is_proper_param_entry(entry_string):
        assert type(entry_string) is str
        blank_line = str.lstrip(entry_string).rstrip() == ''
        starts_from_comment_sign = False
        if not blank_line:
            starts_from_comment_sign = str.lstrip(entry_string)[0] == default_config['param_file_comment_character']
        is_proper_entry = (not starts_from_comment_sign) and (not blank_line)
        return is_proper_entry
    
    def if_exist_get_value(self, param, default_value=None):
        matched_param_obj = [param_obj for param_obj in self.param_objects 
                             if param_obj.is_same_param(param)]
        assert len(matched_param_obj) <= 1
        if len(matched_param_obj) == 0:
            if (basename(self.file_path) == param.file_name) and (default_value is not None):
                return default_value
        elif len(matched_param_obj) == 1:
            return matched_param_obj[0].value
    
    @staticmethod
    def get_value_from_param_objects(param_objects, param_name, param_file_name):
        match_params = [param for param in param_objects if param.name == param_name]
        num_of_matched_param = len(match_params)
        assert type(param_file_name) is str

        ## Find the unique mathced param object if possible
        match_param = None
        if num_of_matched_param == 1:
            match_param, = match_params
        elif num_of_matched_param > 1:
            if param_file_name == '':
                err_mesg = "The parameter file name should be specified " \
                        + "since the parameter with name '{}' seems to exist in multiple parameter files"
                raise Exception(err_mesg.format(param_name))
            param_with_matched_file_name = [param for param in match_params 
                    if param.file_name == basename(param_file_name)]
            if len(param_with_matched_file_name) == 0:
                err_mesg = "The parameter with name '{}' in file name '{}', " \
                        + "doesn't seem to exist."
                raise ValueError(err_mesg.format(param_name, param_file_name))
            elif len(param_with_matched_file_name) > 1:
                err_mesg = "The parameter could not be identified uniquely " \
                        + "for param_name: '{}' and param_file_name: '{}'. " \
                        + "The `param_with_matched_file_name`: '{}'"
                raise ValueError(err_mesg.format(param_name, param_file_name, param_with_matched_file_name))
            match_param, = param_with_matched_file_name
        elif num_of_matched_param == 0:
            err_mesg = "No matched parameter object for given param_name: {}"
            raise ValueError(err_mesg.format(param_name))
        else:
            err_mesg = "Unexpected number of `matched_param`: {} / The `match_params: {}`"
            raise Exception(err_mesg.format(num_of_matched_param, match_params))

        assert match_param is not None
        return match_param.value

    def __getitem__(self, query_string):
        param_name, param_file_name = self.parse_param_query_string(query_string)
        if (param_file_name is not '') and (param_file_name != basename(self.file_path)):
            err_mesg = "The corresponding parameter file name '{}' in the given query_string '{}' " \
                    + "doesn't match with this parameter file name '{}'"
            raise ValueError(err_mesg.format(param_file_name, query_string, basename(self.file_path)))
        return self.get_value_from_param_objects(self.param_objects, param_name, param_file_name)
    
    def __repr__(self):
        return "<Param_File @ {}>".format(self.file_path)

    def __contains__(self, item):
        try: self[item]
        except: return False
        return True

    @staticmethod
    def parse_param_query_string(query_string):
        assert type(query_string) is str
        sep = ':'
        splited = query_string.split(sep)
        param_name, param_file_name = None, None
        if len(splited) == 1:
            ## no colon in query_string, thus it is purely param_name
            param_name = query_string
            param_file_name = ''
        elif len(splited) == 2:
            param_name, param_file_name = splited
        else:
            err_mesg = "Could not parse `query_string` ({}) into param_name (possibly followed by param_file_name)"
            raise ValueError(err_mesg.format(query_string))
        assert (param_name is not None) and (param_file_name is not None)
        return param_name, param_file_name


class Param_File_List(object):
    def __init__(self, param_file_list):
        self.param_file_objects = param_file_list
    
    @classmethod
    def from_dir(cls, dir_path):
        assert isdir(dir_path)
        param_file_paths = cls.get_param_file_path_list(dir_path)
        param_file_objects = [Param_File.from_param_file(param_file_path) for param_file_path in param_file_paths]
        return cls(param_file_objects)
    
    @staticmethod
    def get_param_file_path_list(dir_path):
        assert isdir(dir_path)
        dir_content_paths = [join(dir_path, content) for content in listdir(dir_path) 
                             if match(r".*\.param", content) is not None]
        return dir_content_paths
    
    def if_exist_get_value(self, param, default_value=None):
        matched_param_values = []
        for param_file_obj in self.param_file_objects:
            val = param_file_obj.if_exist_get_value(param, default_value=default_value)
            if val is not None:
                matched_param_values.append(val)
        assert len(matched_param_values) <= 1
        if len(matched_param_values) == 1:
            return matched_param_values[0]
        
    def __repr__(self):
        return self.param_file_objects.__repr__()
    
    def __getitem__(self, index):
        return self.param_file_objects[index]

    def get_value(self, query_string):
        param_name, param_file_name = Param_File.parse_param_query_string(query_string)
        param_objects = []
        for param_file in self.param_file_objects: param_objects += param_file.param_objects
        return Param_File.get_value_from_param_objects(param_objects, param_name, param_file_name)

    def __contains__(self, query_string):
        try: self.get_value(query_string)
        except: return False
        return True



def given_calc_dir_satisfies_param_conditions(calc_dir_path, param_conditions, verbose=False):
    eligible = True
    
    param_file_list = Param_File_List.from_dir(calc_dir_path)
    for param_query_string in param_conditions.keys():
        value_in_file = None
        try: value_in_file = param_file_list.get_value(param_query_string)
        except: eligible &= False
        
        param_condition = param_conditions[param_query_string]
        param_condition_callable = None
        if isinstance(param_condition, Number):
            param_condition_callable = lambda x: x == param_condition
        elif callable(param_condition):
            param_condition_callable = param_condition
        
        if param_condition_callable(value_in_file) is False:
            if verbose:
                log_mesg = "The given directory '{}' doesn't satisfy param_query_string: '{}' " \
                + "where param_condition is '{}' and the parameter value from file is '{}'"
                print(log_mesg.format(calc_dir_path, param_query_string, param_condition, value_in_file))
            eligible &= False
    
    return eligible



def filter_calc_dir(dir_list, param, criteria, verbose=False, default_value=None):
    #dir_list = Qprop20.get_list_of_calc_homes(dir_list, verbose=verbose)
    criteria_callable = criteria
    if not callable(criteria):
        assert type(criteria) in [float, str, int]
        criteria_callable = lambda x: x == criteria
    for dirpath in dir_list: assert isdir(dirpath)
    assert callable(criteria_callable)
    filtered_dir_list = []
    for dirpath in dir_list:
        param_file_list = Param_File_List.from_dir(dirpath)
        val = param_file_list.if_exist_get_value(param, default_value=default_value)
        if val is not None:
            if criteria_callable(val):
                 filtered_dir_list.append(dirpath)
    return filtered_dir_list



# [TODO] If there's no matching parameter, add a line.
# .. I think it is better to make separate method to do this.

def update_param_in_file(filepath, param_name, param_type, new_value):
    for arg in [filepath, param_name, param_type]:
        assert type(arg) is str
    assert isfile(filepath)
    try: new_value_str = str(new_value)
    except: raise TypeError(
            "Failed to convert 'new_value' {0} into string".format(new_value))

    file_content_original, file_content_updated = None, None
    with open(filepath, "r") as f: file_content_original = f.read()
    param_exist = param_exists(file_content_original, param_name, param_type)
    if param_exist:
        pattern = ' '.join([param_name, param_type, '.*'])
        new_string = ' '.join([param_name, param_type, new_value_str])
        file_content_updated = re.sub(pattern, new_string, file_content_original)
    else: 
        err_mesg = "Couldn't find parameter with name {0} of type {1} in file {2}"
        raise Exception(err_mesg.format(param_name, param_type, filepath))
    
    with open(filepath, "w") as f: f.write(file_content_updated)

def param_exists(content_str, param_name, param_type):
    for arg in [content_str, param_name, param_type]:
        assert type(arg) is str
    param_exist_in_given_content = False
    content_lines = content_str.split('\n')
    for line in content_lines:
        if (param_name in line) and (param_type in line):
            words = line.split(' ')
            if (param_name in words) and (param_type in words):
                param_exist_in_given_content = True
    return param_exist_in_given_content                
