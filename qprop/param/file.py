"""Routines for manipulating parameter files"""

from re import sub
from os import listdir
from os.path import isfile, isdir, join, splitext, basename
from sys import stderr

import numpy as np

from ..default import type2castFunction

class Parameter(object):
    file_extension = ".param"

        
class ParameterFile(Parameter):
    """
    
    Note
    ----------
    The uniqueness of each parameter name is ensured.
    """
    
    def __init__(self, file_path, strict=True):
        
        if not self.seems_to_be_param_file:
            _mesg = ("The given file path ('{}') "
                     "doesn't seem to be a parameter file path.")
            if strict: raise ValueError(_mesg)
            else: print(_mesg, file=stderr)
        
        self.path = file_path
        self.name = basename(file_path)
        
        self._text = self.load(self.path)
        self._lines = self._text.splitlines()
        self._array = self.construct_array(self._lines)
    

    def _process_param_name_and_value(self, name, value):
        if not isinstance(name, str): 
            raise ValueError("The `name` should be of type '{}'".format(str))
        try: _value_str = str(value)
        except: raise Exception(
            "Error during converting given value (={}) to string.".format(value))
        return name, _value_str


    def __getitem__(self, name):
        """Get value of a parameter with given name
        
        Notes
        -----
        This routine utilizes the fact that the names of parameters per
        a parameter file is unique.
        """
        if name not in self._array['name']:
            raise ValueError(
                ("The given name ('{}') doesn't seem to exist "
                 "in the parameter file").format(name)
            )
        _indices, = np.where(self._array['name'] == name)
        assert len(_indices) == 1
        _index, = _indices
        _name, _type, _value_str = self._array[_index]
        assert name == _name
        _value = type2castFunction[_type](_value_str)
        return _value


    def add_param(self, name, value, type):

        assert isinstance(name, str)
        try: _value_str = str(value)
        except: raise ValueError("Failed to convert '{}' to string".format(value))
        assert isinstance(type, str)
        
        if name in self._array(['name']):
            _msg = "The parameter with name '{}' already exists."
            raise ValueError(_msg, name)

        ## Add parameter to the file text
        self._text += "{} {} {}".format(name, type, _value_str)

        ## Add parameter to the array
        self._array = self.construct_array(self._text.splitlines())


    def update_param(self, name, value):
        _name, _value_str = self._process_param_name_and_value(name, value)

        if _name not in self._array['name']:
            raise ValueError(
                ("The given name ('{}') doesn't seem to exist "
                 "in the parameter file").format(_name)
            )
            
        _indices, = np.where(self._array['name'] == _name)
        if _indices.size != 1: raise Exception("Duplicate parameter name")
        _index, = _indices
        _, _type, _old_value = self._array[_index]
        
        #### Update text
        _pattern = r"({}\s+{}\s+)(.+)".format(_name, _type)
        self._text = sub(_pattern, r"\1{}", self._text).format(value)
        
        #### Update array
        self._array[_index]['value'] = value
        
    
    def write(self, file_path):
        try: 
            with open(file_path, "w") as f: f.write(self._text)
        except OSError as e: 
            raise Exception("Failed to write to file with given path: {}".format(path))
    
    def write_under_dir(self, dir_path):
        """Write a file under the directory given by 'dir_path'"""
        if not isdir(dir_path): raise ValueError("The 'dir_path' should be a directory")
        _file_path = join(dir_path, self.name)
        self.write(_file_path)
    
    
    @staticmethod
    def construct_array(lines):
        _dtype = [('name','<U256'),('type','<U256'),('value','<U256')]
        try:
            _array = np.array([tuple(line.split()) for line in lines 
                               if not line.startswith("#") and line != ''], dtype=_dtype)
        except: raise Exception("Error during constructing parameter array")
        if _array.ndim != 1: raise Exception("The array is expected to be 1 dimension.")
        if np.unique(_array['name']).size != _array['name'].size:
            raise Exception("There seems to be one or more duplicate in parameter names.")
        return _array
        
        
    @staticmethod
    def load(file_path):
        _text = None
        try:
            with open(file_path, "r") as f:
                _text = f.read()
        except OSError as e:
            raise e("Error during opening a file: {:s}".format(file_path))
        if _text is None:
            raise Exception("Unexpected internal error")
        return _text
        
        
    @staticmethod
    def seems_to_be_param_file(file_path):
        _is_file = isfile(file_path)
        _has_param_extension = splitext(file_path)[-1] == self.file_extension
        _seems_to_be_param_file = _is_file & _has_param_extension
        
        
        
class ParameterFileSet(dict, Parameter):
    
    def __init__(self, param_files):
        for _file in param_files: assert isinstance(_file, ParameterFile)
        self.file_names = [param_file.name for param_file in param_files]
        self.file_paths = [param_file.path for param_file in param_files]
        if len(set(self.file_names)) != len(self.file_names):
            raise Exception("There may be one or more duplicate file name "
                            "in given list: {}".format(self.file_names))
        super().__init__(zip(self.file_names, param_files))
        
    @classmethod
    def from_calc_dir(cls, calc_dir_path):
        if not isdir(calc_dir_path): 
            raise ValueError("The 'calc_dir_path' should be a path to a directory.\n"
                             "Given: '{}'".format(calc_dir_path))
        _file_names = [file_name for file_name in listdir(calc_dir_path) 
                             if splitext(file_name)[-1] == cls.file_extension]
        _file_paths = [join(calc_dir_path, file_name) for file_name in _file_names]
        return cls.from_file_paths(_file_paths)
        
        
    @classmethod
    def from_file_paths(cls, file_paths):
        try: _param_files = [ParameterFile(path) for path in file_paths]
        except: raise RuntimeError("Failed to construct parameter file objects")
        return cls(_param_files)

        
    def update_param(self, file_name, param_name, new_value):
#        if basename(file_name) not in self.file_names + self.file_paths:
#            raise ValueError("The given file name '{}' "
#                             "does not exist in this parameter file set".format(file_name))
#        _index = self.file_names.index(file_name)
#        _param_file = self[_index]
        _param_file = self[file_name]
        try:
            _param_file.update_param(param_name, new_value)
        except: raise Exception(
            "Failed to update parameter in file: {}".format(_param_file.name))
            
    def write_under_dir(self, dir_path):
        """Write all parameter files under given 'dir_path'"""
        for _file in self:
            try:_file.write_under_dir(dir_path)
            except: raise Exception(
                "Failed to write file '{}' to directory '{}'".format(_file.name, dir_path))


#    def __getitem__(self, file_name):
#        _indices = [_i for _i, _name in enumerate(self) if _name == file_name]
#        _num_of_matched_indices = len(_indices)
#        if _num_of_matched_indices == 0: 
#            raise KeyError("No item matched for {}".format(file_name))
#        elif _num_of_matched_indices > 1:
#            _msg = "Multiple matches for given `file_name`:{}"
#            raise Exception(_msg.format(file_name))
#        _index, = _indices
#        return self[_index]


