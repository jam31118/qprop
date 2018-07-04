"""Collection of default values"""

default_config = {
    'wf_file_dimension_order' : ['ell-m','rho'],
    'vecpot_file_name' : 'real-prop-vecpot.dat',
    'wf_file_name' : [
        'current-wf.bin',
        ],
    'wf_index_in_file' : 0,
    'size_of_complex_number' : 16,
    'calc_data_file_extensions' : [".raw", ".bin", ".dat"],
    'param_file_field_per_entry' : 3,
    'param_file_field_names' : ['name', 'type', 'value'],
    'param_file_comment_character' : '#',
    'imag-pot-ampl' : 100.0,
}

def check_param_default_config_consistency():
  assert default_config['param_file_field_per_entry'] == len(default_config['param_file_field_names'])


type2castFunction = {
    'double' : lambda x: float(x),
    'long' : lambda x: int(x),
    'bool' : lambda x: bool(x),
}


## Check consistenc of this default_config dictionaryy
check_param_default_config_consistency()

