from os.path import isfile
import re


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
#    param_exist = ' '.join([param_name, param_type]) in file_content_original
    param_exist = param_exists(file_content_original, param_name, param_type)
    if param_exist:
        pattern = ' '.join([param_name, param_type, '.*'])
        new_string = ' '.join([param_name, param_type, new_value_str])
#        print("parttern: {0} / subs: {1}".format(pattern, new_string))
        file_content_updated = re.sub(pattern, new_string, file_content_original)
    else: 
        err_mesg = "Couldn't find parameter with name {0} of type {1} in file {2}"
        raise Exception(err_mesg.format(param_name, param_type, filepath))
    
    with open(filepath, "w") as f: f.write(file_content_updated)
#    print("written content: ",file_content_updated)

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
