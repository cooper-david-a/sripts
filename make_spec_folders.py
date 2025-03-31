import os

path1 = r'H:\Downloaded specs'
path2 = r'G:\General\Specifications\SAE'
root = []
dirs = []
file_list = []

(root, dirs, file_list) = next(os.walk(path1))

AMS_specs = [file for file in file_list if file.startswith('AMS')]

AMS_specs_split = [file.split(' ',1)[0].split('.',1)[0] for file in AMS_specs]

AMS_spec_numbers = [file[3:7] for file in AMS_specs_split if file[3:7].isdigit()]

