import random
import numpy
import h5py
import random
import math
import os

#directories and files
dir_current      = os. getcwd()
dir_parent       = os.path.split(dir_current)[0]
dir_code_keras   = os.path.join(dir_parent,'code-keras')
dir_code_scikit  = os.path.join(dir_parent,'code-scikit')
dir_data_matlab  = os.path.join(dir_parent,'data-matlab')
dir_data_python  = os.path.join(dir_parent,'data-python')

file_data_python = os.path.join(dir_data_python,'data.npy')
file_ind_python  = os.path.join(dir_data_python,'ind.npy')
file_data_matlab = os.path.join(dir_data_matlab,'data_inou_scl.txt')

#######################

rtt = range(0,5)
rrs = range(0,5)

## load data from .mat and save it in .npy
#with h5py.File(file_data_matlab, 'r') as f:
#   input_data = list(f['datain'])
#   output_data = list(f['dataou'])
f           = open(file_data_matlab,'r')
f_val       = f.read()
data_dict   =  eval(f_val)
input_data  = data_dict['datain']
output_data = data_dict['dataou']
f.close()

data = {'input_data':numpy.array(input_data),'output_data':numpy.array(output_data)}
print(data)

numpy.save(file_data_python, data)


## permutation and geting the indices 
l_data     = len(input_data)   
range_data = list(range(1,l_data))

random.seed(a=None)
ind_train_all = []
ind_test_all  = []
for i0 in rtt:
    ind_test  = []
    ind_train = []
    for i1 in rrs:
        print('{} _ {}'.format(i0+1,i1+1))
        range_data = list(range(1,l_data)) 
        random.shuffle(range_data)
        ind_test.append(range_data[1:math.floor(((i0+1)/1000)*l_data)+1])
        ind_train.append(range_data[math.floor(((i0+1)/1000)*l_data)+2:])
    ind_train_all.append(ind_train)
    ind_test_all.append(ind_test)

ind_all = {'ind_train':ind_train_all,'ind_test':ind_test_all}

numpy.save(file_ind_python, ind_all)
