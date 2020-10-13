# This module decodes the checkpoint
# files and returns the output according 
# to the desired output format

import os
import os.path
import configparser
import struct
import re
import sys
import csv
import time
from itertools import zip_longest
import numpy as np
import pandas as pd
import h5py

# Error codes
# 1001: 'Unknown level'
# 1002: 'Invalid output format'
# 1003: 'Missing totalRanks'
# 2001: 'Config file not found'
# 2002: 'Checkpoint file not found'
# 2003: 'Checkpoint file empty'
# 2004: 'Meta file not found'
# 2005: 'Failed to recover MPI-IO file'
# 3001: 'No variable found in checkpoint'
# 3002: 'Error while writing data to CSV'
# 3003: 'Unavailable mapping of long double for decoding'
# 3004: 'Unsupported decoding for self-defined variable'
# 3005: 'Unsupported decoding for given data type'

nbVars = 0
ckpt_file_size = 0
varheaders = []  # var headers


# variable object
class variable(object):
    def __init__(self, var_id, var_size, var_typeid,
                 var_typesize, var_position,
                 var_name=None, var_ndims=None, var_dims=None):
        self.var_id = var_id
        self.var_size = var_size
        self.var_typeid = var_typeid
        self.var_typesize = var_typesize
        self.var_position = var_position
        if var_name is not None:
            self.var_name = var_name
        if var_ndims is not None:
            self.var_ndims = var_ndims
        if var_dims is not None:
            self.var_dims = var_dims


# This function reads the given meta data
# and returns a list of the variables found
# in the ckpt file
def read_meta(meta_file, ckpt_file, group_size, level):
    ckpt_file = ckpt_file.rsplit('/', 1)[1]
    mysection = ""
    data = []
    # count nbVars from meta_file
    regex = "var[-0-9]+_id"
    var_pattern = re.compile(regex)
    # parse and get value by key
    print("reading meta file:", meta_file)
    config = configparser.ConfigParser()
    config.read(meta_file)

    # get nbVars
    global nbVars
    nbVars = 0  # initialize it for every meta file
    if level == 4:
        # For level 4, ckpt file name does not match ckpt_file_name's value
        for section in config.sections():  # traverse sections
            # read from the first section with a digit as a key
            if section.isdigit() is True:
                mysection = section

    else:  # for any level
        for section in config.sections():  # traverse sections
            if section.isdigit() is True:
                if config[section]['ckpt_file_name'] == ckpt_file:
                    mysection = section
                    break

    for (each_key, each_val) in config.items(mysection):
        # check var pattern to increment nbVars variable
        if var_pattern.match(each_key) and each_key.endswith('_id'):
            nbVars = nbVars + 1
    if nbVars == 0:
        print("No variable found in Checkpoint file")
        sys.exit(3001)
    print("Number of variables to read = "+str(nbVars))
    # create numpy array for variables (instead of data)
    datanumpy = np.array([])
    # get data for each Var
    # for i in range(int(group_size)):
    for j in range(nbVars):
        var_id = config['0']['var'+str(j)+'_id']
        var_size = config['0']['var'+str(j)+'_size']
        var_typeid = config['0']['var'+str(j)+'_typeid']
        var_typesize = config['0']['var'+str(j)+'_typesize']
        var_position = config['0']['var'+str(j)+'_pos']
        var_name = None
        var_ndims = 0
        var_dims = []

        if (config.has_option('0', 'var'+str(j)+'_name') is True and
                config['0']['var'+str(j)+'_name']):
            var_name = config['0']['var'+str(j)+'_name']
        if config.has_option('0', 'var'+str(j)+'_ndims') is True:
            # if variable dims set by FTI_SetAttribute()
            var_ndims = int(config['0']['var'+str(j)+'_ndims'])
            if var_ndims != 0:
                for k in range(var_ndims):
                    dim = config['0']['var'+str(j)+'_dim'+str(k)]
                    var_dims.append(dim)

        datanumpy = np.append(datanumpy, variable(var_id, var_size,
                                                  var_typeid, var_typesize,
                                                  var_position, var_name,
                                                  var_ndims, var_dims))
    return datanumpy


# This function reads the ckpt file
# and saves its content to out.csv
def read_checkpoint(ckpt_file, meta_file,
                    config_file, group_size,
                    level, output):

    if os.path.exists(ckpt_file) is False:  # FileNotFoundError
        print("No checkpoint file found")
    else:
        if os.stat(ckpt_file).st_size == 0:
            print("Checkpoint file empty")
            sys.exit(2003)
        else:
            print(
                "Found checkpoint file with size ",
                os.path.getsize(ckpt_file))
            file = open(ckpt_file, "rb")
            # read meta data
            data = read_meta(meta_file, ckpt_file, group_size, level)

            # read Checkpoint
            allvarsnumpy = np.empty((1, nbVars), dtype=object)
            for i in range(nbVars):
                # for each variable:  create list per variable to hold
                # the value of the variable to be exported to the csv file

                file.seek(int(data[i].var_position), os.SEEK_SET)
                var = file.read(int(data[i].var_size))
                # process the datatype
                if int(data[i].var_typeid) == -1:
                    print("skipping var#", str(i))
                    print("Not a primitive data type")
                    # skip this variable
                    # only primitive types are decoded as of this version
                    continue
                decode_pattern, dtype = decode_fti_type(data[i].var_typeid)
                # data[i].var_ndims already has data
                # if var has no dimension:: one element
                data[i].var_ndims = int(data[i].var_ndims)
                # should verify if dimensions are correct

                if (int(data[i].var_size) == int(data[i].var_typesize)
                   and data[i].var_ndims == 0):
                    # single var
                    decoded_var = struct.unpack(decode_pattern, var)
                    varnumpy = np.array([])
                    varnumpy = np.append(varnumpy, decoded_var)

                else:  # multi-dim variable
                    subvars = int(data[i].var_size) \
                            // (int(data[i].var_typesize))
                    decode_pattern = str(subvars)+decode_pattern
                    decoded_var = struct.unpack(decode_pattern, var)

                    varnumpy = np.array([])
                    # for v in range(subvars):
                    varnumpy = np.append(varnumpy, decoded_var)  # needs debugging

                if hasattr(data[i], 'var_name'):
                    varheaders.append(data[i].var_name)
                else:
                    varheaders.append("var#"+str(i))

                allvarsnumpy[0, i] = varnumpy

            file.close()
            allvarsnumpy = allvarsnumpy[0]
            if output == 'CSV':
                write_data_to_csv(allvarsnumpy, varheaders)
            elif output == 'HDF5':
                write_data_to_hdf5(allvarsnumpy, varheaders)
            elif output == 'data':
                return allvarsnumpy


# This function writes the variables
# stored in a numpy array to the ouput csv file
def write_data_to_csv(allvarsnumpy, varheaders):
    panda = pd.DataFrame()
    lengths = []
    # traverse numpy arrays one by one
    for i in range(nbVars):
        lengths.append(len(allvarsnumpy[i].tolist()))

    maxlength = max(lengths)

    for i in range(nbVars):
        length = len(allvarsnumpy[i].tolist())
        if not length == maxlength:
            toFill = maxlength - length
            for j in range(toFill):
                allvarsnumpy[i] = np.append(allvarsnumpy[i], '')
            #print(allvarsnumpy[i])

            panda[varheaders[i]] = allvarsnumpy[i].tolist()
        elif length == maxlength:
            panda[varheaders[i]] = allvarsnumpy[i].tolist()

    panda.to_csv("checkpoints.csv", sep='\t', encoding='utf-8', index=False)


# to be implemented later to store
# the data numpy array in hf format
def write_data_to_hdf5(allvarsnumpy, varheaders):
    lengths = []
    for i in range(nbVars):
        #allvarsnumpy[i] is one array
        lengths.append(len(allvarsnumpy[i]))

    maxlength = max(lengths)

    for i in range(nbVars):
        length = len(allvarsnumpy[i])
        if not length == maxlength:
            #allvarsnumpy[i].extend(['']*(maxlength-length))
            toFill = maxlength - length
            for j in range(toFill):
                mat[i] = np.append(allvarsnumpy[i], None)

    with h5py.File('checkpoints.h5', 'w') as hf:
        for i in range(nbVars):
            allvarsnumpy[i] = np.asarray(allvarsnumpy[i])
            allvarsnumpy[i] = allvarsnumpy[i].astype(np.float)
            print(allvarsnumpy[i])
            hf.create_dataset(varheaders[i], data=allvarsnumpy[i], dtype='float')
            

# This function returns the struct
# decode pattern for the given FTI type
# and the dtype for python numpy
def decode_fti_type(fti_type):
    decode_pattern = ''
    numpy_dtype = ''

    if fti_type == '0':  # char
        decode_pattern = 'c'
        numpy_dtype = np.byte
    elif fti_type == '1':  # short
        decode_pattern = 'h'
        numpy_dtype = np.short
    elif fti_type == '2':  # int
        decode_pattern = 'i'
        numpy_dtype = np.intc
    elif fti_type == '3':  # long int
        decode_pattern = 'l'
        numpy_dtype = np.int_
    elif fti_type == '4':  # uchar
        decode_pattern = 'B'
        numpy_dtype = np.ubyte
    elif fti_type == '5':  # ushort
        decode_pattern = 'H'
        numpy_dtype = np.ushort
    elif fti_type == '6':  # unint
        decode_pattern = 'I'
        numpy_dtype = np.uintc
    elif fti_type == '7':  # ulong
        decode_pattern = 'L'
        numpy_dtype = np.uint
    elif fti_type == '8':  # float
        decode_pattern = 'f'
        numpy_dtype = np.single
    elif fti_type == '9':  # double
        decode_pattern = 'd'
        numpy_dtype = np.double
    elif fti_type == '10':  # long double
        print("TODO: Unavailable mapping of long double for decoding")
        sys.exit(3003)
    elif fti_type == '-1':
        # TODO: implement self-defined types
        print("TODO: Unsupported decoding for self-defined variable")
        sys.exit(3004)
    else:
        print("Unsupported decoding for given data type")
        sys.exit(3005)

    return decode_pattern, numpy_dtype
