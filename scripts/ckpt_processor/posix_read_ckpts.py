# for now this file works for HeatDistribution app
import os
import os.path
import configparser
import struct
import re
import sys
import csv
import time
from itertools import zip_longest

nbVars = 0
ckpt_file_size = 0
d = {} #temp dictionary struct
var_labels = [] #header for csv file

#variable object
class variable(object):
	def __init__(self, var_id, var_size, var_typeid, var_typesize, var_position,
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

#This function reads the given meta data
#and returns a list of the variables found 
#in the ckpt file
def read_meta(meta_file, ckpt_file, group_size):
	ckpt_file = ckpt_file.rsplit('/', 1)[1]
	mysection = ""
	data = []
	#count nbVars from meta_file
	regex = "var[-0-9]+_id"
	var_pattern = re.compile(regex)
	#parse and get value by key
	config = configparser.ConfigParser()
	config.read(meta_file)

	#get nbVars
	global nbVars
	nbVars = 0 #initialize it for every meta file
	for section in config.sections(): #traverse sections
		if section.isdigit() == True:
			if config[section]['ckpt_file_name'] == ckpt_file:
				#this is the correct section from where to read the number of vars
				mysection = section

	for (each_key, each_val) in config.items(mysection):
		#check var pattern to increment nbVars variable
		if var_pattern.match(each_key) and each_key.endswith('_id'):
			#print("matching ...", each_key)
			nbVars = nbVars + 1
	print("Number of variables to read = "+str(nbVars))
	#get data for each Var
	#for i in range(int(group_size)):
	for j in range(nbVars):
		var_id = config['0']['var'+str(j)+'_id']
		var_size = config['0']['var'+str(j)+'_size']
		var_typeid = config['0']['var'+str(j)+'_typeid']
		var_typesize = config['0']['var'+str(j)+'_typesize']
		var_position = config['0']['var'+str(j)+'_pos']
		var_name = None
		var_ndims = 0
		var_dims = []
		if config.has_option('0', 'var'+str(j)+'_name') == True and config['0']['var'+str(j)+'_name']:
			var_name = config['0']['var'+str(j)+'_name']
			#print("var name ", var_name)
		if config.has_option('0', 'var'+str(j)+'_ndims') == True:
			#if variable dims set by FTI_SetAttribute()
			var_ndims = int(config['0']['var'+str(j)+'_ndims'])
			#print("var_ndims =>", str(var_ndims))
			if var_ndims != 0:
				for k in range(var_ndims):
					dim = config['0']['var'+str(j)+'_dim'+str(k)]
					var_dims.append(dim)
					#print("variable", str(i)," dim #", str(j), " ", dim)

		#print('id: '+var_id+' size:'+var_size+' pos:'+var_position+' name:'+var_name)
		#print("registering variable ",str(j)," with ndims=",var_ndims)
		var = data.append(variable(var_id, var_size, var_typeid, var_typesize,
		 var_position, var_name, var_ndims, var_dims))
	return data

#This function reads the ckpt file
#and saves its content to out.csv
def read_checkpoint(ckpt_file, meta_file, config_file, group_size):
	#read_config(config_file)
	if os.path.exists(ckpt_file) == False:
		print("No checkpoint file found")
	else:
		if os.stat(ckpt_file).st_size == 0:
			print("Checkpoint file empty")
		else: 
			print("Found checkpoint file with size ", os.path.getsize(ckpt_file))
			file = open(ckpt_file, "rb")
			#read meta data
			data = read_meta(meta_file, ckpt_file, group_size)
			
			#read Checkpoint
			for i in range(nbVars):
			#for each variable:  create list per variable to hold 
			# the value of the variable to be exported to the csv file
				var_labels.append("var#"+str(i))
				var_array = [] #arra holding value(s) for every variable

				print("-----verification------")
				if hasattr(data[i], 'var_name') and hasattr(data[i], 'var_ndims'):
					print('id: ',str(data[i].var_id),' size:',str(data[i].var_size),
					 ' typesize: ',str(data[i].var_typesize) ,' ndims:', str(data[i].var_ndims),
					 ' name:',data[i].var_name)
				else:
					print('id: ',str(data[i].var_id),' size:',str(data[i].var_size))

				#print("current position ", file.tell())
				file.seek(int(data[i].var_position), os.SEEK_SET)
				var = file.read(int(data[i].var_size))
				#process the datatype
				decode_pattern = decode_fti_type(data[i].var_typeid)
				#data[i].var_ndims already has data
				#if var has no dimension:: one element
				data[i].var_ndims = int(data[i].var_ndims)
				#should verify if dimensions are correct

				if int(data[i].var_size) == int(data[i].var_typesize) and data[i].var_ndims == 0:
					#single var
					decoded_var = struct.unpack(decode_pattern, var)
					#print(decoded_var)
					var_array.append(decoded_var)
					
				else: #multi-dim variable
					subvars = int(data[i].var_size) // (int(data[i].var_typesize))
					#print("variable is array of ", str(subvars), " elements")
					decode_pattern = str(subvars)+decode_pattern
					decoded_var = struct.unpack(decode_pattern, var)
					#print(decoded_var)
					var_array.append(decoded_var)

				if hasattr(data[i], 'var_name'):
					d[data[i].var_name] = var_array
				else:
					d["var#"+str(i)] = var_array
			file.close()
			#double checking dict content
			for key in d:
				d[key] = list(d[key][0])
				#print("key: ", key, " , value: ", d[key])
			#write to csv file
			write_data_to_csv(d)

#This function writes the variables
#stored in a dictionary to the ouput csv file
def write_data_to_csv(dictionary):
	max_key = ''
	max_value = 0
	#find largest value in dict
	for key in dictionary:
		#traverse every value
		if len(dictionary[key]) > max_value:
			max_value = len(dictionary[key])
			max_key = key
	#print("the largest value in the dict : ", max_value,
	# " and it belongs to key ", max_key)
	#append to all other values until max_value
	for key in dictionary:
		if key != max_key:
			#append from len key to max_value
			for i in range(max_value - len(dictionary[key])):
				dictionary[key].append('')

	#given dictionary (key, value) where value is a list:
	keys = sorted(dictionary.keys())
	with open("out.csv", "w") as outfile:
		writer = csv.writer(outfile, delimiter = "\t")
		writer.writerow(keys)
		writer.writerows(zip(*[dictionary[key] for key in keys]))
	outfile.close()


#This function returns the struct
#decode pattern for the given FTI type
def decode_fti_type(fti_type):
	decode_pattern = ''
	if fti_type == '0': #char
		decode_pattern = 'c'
	elif fti_type == '1': #short
		decode_pattern = 'h'
	elif fti_type == '2': #int
		decode_pattern = 'i'
	elif fti_type == '3': #long int
		decode_pattern = 'l'
	elif fti_type == '4': #uchar
		decode_pattern = 'B'
	elif fti_type == '5': #ushort
		decode_pattern = 'H'
	elif fti_type == '6': #unint
		decode_pattern = 'I'
	elif fti_type == '7': #ulong
		decode_pattern = 'L'
	elif fti_type == '8': #float
		decode_pattern = 'f'
	elif fti_type == '9': #double
		decode_pattern = 'd'
	elif fti_type == '10': #long double
		#unavailable mapping for long double
		decode_pattern = 'c'
	elif fti_type == '-1':
		#self-defined
		#TODO: implement self-defined types
		print("TODO")
	else:
		#TODO: handle error codes
		print("error")
	return decode_pattern
