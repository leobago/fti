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
group_size = 0
ckpt_file_size = 0
d = {} #temp dictionary struct
var_labels = [] #header for csv file

#varibale object
class variable(object):
	def __init__(self, var_id, var_size, var_position, var_name):
		self.var_id = var_id
		self.var_size = var_size
		self.var_position = var_position
		self.var_name = var_name

#This function reads the given meta data
#and returns a list of the variables found 
#in the ckpt file
def read_meta(meta_file, ckpt_file):
	ckpt_file = ckpt_file.rsplit('/', 1)[1]
	mysection = ""
	data=[]
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
		if var_pattern.match(each_key):
			nbVars = nbVars + 1
	print("Number of variables to read = "+str(nbVars))
	#get data for each Var
	for i in range(int(group_size)):
		for j in range(nbVars):
			var_id = config[str(i)]['var'+str(j)+'_id']
			var_size = config[str(i)]['var'+str(j)+'_size']
			var_position = config[str(i)]['var'+str(j)+'_pos']
			var_name = config[str(i)]['var'+str(j)+'_name']
			print('id: '+var_id+' size:'+var_size+' pos:'+var_position+' name:'+var_name)
			var = data.append(variable(var_id, var_size, var_position, var_name))
	return data

#This function reads the ckpt file
#and saves its content to out.csv
def read_checkpoint(ckpt_file, meta_file, config_file):
	read_config(config_file)
	if os.path.exists(ckpt_file) == False:
		print("No checkpoint file found")
	else:
		if os.stat(ckpt_file).st_size == 0:
			print("Checkpoint file empty")
		else: 
			print("Found checkpoint file with size ", os.path.getsize(ckpt_file))
			file = open(ckpt_file, "rb")
			#read meta data
			data = read_meta(meta_file, ckpt_file)
			#read Checkpoint
			for i in range(nbVars):
			#for each variable:  create list per variable to hold 
			# the value of the variable to be exported to the csv file
				var_labels.append("var#"+str(i))
				var_array = []
				print("reading var #", str(i), " of size ", str(data[i].var_size),
				 " starting pos:", str(data[i].var_position))
				print("current position ",file.tell())
				file.seek(int(data[i].var_position), os.SEEK_SET)
				var = file.read(int(data[i].var_size))

				if int(data[i].var_size) == 4: #int
					decoded_var = struct.unpack('<I', var)[0]
					var_array.append(decoded_var)
				elif int(data[i].var_size) % 8 == 0: #double
					subvars = int(data[i].var_size) // 8 #array elements
					pattern = str(subvars)+"d"
					decoded_var = struct.unpack(pattern, var)
					var_array.append(decoded_var)
				d[i] = var_array
			file.close()

			#write to csv file
			biggerlist = []
			for key in d.keys():
				print(key)
				for value in d[key]:
					if type(value) == tuple:
						biggerlist.append(list(value))
					elif type(value) != tuple and type(value) != list:
						arr = []
						arr.append(value)
						biggerlist.append(arr)

			export_data = zip_longest(*biggerlist, fillvalue = '')
			with open('out.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
				wr = csv.writer(myfile)
				wr.writerow(var_labels)
				wr.writerows(export_data)
			myfile.close()

#This function reads the config_file to extract 
#the group size
def read_config(config_file):
	config = configparser.ConfigParser()
	config.read(config_file)
	#read group_size
	global group_size
	group_size = config['basic']['group_size']
