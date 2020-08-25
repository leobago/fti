#this module traverses the meta file
#given the rank
#in given the config_file

import os, glob
import time
from fnmatch import fnmatch
import configparser
import posix_read_ckpts

#runtime variables of FTI (ckpt and meta)
fti_levels = (1,2,3,4)
config_file = ""
ckpt_dir = ""
meta_dir = ""
global_dir = ""
ckpt_abs_path = ""
meta_abs_path = ""
execution_id = ""
level_meta_dir = ""
level_dir = ""

#This function reads the config_file
#to get execution_id, ckpt_dir and meta_dir
def init_config_params(config_file):
	global execution_id
	global ckpt_dir
	global meta_dir
	global global_dir
	config = configparser.ConfigParser()
	config.read(config_file)
	execution_id = config['restart']['exec_id']
	ckpt_dir = config['basic']['ckpt_dir']
	meta_dir = config['basic']['meta_dir']
	global_dir = config['basic']['global_dir']


#This function processes FTI's files
#given config_file and set the absolute
#paths of meta files and ckpt files
def process_fti_paths(config_file):
	global ckpt_dir
	global meta_dir
	global ckpt_abs_path
	global meta_abs_path
	#ckpt dir
	#TODO: should evaluate for global_dir 
	dir_path = os.path.dirname(os.path.realpath(config_file))
	#concatenate paths
	if ckpt_dir.startswith('./') == True: #same directory as config
		ckpt_abs_path = dir_path + ckpt_dir.replace('.','')
	elif "." not in ckpt_dir: #absolute path
		#directly change dirs to 
		ckpt_abs_path = ckpt_dir
	else: #relative path
		#iterate over the number of '../' found in ckpt_path
		os.chdir(dir_path)
		dirs = ckpt_dir.count("..")
		print("dirs : ", dirs)
		for i in range(dirs):
			os.chdir("..")
		#concatenate the remaining part
		for i in range(dirs):
			#remove ../
			ckpt_dir = ckpt_dir.replace('../','')
		os.chdir(ckpt_dir)
		ckpt_abs_path = os.getcwd()
	print("ckpt_abs_path ", ckpt_abs_path)

	#meta dir
	dir_path = os.path.dirname(os.path.realpath(config_file))
	print(dir_path)
	#concatenate paths
	if meta_dir.startswith('./') == True: #same directory as config
		#omit dot + concatenate the rest of the path
		meta_abs_path = dir_path + meta_dir.replace('.','')
	elif "." not in meta_dir: #absolute path
		#directly change dirs to 
		meta_abs_path = meta_dir
	else: #relative path
		#iterate over the number of '../' found in ckpt_path
		os.chdir(dir_path)
		dirs = meta_dir.count("..")
		print("dirs : ", dirs)
		for i in range(dirs):
			os.chdir("..")
		#concatenate the remaining part
		for i in range(dirs):
			#remove ../
			meta_dir = meta_dir.replace('../','')
		os.chdir(meta_dir)
		meta_abs_path = os.getcwd()
	print("meta_abs_path ", meta_abs_path)


#This function returns the path of the
#ckpt corresponding to rank_id
def find_ckpt_file(rank_id):
	pattern_ckpt_file = "*-Rank"+str(rank_id)+".fti"
	pattern_ckpt_path = execution_id+level_dir
	ckpt_file = ""
	# for path, subdirs, files in os.walk(ckpt_abs_path):
	# 	for name in files:
	# 		if fnmatch(name, pattern_ckpt):
	# 			ckpt_file = os.path.join(path, name)
	for root, dirs, files in os.walk(os.path.abspath(ckpt_abs_path)):
		for file in files:
			file = os.path.join(root, file)
			if pattern_ckpt_path in file and fnmatch(file, pattern_ckpt_file):
				ckpt_file = file
				print("FOUND CKPT ", file)

	return ckpt_file


#This function returns the path of the
#meta corresponding to the ckpt_file
#note: for now it works with level 1
def find_meta_file(ckpt_file):
	meta_file = ""
	#traverse all meta files in the directory
	for path, subdirs, files in os.walk(meta_abs_path):
		for file in files:
			file = meta_abs_path+'/'+execution_id+level_dir+file
			#print("META FOUND:::",file)
			if os.path.isfile(file) == True:
				config = configparser.ConfigParser()
				config.read(file)
				ckpt = ckpt_file.rsplit('/', 1)[1]
				for section in config.sections():
					if section.isdigit() == True:
						if config[section]['ckpt_file_name'] == ckpt:
							meta_file = file
							break;
	return meta_file

#This function sets FTI's files paths
#depending on the level
def process_level(level):
	global level_dir
	if level in (1, 2, 3):
		level_dir = '/l'+str(level)+'/'
		print("level dir : ", level_dir)
	elif level == 4:
		#should fetch the global_dir path from config_file
		

#This function compares ckpt directories
#and returns the directory that has the latest ckpt
def get_latest_ckpt():
	latest = max(glob.glob(os.path.join(ckpt_abs_path, '*/')), key=os.path.getmtime)
	latest = latest.rsplit('/', 1)[0]
	latest = latest.rsplit('/', 1)[1]
	level = latest[1]
	return level

#API to read the checkpoints given config and rank
def read_checkpoints(config_file, rank_id, level=None):
	init_config_params(config_file)
	process_fti_paths(config_file)
	if level in fti_levels:
		process_level(level)
	else:
		#check for latest ckpt
		last_level = get_latest_ckpt()
		process_level(level)

	ckpt_file = find_ckpt_file(rank_id) 
	meta_file = find_meta_file(ckpt_file)
	print("Processing ", ckpt_file, " using meta ", meta_file)
	posix_read_ckpts.read_checkpoint(ckpt_file, meta_file, config_file)
