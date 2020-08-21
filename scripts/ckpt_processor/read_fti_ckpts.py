#this module traverses the meta and ckpt files
#in an App root folder to process them
#for variable extraction
import os
import time
from fnmatch import fnmatch
import configparser
import posix_read_ckpts

pattern_ckpt = "*.fti"
pattern_meta = "sector*"
ckpts = []
metas = []
ckpt_dir = ""
meta_dir = ""
ckpt_abs_path = ""
meta_abs_path = ""
config_file = ""
execution_id = ""

#takes list of meta files, processes the 
#corresponding ckpt files per meta file
def map_ckpt_to_meta(metas):
	meta_to_ckpt_mapping = {}
	for meta in metas:
		configs = []
		config = configparser.ConfigParser()
		config.read(meta)
		for section in config.sections():
			if section.isdigit():
				ckptfile = config[section]['ckpt_file_name']
				print(ckptfile)
				ckptfile = ckpt_abs_path+"/node"+str(section)+"/"+execution_id+"/l1/"+ckptfile
				meta_to_ckpt_mapping[ckptfile] = meta
	return meta_to_ckpt_mapping

#reads the execution_id token for files extraction
def init_config_params(config_file):
	global execution_id
	global ckpt_dir
	global meta_dir
	config = configparser.ConfigParser()
	config.read(config_file)
	#read group_size
	execution_id = config['restart']['exec_id']
	ckpt_dir = config['basic']['ckpt_dir']
	meta_dir = config['basic']['meta_dir']

def process_fti_paths(config_file):
	global ckpt_dir
	global meta_dir
	global ckpt_abs_path
	global meta_abs_path
	#ckpt dir
	dir_path = os.path.dirname(os.path.realpath(config_file))
	print(dir_path)
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


#fetches fti's necessary files: config, metas, ckpts
def read_fti_files(config_file):
	process_fti_paths(config_file)
	# get the ckpt files 
	for path, subdirs, files in os.walk(ckpt_abs_path):
		for name in files:
			if fnmatch(name, pattern_ckpt):
				ckpts.append(os.path.join(path, name))
				#TODO: check if ckpt file is empty before proceeding
	# get the meta files
	for path, subdirs, files in os.walk(meta_abs_path):
		for name in files:
			if fnmatch(name, pattern_meta):
				metas.append(os.path.join(path, name))

#reads the ckpts of given app
def read_checkpoints(config_file):
	init_config_params(config_file) #sets execution_id
	read_fti_files(config_file) #populates metas+ckpts+config
	meta_to_ckpt_mapping = map_ckpt_to_meta(metas)
	for ckpt in meta_to_ckpt_mapping:
		print("Processing ", ckpt, " using meta ", meta_to_ckpt_mapping[ckpt])
		posix_read_ckpts.read_checkpoint(ckpt, meta_to_ckpt_mapping[ckpt], config_file)
