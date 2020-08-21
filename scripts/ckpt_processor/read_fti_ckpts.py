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
pattern_config = "*config*"
ckpts = []
metas = []
config_file = ""
execution_id = ""

#takes list of meta files, processes the 
#corresponding ckpt files per meta file
def map_ckpt_to_meta(metas, app_root):
	meta_to_ckpt_mapping = {}
	for meta in metas:
		configs = []
		config = configparser.ConfigParser()
		config.read(meta)
		for section in config.sections():
			if section.isdigit():
				ckptfile = config[section]['ckpt_file_name']
				print(ckptfile)
				ckptfile = app_root+"/local/node"+str(section)+"/"+execution_id+"/l1/"+ckptfile
				print("full path: ",ckptfile)
				meta_to_ckpt_mapping[ckptfile] = meta
	return meta_to_ckpt_mapping

#reads the execution_id token for files extraction
def get_execution_id(config_file):
	print("reading config::", config_file)
	global execution_id
	config = configparser.ConfigParser()
	config.read(config_file)
	#read group_size
	execution_id = config['restart']['exec_id']
	print("exec_id :", execution_id)

#fetches fti's necessary files: config, metas, ckpts
def read_fti_files(app_root):
	if os.path.isdir(app_root) == True:
		print("directory ", app_root, " exists")
	# get the ckpt files 
	for path, subdirs, files in os.walk(app_root+"/local"):
		for name in files:
			if fnmatch(name, pattern_ckpt):
				ckpts.append(os.path.join(path, name))
				#TODO: check if ckpt file is empty before proceeding
	# get the meta files
	for path, subdirs, files in os.walk(app_root+"/meta"):
		for name in files:
			if fnmatch(name, pattern_meta):
				metas.append(os.path.join(path, name))

	#get the config file
	for path, subdirs, files in os.walk(app_root):
		for name in files:
			if fnmatch(name, pattern_config):
				metas.append(os.path.join(path, name))
				config_file = os.path.join(path, name)
	return config_file

#reads the ckpts of given app
def read_checkpoints(app_root):
	config_file = read_fti_files(app_root) #populates metas+ckpts+config
	get_execution_id(config_file) #sets execution_id
	meta_to_ckpt_mapping = map_ckpt_to_meta(metas, app_root)
	for ckpt in meta_to_ckpt_mapping:
		print("Processing ", ckpt, " using meta ", meta_to_ckpt_mapping[ckpt])
		posix_read_ckpts.read_checkpoint(ckpt, meta_to_ckpt_mapping[ckpt], config_file)
