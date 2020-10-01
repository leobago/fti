import sys
sys.path.append("..")

import read_fti_ckpts

config_file = sys.argv[1]
rank_id = 0
level = 1


read_fti_ckpts.read_checkpoints(config_file, rank_id=0, level=level, output='CSV')
