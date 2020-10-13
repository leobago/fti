import sys
sys.path.append("..")

import read_fti_ckpts

config_file = sys.argv[1]
rank_id = 0
level = 4


read_fti_ckpts.read_checkpoints(config_file, rank_id=0, ranks=8, level=level, output='CSV')
