Patterns file names:
	L1, L2, L3, L4 : levels of checkpoints
	INIT : for first execution (program stops after 63 iterations (checkpoints 1 - 7)
	Clean : for second exection, where there was no corruption or erasion (starts from 60 iteration; checkpoints 8 - 12)
		If after "Clean" word is the number 4, it means that checkpoint were flushed to L4
	First number after level (L1/L2/L3/L4) is: 0 - corruping checkpoint files; 1 - erasing checkpoint files
	Next number (if necessary) is corrupting/erasing:  0 - one file; 1 - two non adjacent nodes; 2 - two adjacent nodes; 3 - all files (ckpt or ptner)
