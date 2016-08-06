import numpy as np

train_idxs = range(1,601)
del train_idxs[417-1] # sample 417 is missing, because it had corrput data
train_idxs = np.array(train_idxs, dtype="uint")

valid_idxs = np.arange(601,701) # 100 sample files
test_idxs = np.arange(701,941) # 240 sample files

def use_split2():
	global train_idxs, valid_idxs
	np.random.seed(1337)

	idxs = range(1,701)
	del idxs[417-1] # sample 417 is missing, because it had corrput data
	idxs = np.array(idxs, dtype="uint")
	np.random.shuffle(idxs)

	valid_idxs = idxs[:100] # 100 sample files
	train_idxs = idxs[100:] # 100 sample files

def print_split():
	print "train", len(train_idxs), train_idxs[:10]
	print "valid", len(valid_idxs), valid_idxs[:10]
	print "test", len(test_idxs), test_idxs[:10]


if __name__ == '__main__':
	print_split()
	print

	use_split2()

	print_split()