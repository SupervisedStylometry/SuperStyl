import pandas

# Just some code to count authors
data = pandas.read_csv("feats.csv")
unique_auts = set(data.loc[:, "author"])
len(unique_auts)

# In train
data = pandas.read_csv("feats_tests_train.csv")
unique_auts = set(data.loc[:, "author"])
len(unique_auts) # 252
del data

# In dev
data = pandas.read_csv("feats_tests_valid.csv")
unique_auts_valid = set(data.loc[:, "author"])
len(unique_auts_valid) # 200
del data

# In train but not in dev
for u in unique_auts_valid:
	if u not in unique_auts:
		print(u)

# Wouter Robbertz

