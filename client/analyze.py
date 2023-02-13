import numpy as np

eval = np.load("data/ssl_v2_dummy.npz")["rewards"]

print("Size: ", eval.size)

print(np.mean(eval))

print(np.std(eval)/np.sqrt(500))
