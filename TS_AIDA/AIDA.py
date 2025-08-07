import numpy as np
import math



from TS_AIDA.Relative_contrast import rel_contrast

def Eu_norm(X, Ys, p=1):
	Xs = np.repeat(X, Ys.shape[0], axis=0)
	dif = (np.abs(Xs - Ys)) ** p
	return np.sum(dif, axis=1)

def Mahalanobis(Xs, Ys, cov_inv, p=1):


	dif = Xs-Ys
	A = cov_inv.dot(dif[1,:])
	B = dif.dot(A)
	C =np.sqrt(B)
	return C

def DistanceProfile(seq, coll, p=2):
	seq = seq[np.newaxis, :]
	A = np.repeat(seq, coll.shape[0], axis=0)
	dif = (np.abs(A - coll)) ** p
	S = np.sum(dif, axis=1)
	if len(A.shape) == 3:
		S = np.sum(S, axis=1)
	sorted = np.sort(S)
	del S, dif, A, seq
	return sorted

def DistanceProfile_maha(seq1, coll1, p=2):
	seq = seq1[np.newaxis, 1:]
	A = np.repeat(seq, coll1.shape[0], axis=0)
	coll = coll1[:, 1:]
	cov = np.cov(coll.transpose())
	#np.fill_diagonal(cov, 1)
	try:
		cov_inv = np.linalg.inv(cov)
		S = Mahalanobis(A, coll, cov_inv)
	except np.linalg.LinAlgError:
		print("LinAlgError occurred")
		S = DistanceProfile(seq1, coll1, p=p)

	if len(A.shape) == 3:
		S = np.sum(S, axis=1)
	sorted = np.sort(S)

	return sorted

def Isolation(Z_n, alpha=1):
	Z_top =  Z_n[1:] - Z_n[0:-1]
	Z_bot = Z_n[1:] - Z_n[0]
	div = (Z_top/Z_bot)[1:]
	div = np.nan_to_num(div, copy=True, nan=0.0)
	mean = 1 + np.sum(div)
	var = np.sum(div*(1-div))
	if math.isnan(var):
		pause=1
	Z_top, Z_bot, div = None, None, None
	return mean, var

def DFT(Z):
	Z_hat = np.fft.fftn(Z, axes=(1,))
	return np.abs(Z_hat)

def Score(Z_i, Z, p=1):
	profile = DistanceProfile(Z_i, Z, p=p)
	cont = rel_contrast(profile)
	mean, var = Isolation(profile)
	if math.isnan(var):
		pause=1
	profile =None
	return mean, var, cont

def Score_maha(Z_i, Z, p=1):
	profile = DistanceProfile_maha(Z_i, Z, p=p)
	cont = rel_contrast(profile)
	mean, var = Isolation(profile)
	if math.isnan(var):
		pause=1
	profile =None
	return mean, var, cont