import numpy as np
import matplotlib.pyplot as plt
import roughpy as rp


from TS_AIDA import Window as Wd
from TS_AIDA import AIDA as AIDA
from TS_AIDA import Path_signature as Ps
def sampler(paths, sample_info, i):
    paths = paths[:, sample_info[3][i]]

    if len(paths.shape) > 2:
        reps = paths.shape[2]
    else:
        reps = 1
    # hier gemaakt zodat meer dims in een keer.
    paths = paths[:, np.newaxis, :]
    windows = Wd.sliding_window(paths, sample_info[0][i], 1).transpose(0, 2, 1)
    A = np.repeat(sample_info[1][i][:, np.newaxis], paths.shape[2], axis=1)[:, np.newaxis, :]
    subsample = np.take_along_axis(windows, A, axis=0)

    return subsample




def Score_aggregator(paths, sample_info, K, T, normalize=True, sig=False):
    N = sample_info[0].shape[0]
    output = np.zeros(shape=[paths.shape[0],2])
    output_global = np.zeros(shape=[paths.shape[0]])
    for i in range(N):
        if i %50 == 0: print(i)
        score_sub = np.zeros(shape=[paths.shape[0], 2])

        s1 = sampler(paths, sample_info, i)

        if normalize is True:
            s2 = (s1 - np.mean(s1, axis=0)) / s1.std(axis=0)
        if normalize is False:
            s2 = s1


        if sig == True:
            interval = rp.RealInterval(0, sample_info[0][i])
            indices = np.linspace(0.1, T, sample_info[0][i])
            context = rp.get_context(width=s2.shape[2], depth=K, coeffs=rp.DPReal)

            s3 = Ps.sig_rp2(s2, K, interval, indices, context)
        if sig == False:
            s3 = s2

        contrast = np.zeros(shape=[s3.shape[0], 1])

        for j in range(s3.shape[0]):
            #excl = Exclusion_zone(s3[j, :], s3, r)
            score_mean, score_var, cont = AIDA.Score(s3[j, :], s3)
            contrast[j] = cont
            a = sample_info[1][i][j]
            b = sample_info[2][i][j]

            score_sub[a:b, 0] -= score_var
            score_sub[a:b, 1] += 1

        #score_sub[:, 1] = np.where(score_sub[:, 1] == 0, 1, score_sub[:, 1])
            # score_sub[:, 0] = (score_sub[:, 0] - score_sub[:, 0].mean())/ score_sub[:, 0].std()
        output[:, 0] += score_sub[:, 0]
        output[:, 1] += score_sub[:, 1]

        output_global[:] += score_sub[:, 0]/sample_info[0][i]
        # /score_sub[:,1]
            # score_sub[:, 0] = np.nan_to_num(score_sub[:, 0])

    output[:, 1] = np.where(output[:, 1] == 0, 1, output[:, 1])
    output_avg = output[:, 0] / output[:, 1]
    output_global = (output_global - output_global.mean()) / output_global.std()
    output_avg = (output_avg - output_avg.mean()) / output_avg.std()

    return output_avg, contrast, output_global

""""
def Sample_length(N, length_min_max, rng):
    return rng.integers(low=length_min_max[0], high=length_min_max[1], size=N)

def Sample_length_corr(N, length_min_max, rng, prob):
    return rng.choice(np.arange(length_min_max[0], length_min_max[1]), p=prob ,size=N)

def Sample_dim(N, dim_min_max, D, rng):
    if dim_min_max[1] == 1:
        out = np.zeros(N)
    else:
        n_dims = rng.integers(low=dim_min_max[0], high=dim_min_max[1], size=N)
        out = [rng.choice(D, size=n_dims[i], replace=False) for i in range(len(n_dims))]

    return out

def Sample_n(N, n_samples_min_max, rng):
    return rng.integers(low=n_samples_min_max[0], high=n_samples_min_max[1], size=N)

def sub_sample(paths, N, n_samples_min_max, length_min_max, dim_min_max):
    rng = np.random.default_rng()

    sample_lengths =  Sample_length(N, length_min_max, rng)
    sample_n = Sample_n(N, n_samples_min_max, rng)
    if paths.shape[1] > 1:
        sample_d = Sample_dim(N, dim_min_max, paths.shape[1], rng)
    else:
        sample_d = Sample_dim(N, dim_min_max, 1, rng)

    n_windows = paths.shape[0] - sample_lengths
    indices = [np.arange(0, n_windows[i]) for i in range(N)]

    sample_start = [rng.choice(indices[i], size=sample_n[i], replace=False) for i in range(N)]
    sample_end = [sample_start[i] + sample_lengths[i] for i in range(N)]

    sample_info = (sample_lengths, sample_start, sample_end, sample_d)

    return sample_info

def sampler(paths, sample_info, i):
    paths = paths[:, sample_info[3][i]]

    if len(paths.shape) > 2:
        reps = paths.shape[2]
    else:
        reps = 1
    # hier gemaakt zodat meer dims in een keer.
    paths = paths[:, np.newaxis, :]
    windows = Wd.sliding_window(paths, sample_info[0][i], 1).transpose(0, 2, 1)
    A = np.repeat(sample_info[1][i][:, np.newaxis], paths.shape[2], axis=1)[:, np.newaxis, :]
    subsample = np.take_along_axis(windows, A, axis=0)

    return subsample

def Length_corr(N, length_min_max, rng):
    a = np.arange(length_min_max[0], length_min_max[1])

    p0 = 1/(1+a[0]*np.sum(1/a[1:]))
    pn = p0*(a[0]/a)
    return pn



def local_search_corr(paths, N, n_samples_min_max, length_min_max, dim_min_max, w_ratio=5):
    rng = np.random.default_rng()

    sample_lengths = Sample_length(N, length_min_max, rng)
    central_points = rng.integers(0, paths.shape[0], size=N)
    L = Length_corr(N, length_min_max, rng)
    sample_lengths = Sample_length_corr(N, length_min_max, rng, L)

    starts = np.maximum(central_points - w_ratio*sample_lengths, 0)
    ends = np.minimum(central_points + w_ratio*sample_lengths, paths.shape[0]-sample_lengths)
    length_local = 2*w_ratio*sample_lengths
    n_windows = length_local - sample_lengths

    sample_n = Sample_n(N, n_samples_min_max, rng)

    sample_start = []
    sample_end = []
    for i in range(N):
        indices = np.arange(starts[i], ends[i])
        sample_start.append(rng.choice(indices, size=sample_n[i], replace=False))
        sample_end.append(sample_start[i] + sample_lengths[i])


    if paths.shape[1] > 1:
        sample_d = Sample_dim(N, dim_min_max, paths.shape[1], rng)
    else:
        sample_d = Sample_dim(N, dim_min_max, 1, rng)

    sample_info = (sample_lengths, sample_start, sample_end, sample_d)

    return sample_info

def local_search(paths, N, n_samples_min_max, length_min_max, dim_min_max, w_ratio=5, corrected=False):
    rng = np.random.default_rng()
    if corrected == False:
        sample_lengths = Sample_length(N, length_min_max, rng)

    if corrected == True:
        L = Length_corr(N, length_min_max, rng)
        sample_lengths = Sample_length_corr(N, length_min_max, rng, L)

    central_points = rng.integers(0, paths.shape[0], size=N)

    starts = np.maximum(central_points - w_ratio*sample_lengths, 0)
    ends = np.minimum(central_points + w_ratio*sample_lengths, paths.shape[0]-sample_lengths)
    length_local = 2*w_ratio*sample_lengths
    n_windows = length_local - sample_lengths

    sample_n = Sample_n(N, n_samples_min_max, rng)

    sample_start = []
    sample_end = []
    for i in range(N):
        indices = np.arange(starts[i], ends[i])
        sample_start.append(rng.choice(indices, size=sample_n[i], replace=False))
        sample_end.append(sample_start[i] + sample_lengths[i])


    if paths.shape[1] > 1:
        sample_d = Sample_dim(N, dim_min_max, paths.shape[1], rng)
    else:
        sample_d = Sample_dim(N, dim_min_max, 1, rng)

    sample_info = (sample_lengths, sample_start, sample_end, sample_d)

    return sample_info
    
    
def sub_sampler(paths, sample_info, transform, normalize=True):
    N = sample_info[0].shape[0]
    output = np.zeros(shape=[paths.shape[0],2])

    for i in range(N):
        print(i)
        score_sub = np.zeros(shape=[paths.shape[0], 2])
        s1 = sampler(paths, sample_info, i)
        if normalize is True:
            s2 = (s1 - np.mean(s1, axis=0)) / s1.std(axis=0)
        if normalize is False:
            s2 = s1
        # x = samples, y = values
        s3 = transform(s2)

        contrast = np.zeros(shape=[s3.shape[0], 1])


        for j in range(s3.shape[0]):
            score_mean, score_var, cont = AIDA.Score(s3[j, :], s3)
            contrast[j] = cont
            a = sample_info[1][i][j]
            b = sample_info[2][i][j]

            score_sub[a:b, 0] -= score_var
            score_sub[a:b, 1] += 1

        #score_sub[:, 1] = np.where(score_sub[:, 1] == 0, 1, score_sub[:, 1])
        #score_sub[:, 0] = score_sub[:, 0] / score_sub[:, 1]
        score_sub[:, 0] = np.nan_to_num(score_sub[:, 0])

        #score_sub[:, 0] = (score_sub[:, 0] - score_sub[:, 0].mean())/ score_sub[:, 0].std()
        output[:,1] += score_sub[:, 1]
        output[:, 0] += score_sub[:, 0]#/score_sub[:,1]
        #score_sub[:, 0] = np.nan_to_num(score_sub[:, 0])

    output[:, 1] = np.where(output[:, 1] == 0, 1, output[:, 1])
    output_avg = output[:, 0]/output[:,1]

    output_avg = (output_avg - output_avg.mean())/ output_avg.std()
    return output_avg, contrast
"""