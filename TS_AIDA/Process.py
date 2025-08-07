import numpy as np




def GBM(S0, r, sigma, T, n_samples, n_steps):

    dt = float(T) / n_steps
    rng = np.random.default_rng(seed=1)
    rand = rng.normal(loc=0, scale=1, size=[n_steps, n_samples])

    A = (r-0.5*sigma**2)*dt + sigma * np.sqrt(dt)*rand

    B = np.exp(A)

    C = S0*np.ones(shape=(1, n_samples))

    D = np.concatenate((C, B), axis=0)

    return np.cumprod(D, axis=0), D

def GBM_Jumps(paths, T, rate, mean, var):
    n_steps, n_samples = paths.shape

    dt = float(T) / n_steps
    rate_dt = rate*dt
    rng = np.random.default_rng()

    jump_index = rng.poisson(lam=rate_dt, size=paths.shape)
    jump_size = rng.normal(loc=mean, scale=np.sqrt(var), size=paths.shape)
    jumps = np.multiply(jump_index, jump_size)


    jump_paths = np.zeros_like(paths)
    jump_paths[0, :] = paths[0, :]

    for step in range(1, n_steps):
        jump_paths[step, :] = jump_paths[step-1,:]*paths[step,:] + paths[step-1,:]*jumps[step-1,:]



    x, y  = np.where(jump_index == 1)
    x, y = x[:, np.newaxis], y[:, np.newaxis]
    x = x-1
    z = np.take_along_axis(jump_paths, x, axis=0)

    q = np.take_along_axis(z, y, axis=1)
    jump_loc = np.concatenate((x, q), axis=1)

    return jump_loc, jump_paths

def GBM_jumps_change(path, T, mean, var, event_start, event_end):
    n_steps, n_samples = path.shape



    rng = np.random.default_rng()


    jump_size = rng.normal(loc=mean, scale=np.sqrt(var), size=path.shape)
    jumps = np.multiply(event_start, jump_size)
    jumps_end = np.multiply(event_end, 1/jump_size)
    jump_paths = np.zeros_like(path)
    jump_paths[0, :] = path[0, :]

    normal = True
    for step in range(1, n_steps):
        if normal == True:
            jump_paths[step, :] = jump_paths[step - 1, :] * path[step, :] + path[step - 1, :] * jumps[step - 1, :]
            if jump_index[step,:] == 1:
                jump_paths[step, :] = jump_paths[step - 1, :] * path[step, :] + path[step - 1, :] * jumps[step - 1, :]
                normal = False
        else:
            jump_paths[step, :] = jump_paths[step - 1, :] * path_var[step, :] + path_var[step - 1, :] * jumps[step - 1, :]
            if jump_index[step,:] == 1:
                normal = True

    x, y = np.where(jump_index == 1)
    x, y = x[:, np.newaxis], y[:, np.newaxis]
    x = x - 1
    z = np.take_along_axis(jump_paths, x, axis=0)

    q = np.take_along_axis(z, y, axis=1)
    jump_loc = np.concatenate((x, q), axis=1)

    return jump_loc, jump_paths

def OU(theta, sigma, mu, T, n_samples, n_steps, S0):
    dt = float(T) / n_steps
    rng = np.random.default_rng()
    rand = rng.normal(loc=0, scale=1, size=[n_steps, n_samples])
    Wt = np.sqrt(dt) * rand

    A = 1-theta*dt
    B = theta*mu*dt + sigma*Wt

    paths = np.zeros(shape=(n_steps, n_samples))
    paths[0,:] = S0
    for i in range(1,n_steps):
        paths[i,:] = paths[i-1,:]*A + B[i-1,:]

    return paths

def CIR(theta, sigma, mu, T, n_samples, n_steps, S0):
    dt = float(T) / n_steps
    rng = np.random.default_rng()
    rand = rng.normal(loc=0, scale=1, size=[n_steps, n_samples])
    Wt = np.sqrt(dt) * rand

    A = 1-theta*dt
    B = theta*mu*dt
    C = sigma*Wt

    paths = np.zeros(shape=(n_steps, n_samples))
    paths[0,:] = S0
    for i in range(1,n_steps):
        paths[i,:] = paths[i-1,:]*A + B + C[i-1,:]*np.sqrt(paths[i-1,:])

    return paths

def CIR_shift(theta, sigma, mu, T, n_samples, n_steps, S0):



    dt = float(T) / n_steps
    rng = np.random.default_rng()
    rand = rng.normal(loc=0, scale=1, size=[n_steps, n_samples])
    Wt = np.sqrt(dt) * rand

    A = 1-theta*dt
    B = theta*mu*dt
    C = sigma*Wt

    paths = np.zeros(shape=(n_steps, n_samples))
    paths[0,:] = S0
    for i in range(1, n_steps):
        paths[i,:] = paths[i-1,:]*A[i-1,:] + B[i-1,:] + C[i-1,:]*np.sqrt(paths[i-1,:])

    return paths

def Heston(v, mean, T, n_steps, S0):
    n_samples = v.shape[1]
    v = np.maximum(v, 0)
    dt = float(T) / n_steps
    rng = np.random.default_rng()
    rand = rng.normal(loc=0, scale=1, size=[n_steps, n_samples])
    Wt = np.sqrt(dt) * rand

    A = (mean-0.5*v)*dt
    B = np.sqrt(v*dt)*Wt

    paths = np.zeros(shape=(n_steps, n_samples))
    paths[0, :] = S0

    for i in range(1, n_steps):
        paths[i, :] = paths[i - 1, :] + A[i - 1, :] + B[i - 1, :]

    return paths

def Event_location(n_samples, n_steps, T, rate, dim_min_max, rebound=False, rebound_rate=5):
    rng = np.random.default_rng()

    dt = float(T) / n_steps
    rate_dt = rate * dt
    rebound_rate_dt  = rebound_rate*dt
    event_location = rng.poisson(lam=rate_dt, size=n_steps)
    event_index = np.where(event_location == 1)[0]
    n_events = np.sum(event_location)
    choice_dims = [i for i in range(dim_min_max[0], dim_min_max[1]+1)]
    n_dims = rng.choice(choice_dims, size=n_events)

    dims = [i for i in range(0, n_samples)]
    event_dim = [rng.choice(dims, replace=False, size=n_dims[i]) for i in range(n_events)]

    if rebound == True:
        ret = rng.poisson(lam=rebound_rate_dt, size=(n_steps))
        indices = np.where(ret == 1)[0]
        if len(indices) < n_events-2:
            print("Warning: rebound rate too small")
            indices = 2*np.ones(n_events)

        event_length = indices[1:-1]-indices[0:-2]


        rebound_index = event_index + event_length[0:event_index.shape[0]]
        rebound_index[np.where(rebound_index >= n_steps)[0]] = n_steps

    out = np.zeros(shape=(n_steps, n_samples))
    out_end = np.zeros(shape=(n_steps, n_samples))
    for i in range(n_events):
        dim = event_dim[i]
        event_start = event_index[i]
        out[event_start, dim] = 1
        if rebound == True:
            event_end = np.min((rebound_index[i], n_steps-1))
            out_end[event_end, dim] = 1




    return (event_index, rebound_index), event_dim

def Hawkes(rate, T, n_steps):
    rng = np.random.default_rng()
    dt = float(T) / n_steps
    rate_dt = rate * dt
    rate
    event_location = rng.poisson(lam=rate_dt, size=n_steps)[]
    for i in range(n_steps):
        if event_location[i] == 1:
            rate =1



    return