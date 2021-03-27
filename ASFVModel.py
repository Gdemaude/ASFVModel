from matplotlib import pyplot
import numpy as np
from math import exp, log
from numba import jit
import multiprocessing
import psutil
import glob
import os
import time


intervention = 14 #set to None if no intervention
iter=164000 # Max number of iteration within one simulation
timestop=200 # max time a simulation can reach
subsampling=10 # arrays can get too big with big population, need to subsample it to use less data. Set to one for no subsampling
nb_simulation= 1000 # number of simulations done

data = np.zeros((6, iter), dtype=float)
data[:, 0] = [750498.0, 0.0, 20.0, 0.0, 0.0, 0.0]  # initialise data ->  s e i d c time

stoichiometry = np.array([
        [-1, 1, 0, 0, 0],# s e i d c   <- same column order than initial conditions and same line order as propensities
        [0, -1, 1, 0, 0],
        [0, 0, -1, 1, 0],
        [1, 0, 0, 0, 0],
        [0, 0, -1, 0, 1],
        [0, 0, 1, 0, -1],
        [-1, 0, 0, 0, 0],
        [0, -1, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, -1],
    ])

@jit(nopython=True, cache= True)
def propensity(i, d):
    # parameters
    mu = 0.002
    beta = 0.300
    gamma = 0.125
    rho = 0.7
    sigma = 0.250
    kappa = 0.06
    epsilon = 0.3
    betai = 0.05

    value=0.0
    if intervention == None:
        value = beta * d[0][i] * (d[2][i] + epsilon * d[4][i]) / max(d[0][i] + d[1][i] + d[2][i] + d[4][i], 1) # max to avoid division by 0 <- catch exception faster?
    elif  d[-1][i] < intervention:
        value=beta * d[0][i] * (d[2][i] + epsilon * d[4][i]) / max(d[0][i] + d[1][i] + d[2][i] + d[4][i], 1)
    else:
        value=(betai + (beta - betai) * exp(-(d[-1][i] - intervention)) )* d[0][i] * (d[2][i] + epsilon * d[4][i])/max(d[0][i] + d[1][i] + d[2][i] + d[4][i], 1)

    return np.array([
                     value,
                     sigma * d[1][i],
                     gamma * rho * d[2][i],
                     mu * (d[0][i] + d[1][i] + d[2][i] + d[4][i]),
                     gamma * (1 - rho) * d[2][i],
                     kappa * d[4][i],
                     mu * d[0][i],
                     mu * d[1][i],
                     mu * d[2][i],
                     mu * d[4][i]
                     ])

@jit(nopython=True, cache= True)
def gillespie_direct(data, stoichiometry, iter, timestop=0):
    for i in range(iter-1):
        if timestop > 0:
            if timestop <= data[-1, i]:
                return data[:, :i]
        propensities= propensity(i, data)
        partition = np.sum(propensities)
        if partition==0.0:
            return data[:,:i]

        r1=np.random.random()

        sojourn = log(
            1.0 / r1
        ) / partition

        data[-1,i+1]= data[-1, i]+sojourn
        indexes= np.argsort(propensities)
        partition= np.random.random()*partition
        for j in indexes:
            partition-=propensities[j]
            if partition<=0.0:
                data[:-1,i+1]=data[:-1,i]+stoichiometry[j]
                break
    return data

#using shared memory instead of writing files should be faster <- possible improvement
def monsousprocess( seed, nb_sim):
    np.random.seed(seed)
    path=str(seed)

    # instantiate the SSA container with model
    for i in range(nb_sim):
        res = gillespie_direct(data, stoichiometry, iter, timestop=timestop)
        res=res[:,::subsampling] #subsampling to use less memory
        np.save(path+str(i)+".npy", res) # data written down


if __name__ == '__main__':
    t1=time.time()
    ctx = multiprocessing.get_context('spawn')
    count = psutil.cpu_count(logical=False) - 1
    plist = [] # list of subprocesses
    for i in range(count):
        p0 = ctx.Process(target=monsousprocess, args=( i+1*1235, nb_simulation//count))
        p0.daemon = True # with this option, the subprocess dies when parent process dies
        p0.start()

        print("Subprocess started")
        plist.append(p0)

    pyplot.figure(figsize=(10,10))

    # make a subplot for the susceptible, infected , carrier and dead individuals
    axes_s = pyplot.subplot(411)
    axes_s.set_ylabel("susceptible individuals")

    axes_i = pyplot.subplot(412)
    axes_i.set_ylabel("infected individuals")

    axes_c = pyplot.subplot(413)
    axes_c.set_ylabel("carrier individuals")

    axes_d = pyplot.subplot(414)
    axes_d.set_ylabel("deaths due to ASF ")
    axes_d.set_xlabel("time (days)")

    if intervention:
        axes_s.axvline(x=intervention)
        axes_d.axvline(x=intervention)
        axes_i.axvline(x=intervention)
        axes_c.axvline(x=intervention)

    for p in plist:
        p.join()
    for file in glob.glob("*.npy"):
        res=np.load(file)
        axes_s.plot(res[-1], res[0], color="orange")
        axes_i.plot(res[-1], res[2], color="orange")
        axes_c.plot(res[-1], res[4], color="orange")
        axes_d.plot(res[-1], res[3], color="orange")
    t2 = time.time()
    print(t2-t1)
    pyplot.show()

    #comment this if you want to keep the data for future analysis
    for file in glob.glob("*.npy"):
        os.remove(file)
