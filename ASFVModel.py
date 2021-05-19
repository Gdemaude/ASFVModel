from matplotlib import pyplot
import numpy as np
from math import exp, log
from numba import jit
import multiprocessing
from multiprocessing import shared_memory
import psutil
import glob
import os
import time
from scipy.integrate import odeint


intervention = 14 #set to None if no intervention
iter=1640000 # Max number of iteration within one simulation
timestop=300 # max time a simulation can reach
subsampling=100# arrays can get too big with big population, need to subsample it to use less data. Set to one for no subsampling
nb_simulation= 101 # number of simulations done

data = np.zeros((6, iter), dtype=float)
data[:, 0] = [100000.0, 0.0, 20.0, 0.0, 0.0, 0.0]  # initialise data ->  s e i d c time
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
    mu = 0.0037
    beta = 0.5
    gamma = 0.15
    rho = 0.19
    sigma = 0.22
    kappa = 0.06
    epsilon = 0.3
    betai = 0.05
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

#using shared memory instead of writing files should be faster but overall RAM used would be greater
def monsousprocess( seed, nb_sim,l,mq,pq):
    np.random.seed(seed+1*12345)
    path=str(seed)
    existing_shm = shared_memory.SharedMemory(name=path)
    c = np.ndarray(data.shape, dtype=np.float, buffer=existing_shm.buf)

    for i in range(nb_sim):
        res = gillespie_direct(data, stoichiometry, iter, timestop=timestop)
        #res=res[:,::subsampling] #subsampling to use less memory
        pq.get()
        l.acquire()
        _, lim= res.shape
        #print(c[:,:lim].shape)
        c[:,:lim]=res[:]

        l.release()
        mq.put((seed,lim))

    time.sleep(1)
    mq.put((-1,-1))
    existing_shm.close()

        #np.save(path+str(i)+".npy", res) # data written down

def differential_ASFV(d, t):
    mu = 0.0037
    beta = 0.5
    gamma = 0.15
    rho = 0.19
    sigma = 0.22
    kappa = 0.06
    epsilon = 0.3
    betai = 0.05
    if intervention :
        if t >= intervention:
            beta = (betai + (beta - betai) * exp(-(t - intervention)))
    N= max(d[0] + d[1] + d[2] + d[4], 1)
    dS_dt =  -(beta * d[0] * (d[2] + epsilon * d[4])/N) + mu * N - mu *d[0]
    dE_dt = (beta * d[0] * (d[2] + epsilon * d[4]) / N) - ( sigma + mu) * d[1]
    dI_dt = sigma * d[1] +  kappa * d[4] - gamma * rho *d[2] - gamma * (1-rho) *d[2]- mu * d[2]
    dC_dt = gamma * (1-rho) * d[2] - (kappa +mu) *d[4]
    dD_dt = gamma * rho * d[2]
    return dS_dt, dE_dt, dI_dt, dD_dt, dC_dt


if __name__ == '__main__':
    t1=time.time()
    ctx = multiprocessing.get_context('spawn')
    count = psutil.cpu_count(logical=False) - 1 # count the number of physical CPU ( -1 is used to give some leeway for the processor
    print((nb_simulation//count)*count)
    stats= np.zeros((3,(nb_simulation//count)*count), dtype=float)

    plist = [] # list of subprocesses
    mlist=[]
    llist = []
    qlist=[]
    mainq = multiprocessing.Queue()
    for i in range(count):
        lock = multiprocessing.Lock()
        perq = multiprocessing.Queue()
        shm = shared_memory.SharedMemory(create=True, size=data.nbytes, name=str(i))
        p0 = ctx.Process(target=monsousprocess, args=( i, nb_simulation//count,lock,mainq,perq))
        p0.daemon = True # with this option, the subprocesses dies when parent process dies
        p0.start()
        perq.put("ok")#subprocess can modify

        print("Subprocess started")
        plist.append(p0)
        mlist.append(shm)
        llist.append(lock)
        qlist.append(perq)

    pyplot.figure(figsize=(10,10))

    # make a subplot for the susceptible, infected , carrier and dead individuals
    axes_s = pyplot.subplot(411)
    axes_s.set_ylabel("susceptible individuals")

    axes_i = pyplot.subplot(412)
    axes_i.set_ylabel("infected individuals")

    #axes_c = pyplot.subplot(413)
    #axes_c.set_ylabel("carrier individuals")

    axes_d = pyplot.subplot(413)
    axes_d.set_ylabel("deaths due to ASF ")
    axes_d.set_xlabel("time (days)")

    if intervention:
        axes_s.axvline(x=intervention)
        axes_d.axvline(x=intervention)
        axes_i.axvline(x=intervention)
        #axes_c.axvline(x=intervention)

    cc=0
    i=0
    while True:
        mess=mainq.get()
        if(mess[0]==-1):
            cc+=1
            if cc==count:
                break;
        else:
            llist[mess[0]].acquire()
            b = np.ndarray(data.shape, dtype=data.dtype, buffer=mlist[mess[0]].buf)
            res=b[:,:mess[1]]

            #print(res)
            ind = np.argmax(res[2])  # max infecte
            stats[0, i] = res[2, ind]
            stats[2, i] = res[-1, ind]  # jour pic
            ind = np.argmax(res[3])
            stats[1, i] = res[3, ind]  #
            res=res[:, ::subsampling]
            axes_s.plot(res[-1], res[0], color="orange")
            axes_i.plot(res[-1], res[2], color="orange")
            # axes_c.plot(res[-1], res[4], color="orange")
            axes_d.plot(res[-1], res[3], color="orange")
            llist[mess[0]].release()
            qlist[mess[0]].put("ok")
            i+=1

    for p in plist:
        p.join()  # we wait until subprocesses finish. They will join.

    for mm in mlist:
        mm.close()
        mm.unlink()

    t2 = time.time()
    print(t2-t1)
    y0 = data[:-1,0] #(7498.0,0.0,2.0,0.0, 0.0)
    t = np.linspace(0, timestop, num=300)

    solution = odeint(differential_ASFV, y0, t)
    solution = [[row[i] for row in solution] for i in range(5)]
    # plot numerical solution
    axes_s.plot(t, solution[0], color="black")
    axes_i.plot(t, solution[2], color="black")
    axes_d.plot(t, solution[3], color="black")
    #print(stats)
    print("Mean of: max infected,  max dead, pic day, ")
    
    print(np.mean(stats, axis=1))

    print("Max of: max infected, max dead, pic day, ")
    print(np.max(stats, axis=1))

    print("Min of: max infected,max dead, pic day ")
    print(np.min(stats, axis=1))

    fig1, ax1 = pyplot.subplots(1,3)
    ax1[0].set_title('Infected')
    ax1[0].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
    ax1[0].set_ylabel("Pigs")
    ax1[0].boxplot(stats[0], showfliers=False)

    ax1[1].set_title('Deaths')
    ax1[1].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
    ax1[1].set_ylabel("Pigs")
    ax1[1].boxplot(stats[1], showfliers=False)

    ax1[2].set_title('Peak Day')
    ax1[2].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
    ax1[2].set_ylabel("Days")
    ax1[2].boxplot(stats[2], showfliers=False)
    pyplot.show()
