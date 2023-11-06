import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import pickle

#-------------------------------------------------------------------------------------------------------------
#   Setting the partitioning problem
#-------------------------------------------------------------------------------------------------------------
n = 6
dimHS = 2**n
ntab = np.array([1,5,10,1,4,3])
AllConfig = np.array([i for i in product([-1,1], repeat=n)])
Energy = np.array([sum(AllConfig[i,j]*ntab[j] for j in range(n))**2 for i in range(len(AllConfig))])

# Finding ground states indices
GroundStateIndices = np.array(np.where(Energy == Energy.min())[0])
GroundStateEnergy = np.array([Energy.min() for u in range(len(GroundStateIndices))])
print('Optimal spin configurations: ')
print([AllConfig[GroundStateIndices[i]] for i in range(len(GroundStateIndices))])
print('Optimal spin configuration indices: ')
print(GroundStateIndices)

# Plot of the energy landscape
fig, axs = plt.subplots(2,gridspec_kw={'height_ratios': [2,1]})
fig.suptitle('Energy landscape')
axs[0].set_ylabel('Energy')
axs[0].plot(Energy, marker='o', linewidth=1.0)
axs[0].fill_between(range(dimHS),Energy, color='#539ecd', alpha = 0.25)
axs[0].plot(GroundStateIndices, GroundStateEnergy, marker='o', linestyle='', c='r')
axs[0].set_xticks([0,2**(n-1)-1, 2**n-1])
axs[0].grid()
axs[1].set_xlabel('Spin configuration indices')
axs[1].set_ylabel('Energy')
axs[1].plot(Energy, marker='o', linewidth=1.0)
axs[1].fill_between(range(dimHS),Energy, color='#539ecd', alpha = 0.25)
axs[1].plot(GroundStateIndices, GroundStateEnergy, marker='o', linestyle='', c='r')
axs[1].set_ylim(-4,20)
axs[1].set_xticks([0,2**(n-1)-1, 2**n-1])
axs[1].grid()
plt.savefig('EnergyLandscape_PartitioningProblem_n='+str(n)+'.png')

# set the partitioning problem as an Ising problem
# set h_i & J_ij
h = {}
J = {}
for i in range(n):
    h[i] = 0
    for j in range(i+1,n):
        J[i, j] =  2*ntab[i] * ntab[j]
        
# set Dwave parameters
ta = 20         # annealing time
nreads = 1000   # number of experiments (jobs) and measurements
sample = 1      # job's index

#-------------------------------------------------------------------------------------------------------------
#   Dwave calculations
#-------------------------------------------------------------------------------------------------------------

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
# use DWaveSampler
dw = DWaveSampler()
# embed to Chimera graph
sampler = EmbeddingComposite(dw)
# in the case of ising model, use the following
response = sampler.sample_ising(h, J, num_reads=nreads,annealing_time=ta, label='Partitioning_ta='+str(ta))

print(response)
print(response.info)

# save data
fileObj = open('data_partitioning_ta='+str(ta)+'_sample='+str(sample)+'.obj', 'wb')
pickle.dump(response,fileObj)
fileObj.close()


#-------------------------------------------------------------------------------------------------------------
#   Import data and plots
#-------------------------------------------------------------------------------------------------------------

# import data 
fileObj = open('data_partitioning_ta='+str(ta)+'_sample='+str(sample)+'.obj', 'rb')
response = pickle.load(fileObj)
fileObj.close()

# function that calculates the energy (model energy) of a steadystate spin configuration found by Dwave
def calculate_energy(solution, vartype='BINARY'):
    if vartype == 'SPIN':
        ene = sum(ntab[i] ** 2 for i in range(n))
        for i in range(n):
            for j in range(i + 1, n):
                ene += J[i, j] * solution[i] * solution[j]
    else:
        raise ValueError("vartype mast be 'BINARY' or 'SPIN'.")
    return ene

# count the number of optimal solutions
num_optimal_sol = 0
optimal_sol = []
sol = []
for state in response.record:
    # 0th contains the spin configuration,
    # 1st contains the energy of the configuration (hardware energies),
    # 2nd contains the number of occurrences of the spin configuration in .record
    solution = state[0]
    num_oc = state[2]
    # compute energy
    energy = calculate_energy(solution, vartype='SPIN')
    # count up the times when the energy is zero
    if energy == Energy.min():
        num_optimal_sol += num_oc
        optimal_sol.append(solution)
    # store the spin configuration found by Dwave
    sol.append(solution)
print('Percentage of optimal configurations among all configurations found by Dwave = '+str(round(num_optimal_sol/nreads*100,2))+'%')

# Reordering of the spin configuration occurences according to the indexation
orderedstates = []
count = 0
for state in response.record:
    for j in range(len(AllConfig)):
        if np.linalg.norm(state[0]-AllConfig[j]) == 0:
            orderedstates.append([j, state[2]])
    count = count+1
orderedstates = np.array(orderedstates)

# plot steady states spin configuration occurences on the energy landscape
plt.figure(0)
axes = plt.gca()
plt.gca().set_aspect(0.1)
plt.plot(Energy, marker='o', linewidth=1.0, label="Energy landscape")
plt.fill_between(range(dimHS), Energy, color='#539ecd', alpha = 0.25)
plt.plot(GroundStateIndices, GroundStateEnergy, marker='o', linestyle='', c='r',label='Ground state configuration indices')
axes.bar(orderedstates[:,0],orderedstates[:,1], color='orange', alpha=0.75, label="Dwave spin configuration occurences (reads="+str(nreads)+"; annealing time="+str(ta)+"µs)")
plt.xlabel('Spin configuration indices')
plt.ylabel('Energy')
plt.grid()
axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=1, fancybox=True, shadow=True)
plt.xlim([5,2**(n-1)-1])
plt.ylim([-2.5,150])
plt.savefig('Histo_partitioning_n='+str(n)+'_ta='+str(ta)+'.png', bbox_inches='tight')
