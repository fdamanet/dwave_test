# Doc on how to control annealing parameters: https://docs.dwavesys.com/docs/latest/c_qpu_annealing.html

import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import pickle

n = 2
dimHS = 2**n
AllConfig = np.array([i for i in product([-1,1], repeat=n)])
print(AllConfig)

# set a simple Ising problem
# set h_i & J_ij
h = {}
J = {}
h[0] = 0.0
h[1] = 0.04     
J[0,1] = 0.02

Energy = np.array([sum(h[j]*AllConfig[i,j] + sum(J[j,jp]*AllConfig[i,j]*AllConfig[i,jp] for jp in range(j+1,n)) for j in range(n)) for i in range(len(AllConfig))]) 

# Finding ground states indices
GroundStateIndices = np.array(np.where(Energy == Energy.min())[0])
GroundStateEnergy = np.array([Energy.min() for u in range(len(GroundStateIndices))])
print('Optimal spin configurations: ')
print([AllConfig[GroundStateIndices[i]] for i in range(len(GroundStateIndices))])
print('Optimal spin configuration indices: ')
print(GroundStateIndices)

# Plot of the energy landscape
plt.figure(0)
axes = plt.gca()
#plt.gca().set_aspect(0.1)
plt.plot(Energy, marker='o', linewidth=1.0, label="Energy landscape")
plt.fill_between(range(dimHS), Energy,Energy.min(), color='#539ecd', alpha = 0.25)
plt.plot(GroundStateIndices, GroundStateEnergy, marker='o', linestyle='', c='r',label='Ground state configuration indices')
plt.xlabel('Spin configuration indices')
plt.ylabel('Energy')
plt.grid()
plt.savefig('EnergyLandscape_controlling_annealing_n='+str(n)+'_h1='+str(h[1])+'.png')

# set Dwave parameters
ta = 2.5     # annealing time
nreads = 5000   # number of experiments (jobs) and measurements
tstop = ta

#-------------------------------------------------------------------------------------------------------------
#   Dwave calculations
#-------------------------------------------------------------------------------------------------------------

#my_schedule = [[0.0, 0.0], [tstop, s], [ta, 1.0]]
my_schedule = [[0.0, 0.0], [ta, 1.0]]
print(my_schedule)
#my_schedule = np.array(my_schedule)
#print(np.array([(my_schedule[i+1,1]-my_schedule[i,1])/(my_schedule[i+1,0]-my_schedule[i,0]) for i in range(2)]))


from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

# use DWaveSampler
dw = DWaveSampler()
# embed to Chimera graph
sampler = EmbeddingComposite(dw)
# in the case of ising model, use the following
my_anneal_params = dict(anneal_schedule=my_schedule, auto_scale=False)

response = sampler.sample_ising(h, J, num_reads=nreads, **my_anneal_params)
print(response)

# save data
fileObj = open('data_controlling_annealing_ta='+str(ta)+'_stop='+str(tstop)+'_h1='+str(h[1])+'.obj', 'wb')
pickle.dump(response,fileObj)
fileObj.close()

#-------------------------------------------------------------------------------------------------------------
#   Import data and plots
#-------------------------------------------------------------------------------------------------------------

# import data 
fileObj = open('data_controlling_annealing_ta='+str(ta)+'_stop='+str(tstop)+'_h1='+str(h[1])+'.obj', 'rb')
response = pickle.load(fileObj)
fileObj.close()
print(response.record)

# function that calculates the energy (model energy) of a steadystate spin configuration found by Dwave
def calculate_energy(solution, vartype='BINARY'):
    if vartype == 'SPIN':
        ene = sum(h[i]*solution[i] for i in range(n))
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
    print(energy)
    # count up the times when the energy is zero
    if energy == Energy.min():
        num_optimal_sol += num_oc
        optimal_sol.append(solution)
    # store the spin configuration found by Dwave
    sol.append(solution)
print('Percentage of optimal configurations among all configurations found by Dwave = '+str(round(num_optimal_sol/nreads*100,2))+'%')

# Reordering of the spin configuration occurences according to the indexation
orderedstates = []
orderedenergy = []
count = 0
for state in response.record:
    for j in range(len(AllConfig)):
        if np.linalg.norm(state[0]-AllConfig[j]) == 0:
            orderedstates.append([j, state[2]])
            orderedenergy.append([j, state[1]])
    count = count+1
orderedstates = np.array(orderedstates)
orderedenergy = np.array(orderedenergy)
orderedstates=orderedstates[[np.lexsort(orderedstates.T[::-1])]][0]
orderedenergy=orderedenergy[[np.lexsort(orderedenergy.T[::-1])]][0]
print(orderedstates)
print(orderedenergy)

# plot steady states spin configuration occurences on the energy landscape
plt.figure(2)
axes = plt.gca()
#plt.gca().set_aspect(0.004)
axes.bar(orderedstates[:,0],orderedstates[:,1], color='orange', alpha=0.75, label="Dwave spin configuration occurences (reads="+str(nreads)+"; annealing time="+str(ta)+"µs)")
plt.xlabel('Spin configuration indices')
plt.ylabel('Energy')
plt.grid()
#axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=1, fancybox=True, shadow=True)
plt.xlim([-0.5,3.5])
plt.ylim([-1,5000])
plt.savefig('Histo_controlling_annealing_ta='+str(ta)+'_stop='+str(tstop)+'_h1='+str(h[1])+'.png', bbox_inches='tight')
