import random
import copy
from scipy.stats import binom
from collections import Counter
import matplotlib.pyplot as plt
from statistics import mean
# ------------------------------------------------------------------------------------------

POP_SIZE = 100
GENOME_LENGTH = 10
MUTATION_RATE = 0.1 #per site rate
GENERATIONS = 1000
# ------------------------------------------------------------------------------------------
DR = {}
ER = {}
def getDriftReference(initialPop,finalPop):
    key = (initialPop,finalPop)
    if key not in DR:
        DR[key] = [binom.pmf(x, finalPop, 1/initialPop) for x in range(finalPop+1)]
    return DR[key]

def getEliteReference(initialPop,finalPop):
    key = (initialPop,finalPop)
    if key not in ER:
        ER[key] = [initialPop-1]+[0 for _ in range(finalPop-1)]+[1]
    return ER[key]

def emd(P,Q):
    assert len(P) == len(Q)
    EMD = [0]
    for i in range(len(P)):
        EMD.append(P[i]-Q[i]+EMD[-1])
    return sum([abs(d) for d in EMD])

def getSS(offCounts_1d,initialPop,finalPop):
    Counts = Counter(offCounts_1d)
#     largest = sorted(Counts.items())[-1][0]
    Observed = [Counts[x]/initialPop if x in Counts else 0 for x in range(finalPop+1)]
    return (emd(getDriftReference(initialPop,finalPop),Observed),
            emd(getEliteReference(initialPop,finalPop),Observed))

class Org:
    def __init__(self):
        global GENOME_LENGTH
        self.genome = [0 for _ in range(GENOME_LENGTH)]
        self.fitness = -1 #assume fitness >= 0
        self.offspring = 0

    def __repr__(self):
        return str(self.fitness)


    def make_mutated_copy(self):
        global GENOME_LENGTH,MUTATION_RATE
        self.offspring += 1
        child = Org()
        child.genome = copy.deepcopy(self.genome)
        for i in range(GENOME_LENGTH):
            if random.random() <= MUTATION_RATE:
                PN = random.randint(0,1)
                child.genome[i] += (-1*PN)+(1-PN) #if PN = 0, 1 ; if PN=1 , -1
        return child


    def get_fitness(self):
        if self.fitness == -1: #assume fitness >= 0
            self.fitness = eval_sawTooth(sum(self.genome))
        return self.fitness

def eval_sawTooth(x):
#     x = org.genomeValue #0 is top of first peak
    #a value of x=a+1 correspoinds to one mutation right of a
    w = 6 #valley width
    d = 0.9 #valley depth
    r = 4 #fitness rise peak to peak
    x = x + w+1 #offset to next peak to avoid fitness zero on init
    return x*(-d/w) + (x//(w+1))*(r + d + (d/w)) 


def tournament_select(population,size=2):
    return max(random.choices(population, k=size),key= lambda org: org.get_fitness())
# ------------------------------------------------------------------------------------------

population = [Org() for _ in range(POP_SIZE)]

fitnessLog = []
SSlog_all = []
SSlog_mut = []
lastMutCount = POP_SIZE

for generation in range(GENERATIONS):
    print("Generation:", generation,"\t MAX:", max(population, key= lambda org: org.get_fitness()))
    fitnessLog.append(mean([org.get_fitness() for org in population]))
    children = [tournament_select(population).make_mutated_copy() for _ in range(POP_SIZE)]
    SSlog_all.append(getSS([org.offspring for org in population],POP_SIZE,POP_SIZE)[0])
    maxInt = max([int(org.get_fitness()) for org in population])
    mutOffCounts = [org.offspring for org in population if int(org.get_fitness()) == maxInt]
    len_mutOffCounts = len(mutOffCounts)
    SSlog_mut.append(getSS(mutOffCounts,lastMutCount,len_mutOffCounts)[0])
    lastMutCount = len_mutOffCounts if int(len_mutOffCounts) else 1
    population = children

aveOfMutSel = [mean(SSlog_mut[max(0,i-1):min(i+0+1,GENERATIONS)]) for i in range(GENERATIONS-1)]

f,ax = plt.subplots(1,2)
ax[0].plot(range(GENERATIONS),fitnessLog)
ax[1].plot(range(GENERATIONS),SSlog_all,label="all")
ax[1].plot(range(GENERATIONS),SSlog_mut,label = "mut")
ax[1].plot(range(GENERATIONS-1),aveOfMutSel)
ax[1].set_yscale("symlog")
ax[1].legend()
plt.show()

F_ROC = [fitnessLog[i+1]-fitnessLog[i] for i in range(GENERATIONS-1)]
plt.scatter(F_ROC,aveOfMutSel)
plt.axvline(0)
plt.show()