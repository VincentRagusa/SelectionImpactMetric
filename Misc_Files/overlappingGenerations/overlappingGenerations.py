from cmath import isclose
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, variance, stdev
from scipy.stats import binom
from collections import Counter
from itertools import chain
from matplotlib import animation
from matplotlib import rcParams
rcParams['animation.embed_limit'] = 1000 #MB
from IPython.display import HTML
from multiprocessing import Pool

# ------------------------------------------------------------------------------------------
POPSIZE = 500

GENOME_LENGTH = 5
MUTATION_RATE = 1/(2*GENOME_LENGTH) #per site rate
GENERATIONS = 50000
TSIZE = 3 #negative sets roulette exponent
# ------------------------------------------------------------------------------------------

DR = {}
ER = {}
def getDriftReference(initialPop,finalPop):
    key = (initialPop,finalPop)
    if key not in DR:
        DR[key] = [closedForm2(initialPop,x) for x in range(finalPop+1)]
    return DR[key]

def getEliteReference(initialPop,finalPop):
    key = (initialPop,finalPop)
    if key not in ER:
        ER[key] = [initialPop-1]+[0 for _ in range(finalPop-1)]+[1]
    return ER[key]


DEATH_LOG = []
class Organism:
    def __init__(self,startingFitness = None,timeBorn = 0):
        global GENOME_LENGTH
        self.genome = [random.randint(0,1) for _ in range(GENOME_LENGTH)]
        self.offspringCount = 0
        self.fitness = startingFitness
        self.timeBorn = timeBorn
        self.timeDied = None
        
    def make_mutated_copy(self,gen=-1):
        global GENOME_LENGTH, MUTATION_RATE
        self.offspringCount += 1
        child = Organism(timeBorn = gen)
        child.genome = copy.deepcopy(self.genome)
        for i in range(GENOME_LENGTH):
            if random.random() <= MUTATION_RATE:
                PN = random.randint(0,1)
                child.genome[i] += (-1*PN)+(1-PN)
        return child
    
    def kill(self,gen):
        self.timeDied = gen
        DEATH_LOG.append(self)
    
    def __repr__(self,):
        return str(self.fitness)


def fitness(org):
    if not org.fitness:
        # org.fitness = 1
        org.fitness = eval_sawTooth(sum(org.genome))
#         org.fitness = max( sum(org.genome), 0)
    return org.fitness


def eval_sawTooth(x):
#     x = org.genomeValue #0 is top of first peak
    w = 5 #valley width
    d = 5 #valley depth
    r = 10 #fitness rise peak to peak
    x = x + w+1 #offset to next peak to avoid fitness zero on init
    return x*(-d/w) + (x//(w+1))*(r + d + (d/w)) 



def roulette_wheel(population,exponent=1.05):
    MAX = max([fitness(x) for x in population])
    F = [np.power(exponent,fitness(x)-MAX) for x in population]
    S = sum(F)
    return [f/S for f in F]




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


def newPopulation(population,gen,tournament = 0):
    global POPSIZE

    # STANDARD
    # if tournament < 0:
    #     #do roulette
    #     wheel = roulette_wheel(population,exponent=-tournament)
    #     child = random.choices(population,k=1,weights=wheel)[0].make_mutated_copy(gen)
    # if tournament >= 1:
    #     # do tournament
    #     child = max(random.choices(population,k=tournament),key = lambda org: fitness(org)).make_mutated_copy(gen)
    # death = random.randint(0,POPSIZE-1)
    # population[death].kill(gen)
    # population[death] = child

    # FIT PROP DEATH, RAND BIRTH
    if tournament < 0:
        #do roulette
        wheel = roulette_wheel(population,exponent=-tournament)
        death = random.choices(list(range(POPSIZE)),k=1,weights=[1-w for w in wheel])[0]
    if tournament >= 1:
        # do tournament
        death = min(random.choices(list(range(POPSIZE)),k=tournament),key = lambda index: fitness(population[index]))
    child = population[random.randint(0,POPSIZE-1)].make_mutated_copy(gen)
    population[death].kill(gen)
    population[death] = child

    return population
    

def run():
    global POPSIZE,GENERATIONS,TSIZE
    
    population = [Organism() for _ in range(POPSIZE)]
    fitlog = [mean([fitness(org) for org in population])]
    # popLog = [population]
    for generation in range(GENERATIONS):
        print(100*generation/GENERATIONS,end="\r")

        population = newPopulation(population,generation,tournament=TSIZE)
        fitlog.append(mean([fitness(org) for org in population]))
        # popLog.append(population)
    print()
    return population, fitlog


random.seed(1337)

population, fitlog = run()

plt.plot(fitlog)
plt.show()

print(len(population))
print(len(DEATH_LOG))

print("pop",mean([org.offspringCount for org in population]))
print("death log",mean([org.offspringCount for org in DEATH_LOG]))

plt.hist([org.offspringCount for org in population],label="pop",bins=[0,1,2,3,4,5,6,7,8],alpha=0.5,density=True)
plt.hist([org.offspringCount for org in DEATH_LOG],label="death pool",bins=[0,1,2,3,4,5,6,7,8],alpha=0.5,density=True)
plt.legend()
plt.show()

plt.scatter([org.offspringCount for org in DEATH_LOG],[org.timeDied-org.timeBorn for org in DEATH_LOG])
plt.axhline(mean([org.timeDied-org.timeBorn for org in DEATH_LOG]),color="r")
plt.axvline(mean([org.offspringCount for org in DEATH_LOG]),color="r")
# plt.axhline(POPSIZE,color="g")
# plt.axvline(1,color="g")
plt.xlabel("off counts")
plt.ylabel("lifetime")
plt.show()

plt.scatter([org.offspringCount for org in DEATH_LOG],[fitness(org) for org in DEATH_LOG])
plt.axhline(mean([fitness(org) for org in DEATH_LOG]),color="r")
plt.axvline(mean([org.offspringCount for org in DEATH_LOG]),color="r")
# plt.axhline(POPSIZE,color="g")
# plt.axvline(1,color="g")
plt.xlabel("off counts")
plt.ylabel("fitness")
plt.show()

plt.scatter([org.offspringCount for org in population ],[GENERATIONS-org.timeBorn for org in population])
plt.axhline(mean([GENERATIONS-org.timeBorn for org in population]),color="r")
plt.axvline(mean([org.offspringCount for org in population]),color="r")
# plt.axhline(POPSIZE,color="g")
# plt.axvline(1,color="g")
plt.xlabel("off counts")
plt.ylabel("lifetime")
plt.show()

plt.scatter([org.offspringCount for org in population ],[fitness(org) for org in population])
plt.axhline(mean([fitness(org) for org in population]),color="r")
plt.axvline(mean([org.offspringCount for org in population]),color="r")
# plt.axhline(POPSIZE,color="g")
# plt.axvline(1,color="g")
plt.xlabel("off counts")
plt.ylabel("fitness")
plt.show()

# fitnessLog = [org.fitness for org in population if org.timeDied == None]

# with Pool(16) as MPPOOL:
#     SSdata = MPPOOL.starmap(getSS,zip(offCounts,initialPopSizes,finalPopSizes))
#     AVEdata = MPPOOL.map(mean,fitnessLog)
#     VARdata = MPPOOL.map(variance,fitnessLog)

# driftDists = list(zip(*SSdata))[0]
# eliteDists = list(zip(*SSdata))[1]

# plt.plot(AVEdata)
# plt.show()

def geometric(p,n):
    return (1-p)**(n-1) * p

def binomial(p,n,N):
    choose = np.math.factorial(N)/(np.math.factorial(n)*np.math.factorial(N-n))
    return choose * (p)**n * (1-p)**(N-n)

def moranDriftTheory(n,N,apprx = 1000):
    # acc = 0
    # for dn in range(apprx):
    #     geom = geometric(1/N,n+dn)
    #     binom = binomial(1/N,n,n+dn)
    #     acc += geom*binom
    # return acc

    # acc = 0
    # for d in range(apprx):
    #     choose = np.divide(np.math.factorial(n+d),(np.math.factorial(n)*np.math.factorial(d)))
    #     acc += choose * np.power(1-(1/N), n +2*d -1)
    # return acc * np.power((1/N),n+1)

    # acc = 0
    # a = int(abs(n-0.5)+0.5)
    # for d in range(apprx):
    #     choose = np.divide(np.math.factorial(a+d),(np.math.factorial(n)*np.math.factorial(a+d-n)))
    #     acc += choose * np.power(1-(1/N), 2*(a+d)-n -1)
    # return acc * np.power((1/N),n+1)

    acc = 0
    a = int(abs(n-0.5)+0.5)
    for d in range(apprx):
        choose = np.divide(np.math.factorial(a+d),(np.math.factorial(n)*np.math.factorial(a+d-n)))
        acc += choose * np.power(1-(1/N), 2*(a+d))
    return acc * np.power((1/(N-1)),n+1)

def closedForm1(N,n):
    try:
        assert (1/N != 2.0 or n < -1) and abs(((N-1)**2) / (n**2)) < 1
        p = 1/N
        q = 1.0-p
        r = (2*N-1)/(N**2)
        return (q**(n-1) * p**(n-1) * r**(-n))/(2*N-1)
    except:
        return None

def closedForm2(N,n):
    assert N > 1/2
    # return ((N**2) * (((N-1)/(N**2))**n) * (((2*N-1)/(N**2))**(-n)))/((N-1)*(2*N-1))
    # return (np.power(N,2) * np.power((N-1)/np.power(N,2),n) * np.power((2*N-1)/np.power(N,2),-n))/((N-1)*(2*N-1))
    return (N**2) * (N-1)**(n-1) / (2*N-1)**(n+1) if n>0 else (N-1)/(2*N-1) #CORRECTED WITH SPECIAL CASE FOR ZERO

# plt.hist([org.offspringCount for org in population],label="pop",bins=[0,1,2,3,4,5,6,7,8],alpha=0.5,density=True)
plt.hist([org.offspringCount for org in DEATH_LOG],label="Observation (Death Pool)",bins=[i-0.5 for i in range(10)],alpha=0.5,density=True)
X = [i for i in range(10)]

# for apx in [i*200 for i in range(20)]:
apx = 20
Y = [moranDriftTheory(x,POPSIZE,apprx=apx) for x in X]
print(Y)
print(sum(Y))
if np.isclose(1,sum(Y)): print("approx",apx,"is convergant on 1")
plt.plot(X,Y,label="Theory (drift) apx:{}".format(apx),marker="o")


# Yc1 = [closedForm1(POPSIZE,x) for x in X]
Yc2 = [closedForm2(POPSIZE,x) for x in X]
# print(Yc1)
print(Yc2)
print(sum(Yc2))
# plt.plot(X,Yc1,label="Theory (drift) cf 1",marker="o")
plt.plot(X,Yc2,label="Theory (drift) closed form",marker="o")

plt.legend()
plt.yscale("log")
plt.show()

for i in range(100):
    print(i,sum([closedForm2(POPSIZE,j) for j in range(i)]))

ssLog = []
resolution = POPSIZE//2
for i in range(GENERATIONS//resolution):
    genDL = DEATH_LOG[i*resolution:(i+1)*resolution]
    dr,el = getSS([org.offspringCount for org in genDL],resolution,resolution)
    ssLog.append(dr)
plt.plot(ssLog)
plt.show()