#theory
import matplotlib.pyplot as plt
import numpy as np


#normal distribution: returns P(x|mu,sigma) (does not generate random numbers)
def norm(x,mu,sigma):
    return (1/np.sqrt(2*np.pi*sigma**2)) * np.exp( -((x-mu)**2 / (2*sigma**2)) )


#given a probability distribution Q, defined over support Qx, with probabilities Qy,
# return a new probability distribution.
#note the update probability distribution is norm(x,center,1). this presents a twofold problem:
#first, there is no time dependance, so regardless of observation window size, the update is the same
#second, although assuming gaussian mutation is typical, it is not general.
def Qd(x,Qx,Qy,dt):
    return sum([Qy[center]*norm(x,center,dt)*dx for center in Qx])

r = 10 #defines the radius from zero of the support (support width)
dx = .1 #defines the resolution of the "integral" (this controls the resolution of the support too)
dt = .1 #defines the time between each observation

Qx = list(np.linspace(-r,r,int(1/dx)*2*r+1)) #establish the supoprt
Qy = {x: 0 if x not in [-1,1] else 1/2 for x in Qx} #begin with all probability at zero (this is where the real prior will go)

# print(sum(Qy.values()))
plt.plot(Qy.keys(),Qy.values())
# plt.show()
for update in range(200):
    Qy = {x:Qd(x,Qx,Qy,dt) for x in Qx}
    s = sum(Qy.values())
    if not np.isclose(s,1):
        print("WARNING: probability distribution does not sum to 1!")
        print(s,"!=",1)
        if s < 1:
            print("Is your support width too small?")
        if s > 1:
            print("Is dx too course?")
        print()
    plt.plot(Qy.keys(),Qy.values())
plt.show()