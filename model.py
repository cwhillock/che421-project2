import numpy as np
from scipy.integrate import solve_ivp
import scipy.optimize
import matplotlib.pyplot as plt
from pandas import DataFrame
R = 8.31446261815324E-3 #kj/mol/K

def calc_permeability(P0,Ep,T):
    return P0*np.exp(-Ep/R/T)

def calc_Vall(flows,permeabilities,P_high,P_low,thickness):
    scale = 1e8
    n = flows.size #number of components
    Ft = np.sum(flows) #total flowrates
    Vall = np.full_like(flows, 5) #initial guess, avoid divide by 0
    bnds = []
    for i in range(n):
        bnds.append((1e-12,None))
    bnds = tuple(bnds)

    def objective_function(Vall):
        temp = 0
        for i in range(n):
            temp += abs(permeabilities[i]*scale/thickness*(P_high*flows[i]/Ft-P_low*Vall[i]/sum(Vall))-Vall[i])
        return temp
    
    res = scipy.optimize.minimize(objective_function,Vall,tol=1e-9,method='Powell',bounds=bnds,options={'disp':False,'ftol':1e-9})
    #print(res)
    return res.x/scale

#initial conditions
#init_flows = np.array([0.13,0.11,0.76]) #CO2,O2,N2
init_flows = np.array([0.33,0.33,0.33])

P0_CO2 = 1.14E-3
Ep_CO2 = 41.5
P0_O2 = 3.12E-4
Ep_O2 = 41.4
P0_N2 = 3.99E-3
Ep_N2 = 52.8
T = 303.15
permeabilities = np.array([calc_permeability(P0_CO2,Ep_CO2,T),calc_permeability(P0_O2,Ep_O2,T),calc_permeability(P0_N2,Ep_N2,T)])
P_high = 280
P_low = 20
thickness = 2.5E-3

print(permeabilities)
print(calc_Vall(init_flows,permeabilities,P_high,P_low,thickness))

def solve_funcs(t,y):
    flows = y
    flows = np.maximum(flows,0)
    Vall = calc_Vall(flows,permeabilities,P_high,P_low,thickness)
    #print(flows,Vall)
    return -1*Vall

bouds = (0,0.5E6)
sol_array = solve_ivp(solve_funcs,t_span=bouds,y0=init_flows,max_step=5000)

for i in range(3):
    plt.plot(sol_array.t,sol_array.y[i])
plt.show()

"""
df = DataFrame({'Am':sol_array.t,'Flowrate CO2':sol_array.y[0],'Flowrate O2':sol_array.y[1],'Flowrate N2':sol_array.y[2]})
df.to_excel('results2.xlsx',sheet_name='sheet1',index=False)
"""
