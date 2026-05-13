#Solving Aiygari model using Value Function Iteration and Endogenous Grid Points
#Elyse St Sauver
#Adapted from code made available for MATLAB by Oliko Vardishvili, https://github.com/ovardish/2022_Computational_Course_Code


#Assumes CRRA utility preferences over consumption, inelastic labor, and a Cobb-Douglas production function.
#Computes the equilibrium consumption policy, asset policy and stationary distribution in the Aiyagari model using discrete Value Function Iteration

#1. Define the aiyagari household problem, and solve for optimal assets next period given an interest rate r using either VFI or EGP.
#2. Plot the equilibrium policy functions and stationary distribution.


#Step 0:
import numpy as np
from scipy.interpolate import interp1d #for interpolation in EGP
import quantecon as qe #for simulating AR(1) as Markov process
import matplotlib.pyplot as plt

#Define initial parameters
#Calibrated model parameters (US unless otherwise specified)
gamma = 3 #risk aversion parameter
beta = 0.96 #discount factor
delta = 0.08 #depreciation
A = 1 #aggregate productivity of production Y = A*F(K,L)
alpha = 0.36 #capital share of income
b = -2 #borrowing limit, exogenously given
B = 30 #upper bound to assets, should not bind
incgridk = 0.09 #asset grid step size

#Algorithm parameters
tol_r = 0.0001 #tolerance for convergence of interest rate
tol_vf = 0.0001 #tolerance for convergence of value function
tiny = 0.1**10 #arbitrary epsilon
minrate = -delta #Initialize min interest rate for bisection
maxrate = (1-beta) / beta #Initialize max interest rate for bisection
err_r = 1 #Inital value for interest rate convergence error
err_v = 10 #Initial value for tracking value function convergence error
err_lam = 1 #Initial value for tracking lambda convergence error in VFI
err_lam_egm = 1 #Initial value for tracking lambda convergence error in EGM
    
#Income productivity process is AR(1), approximate with Markov Chain with 5 states
#z_t = lambda * z_(t-1) + u_t s.t. u_t ~ N(0, sigma)
nz = 7 #number of productivity states
rho = 0.2 #first order autoregressive coefficient of prod states
sigmaLR = 0.4 #long run stdev of income
sigma = sigmaLR*np.sqrt(1-rho**2) #stdev of error_t

#Using Rouwenhorst approximation 
#Output markov chain that contains transition matrix P and state values
prod_mc = qe.markov.approximation.rouwenhorst(nz, rho, sigma)

P = prod_mc.P #transition matrix P
log_prod_states = prod_mc.state_values #discretized states of productivity produced by the AR(1)
invariant_dist = prod_mc.stationary_distributions[0]  #invariant distribution of the Markov Chain

#Exponeniate the logged productivity states
z = np.exp(log_prod_states)
#Calculate average labor efficiency
# L = productivity of state * proportion agents in that state * labor population (normalized to 1)
labor = z*invariant_dist

#Step 1A:
#Now that we know the productivity shock process, can iterate to find equilibrium.
#Will compute solutions by converging to optimal r.
#For each guess at r, we will solve the aiyagari household problem using value function iteration, solve_hh_vfid(r).
#Then we will iterate by bisection using this function until r1 - r0 < tol_r

#A: Function that solves the Aiyagari household problem for a given r using discrete value function iteration. Returns the mean capital demand.
def solve_hh_vfid(r):
    #Initialize error vars
    err_v = 10
    err_lam = 1
    #Compute wage given r. Cobb-Douglas production function gives w depending on just r, delta, alpha and A
    wage = (1-alpha)*(A*(alpha/(r+delta))**alpha)**(1/(1-alpha))

    #Set borrowing limit to either lower limit b or the value of income in the worst productivity state, whichever is lower
    if r <= 0:
        phi = b
    else:
        phi = min(b, wage*z[0]/r)
            
    #Define the discretized grid of capital assets
    maxgrid = 25
    mingrid = phi
    incgridk = 0.09
    gridk = np.arange(mingrid, maxgrid + incgridk, incgridk)
    ngridk = len(gridk)

    #Initialize the value function and utility matrices
    v = np.zeros((nz, ngridk))
    utilm = np.zeros((nz, ngridk, ngridk)) #utility for each state z, assets k, and choice of next period assets k+t1
    tv = np.zeros((nz, ngridk)) #This will take the max utility of the computed utility grid, to compare with last computed v.
    tdecis = np.zeros((nz, ngridk), dtype=int) #Takes actual argument (k') that produced max utility from grid

    for j in range(nz):
        for i in range(ngridk):
            #compute vector of consumption for all possible choices for k'
            c = z[j]*wage + (1+r)*gridk[i] - gridk
            #compute utility given c
            utilm[j,i,:] = (np.maximum(tiny, c)**(1-gamma))/(1-gamma)

    #Now that we have utility, can compute value function. Compute value function for each u, stop when converges.
    while err_v > tol_vf:
        for j in range(nz):
            for i in range(ngridk):
                u = utilm[j,i,:]
                vint = u + beta*P[j,:]@v
                #Pick highest value for this state and asset level, save as tv
                tv[j, i] = np.max(vint)
                #Save the corresponding k'
                tdecis[j, i] = np.argmax(vint)
        err_v = max(abs(tv-v).flatten())
        v = tv.copy()

    #Now we have the utility-maximizing asset choice for any state and value of current assets, and the corresponding value function.
    #Next we can calculate the stationary distribution and get aggregate k, the capital supply given our initial input r.
    
    #Initialize grid for the stationary distribution, normalize so that the grid sums to one.
    lam = np.ones((nz, ngridk))/(nz*ngridk)
    lam_new = np.zeros((nz, ngridk))

    #Now loop until the distribution is stationary. 
    while err_lam > tiny:
        for j in range(nz):
            for i in range(ngridk):
                for j_p in range(nz):
                    #Compute the new distribution by taking the current distribution and multiplying by the MC transition matrix
                    lam_new[j_p, tdecis[j,i]] += lam[j, i] * P[j,j_p]
        #Compute difference between old and new distribution
        err_lam = max(abs(lam_new-lam).flatten())
        lam = lam_new
        lam_new = np.zeros((nz, ngridk))
    #Once lambda convergences to the stationary distribition, compute the average assets (capital) supplied, and return it as meank
    kk = gridk[tdecis]
    c = z[:, None]*wage + (1+r)*gridk[None, :] - kk
    meank = np.sum(kk * lam)
    return c, kk, gridk, lam, meank

print("Starting VFI")
#Now we know how to compute the average capital supplied for a given r. 
#Using bisection, compute r from k supplied and from k demanded, and stop when those two rs are equal (within tolerance). This is the equilibrium r!
while abs(err_r) > tol_r:
    #initial guess for r
    r0 = 0.5*(maxrate + minrate)
    #Compute asset demand given r0
    k0 = ((r0 + delta) / (alpha*A*np.sum(labor)**(1-alpha)))**(1/(alpha-1))
    #Compute asset supply given r0 using the solve_hh function we defined. 
    #Also get the stationary distribution and kk, the grid of optimal asset choices 
    cpol, kpol, gridk, lam, k1 = solve_hh_vfid(r0)
    r1 = alpha*A*max(0.001, k1)**(alpha-1) * np.sum(labor)**(1-alpha) - delta
    err_r = abs(r1-r0)
    if k1 > k0:
        maxrate = r0;
    else:
        minrate = r0;
    #print out k0, k1, r0, r1
    print(f'k0 = {k0:.4f}, k1 = {k1:.4f}, r0 = {r0:.4f}, r1 = {r1:.4f}')

print("VFI complete\n")

#Step 1B:
#Instead of using value function iteration, we could use the Endogenous Grid Points Method (EGM). 
#This function solves the Aiyagari household problem for a given r using EGM and returns the consumption and asset policy functions.
#Because it's not using so many iterated bisection loops, it's much faster!
#First we compute expected future utility of consumption given our initial guess at capital, then use the Euler equation to get the implied current consumption. 
#Using the budget constraints, we can find current assets, this gives assets next period. Iterate until the implied consumption converges to consumption.

print("Starting EGM\n")

#Initialize min/maxrate and err_r analogue for EGM method
minrate_egm = -delta #Initialize min interest rate for bisection
maxrate_egm = (1-beta) / beta #Initialize max interest rate for bisection
err_r_egm = 1

def solve_hh_egm(r):
    #initialize variables tracking error in consumption policy (so we can update and stop at convergence within tolerance) and initial policy guess for consumption and assets
    err_cpol = 10
    #We want to guess cpol[j,i] = z[j]*w + r*gridk[i] so let's calculate r and gridk
    #We know w given r
    w = (1-alpha)*(A*(alpha/(r+delta))**alpha)**(1/(1-alpha))

    #To calculate gridk, need the borrowing limit
    #Set borrowing limit to either lower limit b or the value of income in the worst productivity state, whichever is lower
    if r <= 0:
        phi = b
    else:
        phi = min(b, w*z[0]/r)

    #Initial guess that c = z*w + r*k (In SS, k_t+1 = k_t = k* so (1+r)k_t+1 - k_t = rk*)
    #To define initial guess for consumption policy, we need to define asset grid
    maxgrid = 25
    mingrid = phi
    incgridk = 0.09
    gridk = np.arange(mingrid, maxgrid + incgridk, incgridk)
    ngridk = len(gridk)

    #Initial guess for optimal asset policy function and consumption policy functions
    kpol = np.zeros((nz,ngridk))
    cpol = np.zeros((nz, ngridk))
    cpol_compute = np.zeros((nz, ngridk)) #For the implied consumption calculation
    
    #Initial guess for optimal consumption policy function
    for j in range(nz):
        for i in range(ngridk):
            cpol[j,i] = z[j]*w + r*gridk[i]

    while err_cpol > tol_r:
        #Compute E_t[U_t+1(c)] = P * c^-gamma where P is the transition matrix. Computing this for every possible current asset k_t value in gridk
        expected_muc = P @ cpol**(-1*gamma)
        #Using Euler equation, find implied current consumption. c_t^(-gamma) = (1+r)*beta* E_t[c_t+1^(-gamma) | z_t]
        c_impl = ((1+r)*beta*expected_muc)**(-1/gamma)
        #Using budget constraint, find implied future assets k_t+1.
        k_impl = (c_impl + gridk - z.reshape(nz, 1)*w)/(1+r)
        #k_impl likely not on grid points, will have to interpolate.
        for j in range(nz):
            kpol[j,:] = interp1d(k_impl[j,:], gridk, kind='linear', fill_value="extrapolate", bounds_error=False)(gridk)
        
        #Ensure that kpol is bounded above/below
        kpol = np.clip(kpol, gridk[0], gridk[-1])

        #Now compute current consumption from budget constraint given current assets k and implied next period assets kpol.
        for j in range(nz):
            for i in range(ngridk):
                cpol_compute[j,i] = z[j]*w + (1+r)*gridk[i] - kpol[j,i]
        
        #Compute error b/t cpol and cpol_compute
        err_cpol = np.max(np.abs(cpol - cpol_compute))
        cpol = cpol_compute.copy()

    #Initialize variable tracking difference in lambda
    err_lam_egm = 1

    #Compute stationary distribution by interpolating kpol across states
    lam_egm = np.ones((nz, ngridk)) / (nz * ngridk)
    while err_lam_egm > tiny:
        #Initialize stationary distribution matrix
        lam_new_egm = np.zeros((nz, ngridk))
        #For every state j, k_t+1 i, determine mass of population between grid points just above and below k_t+1 and compute the population mass
        for j in range(nz):
            for i in range(ngridk):
                k_next_period = kpol[j,i]
                low_side_index = np.clip(np.searchsorted(gridk, k_next_period) - 1, 0, ngridk-2)
                high_side_index = low_side_index + 1
                #linear interpolation weight
                weight = np.clip((k_next_period - gridk[low_side_index]) / (gridk[high_side_index] - gridk[low_side_index]), 0.0, 1.0)
                #Distribute mass across future productivity states and grid points
                for j_next in range(nz):
                    lam_new_egm[j_next, low_side_index] += (1-weight) * lam_egm[j,i]*P[j,j_next]
                    lam_new_egm[j_next, high_side_index] += weight * lam_egm[j,i]*P[j,j_next]
        err_lam_egm = np.max(np.abs(lam_new_egm - lam_egm))
        lam_egm = lam_new_egm.copy()

    #Compute average assets supplied
    meank = np.sum(kpol * lam_egm)

    #Return the consumption policy function, asset policy function, stationary distribution and vector of assets supplied
    return cpol, kpol, lam_egm, meank

while abs(err_r_egm) > tol_r:
    #initial guess for r
    r0 = 0.5*(maxrate_egm + minrate_egm)
    #Compute asset demand given r0
    k0 = ((r0 + delta) / (alpha*A*np.sum(labor)**(1-alpha)))**(1/(alpha-1))
    #Compute asset supply given r0 using the solve_hh function we defined. 
    #Also get the stationary distribution and k1, the average asset supply
    cpol_egm, kpol_egm, lam_egm, k1 = solve_hh_egm(r0)
    #Compute the r implied by this capital supplied
    r1 = alpha*A*max(0.001, k1)**(alpha-1) * np.sum(labor)**(1-alpha) - delta
    if k1 > k0:
        maxrate_egm = r0;
    else:
        minrate_egm = r0;
    #print out k0, k1, r0, r1
    print(f'k0 = {k0:.4f}, k1 = {k1:.4f}, r0 = {r0:.4f}, r1 = {r1:.4f}')
    err_r_egm = abs(r1 - r0)

#Step 2: 
#Now that we have the equilibrium distribution and asset-next period asset correspondence, graph them!
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

base_colors = plt.cm.viridis(np.linspace(0.2, 0.8, nz))
light_colors = base_colors.copy()
light_colors[:, :3] = 1 - 0.45 * (1 - base_colors[:, :3])  # blend toward white
dark_colors = base_colors.copy()
dark_colors[:, :3] = base_colors[:, :3] * 0.55  # darken

from matplotlib.lines import Line2D

# Asset policy function
ax = axes[0]
for j in range(nz):
    ax.plot(gridk, kpol[j,:], color=light_colors[j], label=f'z={z[j]:.2f}')
    ax.plot(gridk, kpol_egm[j,:], color=dark_colors[j], linestyle='--')
ax.plot(gridk, gridk, 'k-', linewidth=0.8, label='45°')
ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')
ax.set_xlabel('Current assets k')
ax.set_ylabel("Next period assets k'")
ax.set_title('Asset Policy Function')
handles, labels = ax.get_legend_handles_labels()
method_handles = [Line2D([0], [0], color='black', linestyle='-', label='VFI'),
                  Line2D([0], [0], color='black', linestyle='--', label='EGM')]
ax.legend(handles=handles[:-1] + method_handles, fontsize=7)

# Consumption policy
ax = axes[1]
for j in range(nz):
    ax.plot(gridk, cpol[j,:], color=light_colors[j], linestyle='-')
    ax.plot(gridk, cpol_egm[j,:], color=dark_colors[j], linestyle='--')
ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')
ax.set_xlabel('Current assets k')
ax.set_ylabel('Consumption c')
ax.set_title('Consumption Policy Function')
method_handles = [Line2D([0], [0], color='black', linestyle='-', label='VFI'),
                  Line2D([0], [0], color='black', linestyle='--', label='EGM')]
ax.legend(handles=method_handles, fontsize=7)

# Stationary distribution
ax = axes[2]
marginal_vfi = lam.sum(axis=0)
marginal_egm = lam_egm.sum(axis=0)
ax.bar(gridk, marginal_vfi, width=incgridk, color='steelblue', alpha=0.6, label='VFI')
ax.plot(gridk, marginal_egm, color='darkorange', linewidth=1.5, label='EGM')
ax.set_xlabel('Assets k')
ax.set_ylabel('Mass of agents')
ax.set_title('Stationary Distribution of Assets')
ax.legend(fontsize=7)

plt.tight_layout()
plt.show()