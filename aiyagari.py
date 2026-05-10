#Solving Aiygari model using Value Function Iteration
#Elyse St Sauver
#Adapted from code made available for MATLAB by Oliko Vardishvili, https://github.com/ovardish/2022_Computational_Course_Code


#Assumes CRRA utility preferences over consumption, inelastic labor, and a Cobb-Douglas production function.
#Computes the equilibrium consumption policy, asset policy and stationary distribution in the Aiyagari model using discrete Value Function Iteration

#1. Define the aiyagari household problem, and solve for optimal assets next period given an interest rate r.
#2. Find equilibrium r by bisection method, compute assets and consumption.
#3. Plot the equilibrium policy functions and stationary distribution.


#Step 0:
import numpy as np
import quantecon as qe
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
tiny = 0.1**10 #arbitrary epsilon

#Algorithm parameters
tol_r = 0.0001 #tolerance for convergence of interest rate
tol_vf = 0.0001 #tolerance for convergence of value function
minrate = -delta #Initialize min interest rate for bisection
maxrate = (1-beta) / beta #Initialize max interest rate for bisection
err_r = 1 #Inital value for convergence error

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

#Step 1:
#Now that we know the productivity shock process, can iterate to find equilibrium.
#Will compute solutions by converging to optimal r.
#For each guess at r, we will solve the aiyagari household problem using aiyagari(r).
#We will iterate by bisection using this function until r1 - r0 < tol_r

# Function that solves the Aiyagari household problem for a given r. Returns the mean capital demand.
def solve_hh(r):
    #initialize variables tracking error in value functions, lambda (so we can update and stop at convergence within tolerance)
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
    mingrid = -phi
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
    meank = np.sum(kk * lam)
    return gridk, lam, kk, meank

#Step 2:
#Now we know how to compute the average capital supplied for a given r. 
#Using bisection, compute r from k supplied and from k demanded, and stop when those two rs are equal (within tolerance). This is the equilibrium r!
while abs(err_r) > tol_r:
    #initial guess for r
    r0 = 0.5*(maxrate + minrate)
    #Compute asset demand given r0
    k0 = ((r0 + delta) / (alpha*A*np.sum(labor)**(1-alpha)))**(1/(alpha-1))
    #Compute asset supply given r0 using the solve_hh function we defined. 
    #Also get the stationary distribution and kk, the grid of optimal asset choices 
    gridk, lam, kk, k1 = solve_hh(r0)
    r1 = alpha*A*max(0.001, k1)**(alpha-1) * np.sum(labor)**(1-alpha) - delta
    err_r = abs(r1-r0)
    if k1 > k0:
        maxrate = r0;
    else:
        minrate = r0;
    #print out k0, k1, r0, r1
    print(f'k0 = {k0:.4f}, k1 = {k1:.4f}, r0 = {r0:.4f}, r1 = {r1:.4f}')


#Step 3: 
#Now that we have the equilibrium distribution and asset-next period asset correspondence, graph them!
ngridk = len(gridk)
wage = (1-alpha)*(A*(alpha/(r0+delta))**alpha)**(1/(1-alpha))
c = np.zeros((nz, ngridk))
for j in range(nz):
        for i in range(ngridk):            
            c[j,i] = z[j]*wage + (1+r0)*gridk[i] - kk[j,i]

#Asset policy function: k' as a function of k, for each z
plt.figure()
for j in range(nz):
    plt.plot(gridk, kk[j,:], label=f'z={z[j]:.2f}')
plt.plot(gridk, gridk, 'k--', label='45 degree line')
plt.xlabel('Current assets k')
plt.ylabel("Next period assets k'")
plt.title('Asset Policy Function')
plt.legend()
plt.show()

#Consumption policy: c as a function of k, for each z
plt.figure()
for j in range(nz):
    plt.plot(gridk, c[j,:], label=f'z={z[j]:.2f}')
plt.xlabel('Current assets k')
plt.ylabel('Consumption c')
plt.title('Consumption Policy Function')
plt.legend()
plt.show()

#Stationary distribution: marginal distribution over assets
plt.figure()
marginal = lam.sum(axis=0)  # sum over productivity states
plt.plot(gridk, marginal)
plt.xlabel('Assets k')
plt.ylabel('Mass of agents')
plt.title('Stationary Distribution of Assets')
plt.show()