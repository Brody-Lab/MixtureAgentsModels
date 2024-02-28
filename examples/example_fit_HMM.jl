using MixtureAgentsModels # import package

########################################################################################
####################              Fit to simulated data            #####################
########################################################################################
##- define simulation options -##
nsess = 10          # number of sessions
ntrials = 1000      # total number of trials
mean_ntrials = 350  # mean number of trials per session; will override ntrials
sim_options = TwoStepSim(nsess=nsess,ntrials=ntrials,mean_ntrials=mean_ntrials)

##- define agent options -##
# agent vector, user set learning rates for RL agents
agents_sim = [MBrewardB(α=0.60),MBchoiceB(α=0.70),MFrewardB(α=0.40),MFchoiceB(α=0.50),Bias()]
fit_symbs = [:α,:α,:α,:α]  # symbols for each parameter to fit
fit_params = [1,2,3,4,0]   # index linking fit_symbs to corresponding agent in agents_sim
agent_options = AgentOptions(agents_sim,fit_symbs,fit_params)

##- define HMM options -##
nstates = 2    # number of hidden states
# user set initial values for HMM parameters to be used for simulation
β0 = [0.79 1.25; -0.82 -0.19; -0.01 0.1; 0.25 0.54; 0.32 0.26] # (nagents x nstates) matrix of agent weights
π0 = [1.0,0.0] # (nstates x 1) vector of initial state probabilities
A0 = [0.9 0.1; 0.01 0.99] # (nstates x nstates) transition matrix
maxiter = 100  # maximum number of iterations for EM algorithm
nstarts = 1    # number of reinitializations for EM algorithm
tol = 1E-4     # tolerance for convergence of EM algorithm
model_options = ModelOptionsHMM(nstates=nstates,β0=β0,π0=π0,A0=A0,maxiter=maxiter,tol=tol,nstarts=nstarts)

##- simulate data, fit model, and plot results -##
# simulate data, set init_model to false to use user-defined model parameters
data,model_sim,agents_sim = simulate(sim_options,model_options,agent_options;init_model=false)
# plot simulated model and hidden state probabilities in example sessions
plot_model(model_sim,agents_sim,agent_options,data)
# fit model to simulated data
model_fit,agents_fit,ll_fit = optimize(data,model_options,agent_options;disp_iter=10) # only print every 10th iteration
# plot fit model and hidden state probabilities in example sessions
plot_model(model_fit,agents_fit,agent_options,data)


########################################################################################
######################              Fit to real data            ########################
########################################################################################
##- load in data -##
# load two-step data using custom function that parses twostep .mat files
file = "data/MBB2017_behavioral_dataset.mat" # path to data file
rat = 17              # rat number
data = load_twostep(file,rat)

##- define agent options -##
# agent vector
agents = [MBrewardB(),MBchoiceB(),MFrewardB(),MFchoiceB(),Bias()]
fit_symbs = [:α,:α,:α,:α]  # symbols for parameters to fit
fit_params = [1,2,3,4,0]   # corresponding fit parameter index for each agent
agent_options = AgentOptions(agents,fit_symbs,fit_params)

##- define HMM options -##
nstates = 3   # number of hidden states
maxiter = 10  # maximum number of iterations for EM algorithm (this probably won't converge, just for example. defaults to 300)
nstarts = 1   # number of reinitializations for EM algorithm
tol = 1E-4    # tolerance for convergence of EM algorithm
model_options = ModelOptionsHMM(nstates=nstates,tol=tol,maxiter=maxiter,nstarts=nstarts)

##- fit model to data -##
model,agents,ll = optimize(data,model_options,agent_options)
# need more iterations? restart from previous fit
model,agents,ll = optimize(data,model,agents,model_options,agent_options)
# plot fit model and hidden state probabilities in example sessions
plot_model(model,agents,agent_options,data)
# save fit to .mat file
savefit("example_fit_HMM.mat",model,model_options,agents,agent_options,data)
# load fit from .mat file
model,model_options,agents,agent_options,data,ll = loadfit("example_fit_HMM.mat")
