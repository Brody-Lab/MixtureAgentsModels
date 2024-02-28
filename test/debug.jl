

using Test, MixtureAgentsModels, StatsBase, Distributions
import MixtureAgentsModels: moment_α


# @testset "MoAHMM-agents" begin
file = "../data/MBB2017_behavioral_dataset.mat"
rat = 8
data = load_twostep(file,rat)
nstates = 1
model_options = ModelOptionsHMM(nstates=nstates,nstarts=10)

# @testset "miller-agents" begin
agent_options = miller2017()
model,agents,ll = optimize(data,model_options,agent_options;verbose=false)

β = [2.70;-1.11;0.54;0.45;;]
α = 0.71
isapprox(round.(model.β,digits=2),β,atol=0.01)
isapprox(round.(get_params(agents,agent_options),digits=2)[1],α,atol=0.01)
isapprox(ll,-15095.7,atol=0.1)

print(round.(model.β,digits=2))
print(round.(get_params(agents,agent_options),digits=2))

# @testset "venditto-agents" begin
agent_options = venditto2023()
model,agents,ll = optimize(data,model_options,agent_options;init_beta=false,verbose=false)

β = [1.01;-1.06;0.17;0.42;0.34;;]
α = [0.83,1.0,0.87,0.30]
isapprox(round.(model.β,digits=2),β,atol=0.1)
isapprox(round.(get_params(agents,agent_options),digits=2)[[1,4]],α[[1,4]],atol=0.1)
isapprox(ll,-14628.2,atol=0.1)

 

# @testset "sim-agents" begin
agents = [MBbellmanB(), TD1B(), Bias()]
fit_symbs = [:α,:α]
fit_params = [1,2,0]
agent_options = AgentOptions(agents,fit_symbs,fit_params)
model,agents,ll = optimize(data,model_options,agent_options;verbose=false)

β = [1.5; 0.2; 0.39;;]
α = [0.73, 1.0]
isapprox(round.(model.β,digits=2),β,atol=0.01)
isapprox(round.(get_params(agents,agent_options),digits=2)[1],α[1],atol=0.01)
isapprox(ll,-16686.5,atol=0.1)


# @testset "shared-param" begin
agents = [MBreward(), MBchoice(), MFreward(), MFchoice(), Bias()]
fit_symbs = [:α,:α]
fit_params = [1,1,2,2,0]
agent_options = AgentOptions(agents,fit_symbs,fit_params)
model,agents,ll = optimize(data,model_options,agent_options;verbose=false)

β = [1.13; -1.13; 0.25; 1.44; 0.34;;]
α = [0.97, 0.36]
isapprox(round.(model.β,digits=2),β,atol=0.01)
isapprox(round.(get_params(agents,agent_options),digits=2),α,atol=0.01)
isapprox(ll,-14694.9,atol=0.1)


rat = 17
data = load_twostep(file,rat)

# # @testset "multi-param" begin
# agents = [TD0(), TD1(), Bias()]
# fit_symbs = [:α,:γ,:α]
# fit_params = [1:2,3,0]
# agent_options = AgentOptions(agents,fit_symbs,fit_params)
# model,agents,ll = optimize(data,model_options,agent_options;init_beta=false,verbose=false)

# β = [1.45; 2.21; -0.15;;]
# α = [0.12, 1.0, 0.2]
# isapprox(round.(model.β,digits=2),β,atol=0.01)
# isapprox(round.(get_params(agents,agent_options),digits=2),α,atol=0.01)
# isapprox(ll,-6410.2,atol=0.1)



# @testset "GLM" begin
agents = TransReward(1:5)
agent_options = AgentOptions(agents)
model,agents,ll = optimize(data,model_options,agent_options;verbose=false)

β = [1.53; -0.38; -0.24; 1.34; 0.49; 0.14; -0.29; 0.75; 0.24; 0.04; -0.11; 0.29; 0.21; -0.12; -0.05; 0.22; 0.12; 0.16; -0.01; 0.13;;]
isapprox(round.(model.β,digits=2),β,atol=0.01)
isapprox(ll,-5348.8,atol=0.1)



# @testset "clicks-scaling" begin
file = "../data/pclicks_dataset.csv" 
rat = "A327"         
data = load_pclicks(file,rat)

nstates = 1   
nstarts = 10   
model_options = ModelOptionsHMM(nstates=nstates,nstarts=nstarts,βprior=false)

agents = [DeltaClicks(),MFreward(),MFchoice(),Bias()]
fit_symbs = [:α,:α]  
fit_params = [0,1,2,0]  
agent_options = AgentOptions(agents,fit_symbs,fit_params)
model,agents,ll = optimize(data,model_options,agent_options;verbose=false)

β = [0.06; 1.75; 2.42; 0.12;;]
α = [0.24, 0.4]
isapprox(round.(model.β,digits=2),β,atol=0.01)
isapprox(round.(get_params(agents,agent_options),digits=2),α,atol=0.01)
isapprox(ll,-2743.2,atol=0.1)

agent_options = AgentOptions(agents,fit_symbs,fit_params;scale_x=true)
model,agents,ll = optimize(data,model_options,agent_options;verbose=false)

β = [1.09; 1.75; 2.42; 0.25;;]
α = [0.24, 0.4]

isapprox(round.(model.β,digits=2),β,atol=0.01)
isapprox(round.(get_params(agents,agent_options),digits=2),α,atol=0.01)
isapprox(ll,-2931.2,atol=0.1)


# @testset "MoAHMM-sim" begin
agents = [MBrewardB(α=0.60),MBchoiceB(α=0.70),MFrewardB(α=0.40),MFchoiceB(α=0.50),Bias()]
fit_symbs = [:α,:α,:α,:α]
fit_params = [1,2,3,4,0]
agent_options = AgentOptions(agents,fit_symbs,fit_params)

nstates = 3
β0 = [0.79 1.56 0.94; -0.82 -0.47 0.09; -0.01 0.12 0.08; 0.25 0.41 0.67; 0.32 0.23 0.29]
π0 = [1.0,0.0,0.0]
A0 = [0.95 0.05 0.0; 0.0 0.99 0.01; 0.0 0.0 1.0]
model_options = ModelOptionsHMM(nstates=nstates,β0=β0,π0=π0,A0=A0)

# @testset "3X1state-sim" begin
nsess = 100
mean_ntrials = 200
sim_options = TwoStepSim(nsess=nsess,mean_ntrials=mean_ntrials)


data,model_sim,agents_sim,z_sim = simulate(sim_options,model_options,agent_options;init_model=false,return_z=true)
γs,_,_ = compute_posteriors(model_sim,agents_sim,data)
z_fit = argmax.(eachcol(γs))

cor(z_sim,z_fit) >= 0.8


# @testset "3X1state-fit" begin
nsess = 25
mean_ntrials = 200
sim_options = TwoStepSim(nsess=nsess,mean_ntrials=mean_ntrials)

data,model_sim,agents_sim = simulate(sim_options,model_options,agent_options;init_model=false)
model_fit,agents_fit,ll_fit = optimize(data,model_options,agent_options;init_hypers=false,verbose=false)
mean(cor.(eachcol(model_sim.β),eachcol(Array(model_fit.β)))) > 0.9


# @testset "MoAHMM-priors" begin

file = "../data/MBB2017_behavioral_dataset.mat"

# @testset "agent-priors" begin
rat = 8
data = load_twostep(file,rat)

nstates = 1
model_options = ModelOptionsHMM(nstates=nstates,nstarts=10)

αprior = Beta(100,100)
agents = [MBrewardB(αprior=αprior),MBchoiceB(αprior=αprior),MFrewardB(αprior=αprior),MFchoiceB(αprior=αprior),Bias()]        
fit_symbs = repeat([:α],4)
fit_params = [1,2,3,4,0]
agent_options = AgentOptions(agents,fit_symbs,fit_params;fit_priors=true)
model,agents,ll = optimize(data,model_options,agent_options;init_beta=false,verbose=false)

β = [1.02; -1.09; 0.11; 0.44; 0.36;;]
α = [0.78, 0.89, 0.58, 0.32]        
isapprox(round.(model.β,digits=2),β,atol=0.1)
isapprox(round.(get_params(agents,agent_options),digits=2),α,atol=0.1)
isapprox(ll,-15401.3,atol=0.1)




# @testset "HMM-priors" begin

rat = 1
data = load_twostep(file,rat)

β0 = [1.27 3.07 1.3; 0.37 -2.32 0.52; 0.21 0.27 0.21; 0.15 0.17 0.3; 0.06 2.69 0.32]
A0 = [0.99 0.01 0.0; 0.0 0.76 0.24; 0.0 0.05 0.95]
π0 = [0.95, 0.05, 0.0]
α0 = [0.77, 0.51, 0.33, 0.24]

nstates = 3
model_options = ModelOptionsHMM(nstates=nstates,β0=β0,π0=π0,A0=A0,tol=1E-4)

agents = [MBrewardB(α=α0[1]),MBchoiceB(α=α0[2]),MFrewardB(α=α0[3]),MFchoiceB(α=α0[4]),Bias()]
fit_symbs = [:α,:α,:α,:α]
fit_params = [1,2,3,4,0]
agent_options = AgentOptions(agents,fit_symbs,fit_params)

# no prior initialized at one local min
model,agents,ll = optimize(data,model_options,agent_options;init_hypers=false,disp_iter=10);
isapprox(round.(model.β,digits=2),β0,atol=0.1)
isapprox(model.π,π0,atol=0.1)
isapprox(model.A,A0,atol=0.1)
isapprox(round.(get_params(agents,agent_options),digits=2),α0,atol=0.1)

βmean = [-0.13 1.51 1.16; -0.04 0.41 0.23; 0.34 0.28 0.14; 0.26 0.08 0.28; 0.15 0.06 0.44]
Amean = [0.83 0.17 0.0; 0.01 0.98 0.01; 0.0 0.0 1.0]
πmean = [1.0, 0.0, 0.0]
αmean = [0.72, 0.3, 0.41, 0.28]

βpriors = Normal.(βmean,0.01)
Apriors = round.(abs.(moment_α.(Amean,0.01))) .+ 1
πpriors = round.(abs.(moment_α.(πmean,0.01))) .+ 1
model_options = ModelOptionsHMM(nstates=nstates,β0=β0,π0=π0,A0=A0,α_π=πpriors,α_A=Apriors,tol=1E-4)

agents = [MBrewardB(α=α0[1],βprior=βpriors[1,:]),MBchoiceB(α=α0[2],βprior=βpriors[2,:]),MFrewardB(α=α0[3],βprior=βpriors[3,:]),MFchoiceB(α=α0[4],βprior=βpriors[4,:]),Bias(βprior=βpriors[5,:])]
agent_options = AgentOptions(agents,fit_symbs,fit_params;fit_priors=true)

# prior of a different local min, initialized at the same local min as above
model,agents,ll = optimize(data,model_options,agent_options;init_hypers=false,disp_iter=10);
isapprox(round.(model.β,digits=2),βmean,atol=0.1)
isapprox(model.π,πmean,atol=0.1)
isapprox(model.A,Amean,atol=0.1)
isapprox(round.(get_params(agents,agent_options),digits=2),αmean,atol=0.1)
