## DISCLAIMER: This model is not thorougly tested and may contain bugs / not be accurate. ##

using MixtureAgentsModels,Distributions,Plots

nsess = 10
ntrials = 2000
sim_options = TwoStepSim(ntrials=ntrials,nsess=nsess)

# agents = [MBrewardB(α=0.60),MBchoiceB(α=0.70),MFrewardB(α=0.40),MFchoiceB(α=0.50),Bias()]

βprior = [Normal(0,1)] # βprior is used for σInit
agents = [MBrewardB(α=0.60,βprior=βprior),MBchoiceB(α=0.70,βprior=βprior),MFrewardB(α=0.40,βprior=βprior),MFchoiceB(α=0.50,βprior=βprior),Bias(βprior=[Normal(0,0.5)])]

fit_symbs = [:α,:α,:α,:α]
fit_params = [1,2,3,4,0]
agent_options = AgentOptions(agents,fit_symbs,fit_params)


σ0 = [0.01,0.03,0.01,0.02,0.008]
σSess0 = [0.02,0.06,0.02,0.04,0.016]
σInit0 = [get_param(agent,:βprior)[1].σ for agent in agents]
maxiter = 10
nstarts = 1
tol = 1E-4
model_options = ModelOptionsDrift(σ0=σ0,σSess0=σSess0,σInit0=σInit0,tol=tol,maxiter=maxiter,nstarts=nstarts)

seed = 17040
data, model_sim, agents_sim = simulate(sim_options,model_options,agent_options;init_model=false,seed=seed)
plot_model(model_sim,agents_sim)


model_fit,agents_fit,ll_fit = optimize(data,model_options,agent_options;seed=seed)

plot_model(model_fit,agents_fit)
plot_model!(model_sim,agents_sim)
plot!(legend=false)


################################################################
using MixtureAgentsModels

file = "data/MBB2017_behavioral_dataset.mat"
varname = "dataset"
rat = 17
data = load_twostep(file,rat)



βprior = [Normal(0,1)] # βprior is used for σInit
agents = [MBrewardB(α=0.45,βprior=βprior),MBchoiceB(α=0.45,βprior=βprior),MFrewardB(α=0.8,βprior=βprior),MFchoiceB(α=0.8,βprior=βprior),Bias(βprior=[Normal(0,0.5)])]
fit_params = [1,2,3,4,0]
fit_symbs = [:α,:α,:α,:α]
agent_options = AgentOptions(agents,fit_symbs,fit_params)


maxiter = 10
nstarts = 1
tol = 1E-4
# σ0 = [0.052952582337305515,0.44796527375152967,0.08504320057468938,0.2707409260611602,0.3577253570431003]
# σSess0 = [0.052952582337305515,0.44796527375152967,0.08504320057468938,0.2707409260611602,0.3577253570431003]
# σInit0 = (get_param(agents,:θL2)) 

model_options = ModelOptionsDrift(tol=tol,maxiter=maxiter)#,σ0=σ0,σInit0=σInit0,σSess0=σSess0,nstarts=nstarts)

#break_on(:error)
model,agents,y,x,ll = optimize(data,model_options,agent_options;init_hypers=true)

plot_model(model,agents)
#savemat("220708_rat17_betaModel.mat",model)



@unpack new_sess_free = data
xs = findall(new_sess_free)
xss = hcat(xs,xs)
ys = ones(length(xs)) * 10
yss = hcat(-ys,ys)

pyplot()
plot(xss',yss',lc=:black,label="")
plot!(model.β[1,:],lc=:purple,label="MBr")
plot!(model.β[2,:],lc=:red,label="MBc")
plot!(model.β[3,:],lc=:green,label="MFr")
plot!(model.β[4,:],lc=:blue,label="MFc")
plot!(model.β[5,:],lc=:black,label="bias")
plot!(xlabel="trials",ylabel="weight")
#plot!(legend=false)
plot!(size=(1000,500))
plot!(xlim=[0,5000],ylim=[-2,2.5])
plot!(title="rat 17")