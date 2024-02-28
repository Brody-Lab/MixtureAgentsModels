using MixtureAgentsModels
using Plots
using RollingFunctions
using Measures
include("dirty_plotting.jl")


########################################################################################
####################              Fit to simulated data            #####################
agents = [MBrewardB(α=0.5),MBchoiceB(α=0.5),MFrewardB(α=0.5),MFchoiceB(α=0.5)]
model_mbr = ModelHMM(β=[1.;0.;0.;0.],π=[1.],A=[1.;;])
model_mbc = ModelHMM(β=[0.;1.;0.;0.],π=[1.],A=[1.;;])
model_mfr = ModelHMM(β=[0.;0.;1.;0.],π=[1.],A=[1.;;])
model_mfc = ModelHMM(β=[0.;0.;0.;1.],π=[1.],A=[1.;;])
models = [model_mbr,model_mbc,model_mfr,model_mfc]

nsess = 20 # 25
ntrials = 5000 # 10000
sim_op = TwoStepSim(nsess=nsess,ntrials=ntrials)
datas = []
for model in models
    datas = vcat(datas,simulate(sim_op,model,agents))
end

glm_ops = twostep_glm()
model_ops = ModelOptionsHMM(nstates=1,nstarts=10)
glm_fits = Vector{ModelHMM}(undef,length(datas))
for (d,data) in enumerate(datas)
    glm_fit,_,_ = optimize(data,model_ops,glm_ops)
    glm_fits[d] = deepcopy(glm_fit)
end

p = Vector{Any}(undef,4)
set_i = ["MBr","MBc","MFr","MFc"]
for (i,glm_fit) in enumerate(glm_fits)
    p[i] = plot_tr(glm_fit,glm_ops.agents)[1]
    title!(p[i],set_i[i])
    yticks!([-1,0,1])
end
pl = plot(p...,lw=5,layout = (2,2),size=(1000,600),left_margin=10mm,margin=5mm,framestyle=:box,
    ylims=(-1.7,1.7),legend=false,ytickfontsize=14,xtickfontsize=14,ylabelfontsize=16,titlefontsize=18)
savefig("/Users/sarah/Library/CloudStorage/Dropbox/Princeton/Manuscripts/MoA-HMM/figures/fig1_agent_glms.svg")


########################################################################################
######################              Fit to real data            ########################
file = "data/MBB2017_behavioral_dataset.mat"
varname = "dataset"
# rat = 7 
data_rat = []
for rat in vcat(collect(1:19),21)
    data_rat = vcat(data_rat,load_twostep(file,rat))
end


rat_fits = Vector{ModelHMM}(undef,length(data_rat))
for (d,data) in enumerate(data_rat)
    rat_fit,_,_ = optimize(data,model_ops,glm_ops)
    rat_fits[d] = deepcopy(rat_fit)
end


betas = hcat([Array(rat_fit.β) for rat_fit in rat_fits]...)
errs = 1.96 .* sem.(eachrow(betas))
glm_mean = mean_model(rat_fits)
p2 = plot_tr(glm_mean,glm_ops.agents,err=errs)[1]
plot!(p2,lw=5,size=(500,300),left_margin=10mm,margin=5mm,framestyle=:box,ylims=(-1.7,1.7),legend=false,ytickfontsize=14,xtickfontsize=14,ylabelfontsize=16,titlefontsize=18)
yticks!([-1,0,1])


savefig("/Users/sarah/Library/CloudStorage/Dropbox/Princeton/Manuscripts/MoA-HMM/figures/fig1_rat_glms.svg")

# savefig("/Users/sarah/Library/CloudStorage/Dropbox/Princeton/Manuscripts/MoA-HMM/figures/fig4_agent_interactions.svg")
