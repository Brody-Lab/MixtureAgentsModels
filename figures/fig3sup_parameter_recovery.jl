using MixtureAgentsModels
using MAT,Plots,Measures,Statistics, LinearAlgebra


# data_fldr = "/Users/sarah/Library/CloudStorage/Dropbox/data/julia/parameter_recovery"
function load_pr_data(data_fldr)
    files = readdir(data_fldr)
    cat_fields = ["alpha_recovery","beta_recovery","pi_recovery","A_recovery","ll_recovery"]
    data_all = [matread(joinpath(data_fldr,files[f])) for f in 1:length(files)]
    data = deepcopy(data_all[1])

    for fld in cat_fields
        for d = 2:length(data_all)
            data[fld] = cat(data[fld],data_all[d][fld],dims=length(size(data_all[d][fld]))+1)
        end
    end
    return data,length(data_all)
end

function plot_parameter_recovery(data,model_i=2)
    model_ops,agent_ops,_ = get_presets(1)
    model_ops = model_ops[model_i]
    agent_ops = agent_ops[model_i]

    pa,pb,pc,pd,pe = plot_recovery(data["alpha_recovery"],data["beta_recovery"],data["pi_recovery"],data["A_recovery"],data["ll_recovery"],agent_ops;return_plots=true)
    # pr = plot_recovery(data["alpha_recovery"],data["beta_recovery"],data["pi_recovery"],data["A_recovery"],data["ll_recovery"],agent_ops)
    [annotate!(pa[i],mean(ylims(pa[i])),xlims(pa[i])[2]+xlims(pa[i])[2]/8,text("r="*string(round(cor(data["beta_recovery"][i,:,1,:][:],data["beta_recovery"][i,:,2,:][:]),digits=2)))) for i in 1:5]
    [annotate!(pb[i],0.5,1.075,text("r="*string(round(cor(data["alpha_recovery"][i,1,:][:],data["alpha_recovery"][i,2,:][:]),digits=2)))) for i in 1:4]
    annotate!(pc,0.5,1.075,text("r="*string(round(cor(data["pi_recovery"][:,1,:][:],data["pi_recovery"][:,2,:][:]),digits=2))))

    ns = 3
    diag_A = (diag.(eachslice(data["A_recovery"][:,:,1,:],dims=3)),diag.(eachslice(data["A_recovery"][:,:,2,:],dims=3)))
    annotate!(pd[1],0.5,1.075,text("r="*string(round(cor(reduce(vcat,diag_A[1]),reduce(vcat,diag_A[2])),digits=2))))

    off_A = ([reduce(vcat,diag.(eachslice(data["A_recovery"][:,:,1,:],dims=3),z)) for z=1:ns-1],[reduce(vcat,diag.(eachslice(data["A_recovery"][:,:,2,:],dims=3),z)) for z=1:ns-1])
    annotate!(pd[2],0.5,1.075,text("r="*string(round(cor(reduce(vcat,off_A[1]),reduce(vcat,off_A[2])),digits=2))))

    l = @layout [a b c d e; f g h i _; j k l _ _]
    pr = plot(pa...,pb...,pc,pd...,layout = l,size=(1500,1000),left_margin=5mm,top_margin=0mm,framestyle=:box)
    plot!(xtickfontsize=12,ytickfontsize=12,xlabelfontsize=16,ylabelfontsize=16)

    return pr
end

data_fldr = "/Users/sarah/Library/CloudStorage/Dropbox/data/julia/parameter_recovery"
data,nsims = load_pr_data(data_fldr)
p1 = plot_parameter_recovery(data)
savefig("/Users/sarah/Library/CloudStorage/Dropbox/Princeton/Manuscripts/MoA-HMM/figures/fig3sup_parameter_recovery.svg")

data_fldr = "/Users/sarah/Library/CloudStorage/Dropbox/data/julia/parameter_recovery_rat"
data,nsims = load_pr_data(data_fldr)
plot_parameter_recovery(data)
savefig("/Users/sarah/Library/CloudStorage/Dropbox/Princeton/Manuscripts/MoA-HMM/figures/fig3sup_parameter_recovery_rat.svg")

