using MixtureAgentsModels
include("dirty_plotting.jl")
using Statistics
using StatsPlots: violinoffsets

function ci(x)
    med_bs = bootstrap(median,x,BasicSampling(10000))
    med_ci = confint(med_bs,BCaConfInt(0.95))
    return (med_ci[1][1]-med_ci[1][2],med_ci[1][3]-med_ci[1][1])
end

moa_fldr = "model_fits"
glm_fldr = "model_fits"
glm_i = 5
ex_rat = 10

moa_glm_cor = zeros(3,20)
for (i,rat_i) in enumerate(vcat(collect(1:19),[21]))
    println(rat_i)
    # if rat_i == 7
    model_moa,agents_moa,_,agent_ops_moa,_ = load_model_fit(rat_i,2;data_fldr=moa_fldr)
    model_glm,agents_glm,model_ops,agent_ops_glm,data = load_model_fit(rat_i,glm_i;data_fldr=glm_fldr)

    match_states!(model_glm,agents_glm,model_moa,agents_moa,data)

    γs_moa = compute_posteriors(model_moa,agents_moa,data)[1]
    γs_glm = compute_posteriors(model_glm,agents_glm,data)[1]

    # moa_glm_cor[:,i] = diag(cor(γs_moa,γs_glm,dims=2))
    moa_glm_cor[:,i] = cor.(eachrow(γs_moa),eachrow(γs_glm))
end


function plot_moa_glm_comparison(moa_glm_cor,ex_rat)

    states = repeat([1,2,3],1,20)
    rats = permutedims(repeat(collect(1:20),1,3))
    df = DataFrame(rat=rats[:],state=states[:],cor=moa_glm_cor[:])
    df_grp = groupby(df,:state)
    df_avg = combine(df_grp,:cor=>median,:cor=>ci)

    offsets = vcat(permutedims.([violinoffsets(0.3,grp.cor) for (i,grp) in enumerate(df_grp)])...)[:]


    pcor = plot([0.5,3.5],[0,0],c=:black,lw=2,label="",xlims=[0.5,3.5])
    @df df violin!(:state,:cor,group=:state,c=[1 2 3],label="",alpha=0.5)
    @df df dotplot!(:state,:cor,group=:state,mc=[1 2 3],label="",ms=5)
    df_rat = groupby(df,:rat)
    @df df_rat[ex_rat] scatter!(:state,:cor,mc=:black,markershape=:star5,markersize=15,label="ex. rat")

    # @df df scatter!(:state.+offsets,:cor,mc=:black,label="",alpha=0.5,ms=5)
    # @df df plot!(:state.+offsets,:cor,group=:rat,c=:black,label="",alpha=0.5,ms=5)
    @df df_avg scatter!(:state,:cor_median,yerror=:cor_ci,color=:black,fc=:gray33,markersize=7,lw=3,label="")
    ylims!(-0.2,1)
    xlabel!("state")#,xlabelfontsize=16)
    xticks!(1:3)
    ylabel!("correlation")#,ylabelfontsize=16)
    title!("Average state prob. cor.",titlefontsize=18)
    plot!(framestyle=:box,xtickfontsize=12,ytickfontsize=12)

    # plot_model_fit(7,2;data_fldr=moa_fldr)[1]
    moa_plots,model_moa,agents_moa,data = plot_model_fit(ex_rat,2;data_fldr=moa_fldr,return_plots=true)
    glm_plots = plot_model_fit(ex_rat,glm_i;data_fldr=glm_fldr,return_plots=true)[1]

    # ns = size(model_moa.β,2)
    # na = size(model_moa.β,1)
    # agent_inds = repeat(1:na,1,ns)
    # state_inds = permutedims(repeat(1:ns,1,na))
    # data = DataFrame(state=state_inds[:],agent=agent_inds[:],beta=model_moa.β[:])
    # βplot = @df data groupedbar(:state, :beta, group=:agent, c=[1,2,3],label="")


    l1 = @layout [c{0.65w} d a{0.09w} b{0.14w}]
    p1 = plot(moa_plots[3:4]...,moa_plots[1:2]...,layout=l1)#,size=(1600,200),margin=5mm)
    title!(p1[3],"init. prob.")
    yticks!(p1[3],[0,0.5,1])
    ylims!(p1[3],0,1)
    xlabel!(p1[3],"state")
    ylims!(p1[1],-1.5,1.5)
    yticks!(p1[1],[-1,0,1])
    plot!(p1[1],xrotation=45)
    plot!(p1[1],legend=:top)
    plot!(p1[2],yticks=[0,0.5,1],xrotation=45)


    # l2 = @layout [a{0.1w} b c{0.77w}]
    l1 = @layout [c{0.65w} d a{0.09w} b{0.14w}]

    p2 = plot(glm_plots[3:4]...,glm_plots[1:2]...,layout=l1)#,size=(1950,200),margin=5mm)
    title!(p2[5],"init. prob.")
    yticks!(p2[5],[0,0.5,1])
    ylims!(p2[5],0,1)
    xlabel!(p2[5],"state")
    plot!(p2[1],right_margin=0mm)
    title!(p2[1],"state 1")
    ylims!(p2[1],-1.5,2.5)
    ylabel!(p2[1],"weight")
    title!(p2[2],"state 2")
    plot!(p2[2],yformatter=_->"",left_margin=0mm,right_margin=0mm)
    ylims!(p2[2],-1.5,2.5)
    title!(p2[3],"state 3")
    plot!(p2[3],yformatter=_->"",left_margin=0mm)
    ylims!(p2[3],-1.5,2.5)

    ylims!(p2[4],-1.5,1.5)
    title!(p2[4],"bias weights")
    xlabel!(p2[4],"state")
    xticks!(p2[4],[1,2,3],["1","2","3"])
    plot!(p2[4],legend=false)
    yticks!(p2[4],[-1,0,1])


    # l3 = @layout [e{0.4w} f]# g{0.4w} h]
    l3 = @layout [[e{0.4w} f; g{0.4w} h] i{0.3w}]
    # p3 = plot(moa_plots[5:6]...,layout=l3)
    p3 = plot(moa_plots[5:6]...,glm_plots[5:6]...,pcor,layout=l3)
    title!(p3[1],"expected state")
    yticks!(p3[1],[0,0.5,1])
    plot!(p3[1],right_margin=0mm)
    plot!(p3[2],yformatter=_->"",left_margin=0mm)
    title!(p3[3],"expected state")
    yticks!(p3[3],[0,0.5,1])
    plot!(p3[3],right_margin=0mm)
    plot!(p3[4],yformatter=_->"",left_margin=0mm)


    # p4 = plot(glm_plots[5:6]...,layout=l3)

    l = @layout [a; b; c{0.5h}]
    p = plot(p1,p2,p3,layout=l,size=(1500,800),framestyle=:box,margin=5mm)#,right_margin=0mm,top_margin=2mm)
    # plot!(p[1],left_margin=10mm)
    # plot!(p[5],left_margin=10mm)
    # plot!(p[8],left_margin=0mm)
    # plot!(p[9],left_margin=0mm)
    # plot!(p[11],left_margin=10mm)
    # plot!(p[13],left_margin=10mm)
    plot!(legend=false,xtickfontsize=12,ytickfontsize=12,ylabel="",xlabel="",title="")

    return p

end

p = plot_moa_glm_comparison(moa_glm_cor,ex_rat)
# plot!(left_margin=10mm)
savefig("/Users/sarah/Library/CloudStorage/Dropbox/Princeton/Manuscripts/MoA-HMM/figures/fig5sup_moa_to_glm_cors.svg")
