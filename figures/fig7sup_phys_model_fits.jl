using MixtureAgentsModels
using Bootstrap,StatsBase,Plots,StatsPlots,DataFrames,HypothesisTests
using StatsPlots: violinoffsets
include("dirty_plotting.jl")

ex_rat = "M055"
fit_i = 2
fldr = "model_fits_prior_mean_phys"
sort_args = Dict(:method=>"agent",:a=>1)
sort_states = true

ex_plots = plot_model_fit(ex_rat,fit_i;return_plots=true,data_fldr=fldr,sort_states=sort_states,sort_args...)[1];
sum_plots,all_models,all_agents,all_datas=plot_model_fit_summary(fit_i;return_plots=true,data_fldr=fldr,sort_states=sort_states,sort_args...);

function plot_all_fits(ex_plots,sum_plots,all_agents,all_datas)
    df_all = DataFrame()
    for (model,agents,data,rat_i) in zip(all_models,all_agents,all_datas,1:length(all_models))
        _,_,_,df = plot_gammas(model,agents,data)
        df[!,:rat] .= rat_i
        df_all = vcat(df_all,df)

    end
    df_grp = groupby(df_all,[:state,:trial])
    df_avg = combine(df_grp,:gamma_mean=>mean,:gamma_mean=>sem)
    replace!(df_avg.gamma_mean_sem,NaN=>0)
    df_z = groupby(df_avg,:state)
    avgplot = plot()
    ns = 3
    df_grp = groupby(df_all,:state)
    for z = 1:ns
        grp_i = df_grp[z][!,:trial] .<= 500
        z_i = df_z[z][!,:trial] .<= 500
        @df df_grp[z][grp_i,:] plot!(avgplot,:trial,:gamma_mean,group=:rat,alpha=0.3,c=z,label="")
        @df df_z[z][z_i,:] plot!(avgplot,:trial,:gamma_mean_mean,ribbon=1.96 .* :gamma_mean_sem, c=z, linewidth=3, legend=false, label="state "*string(z))#, legend=false)
    end
    plot!(ylabel="prob.",xlabel="trial",title="expected state given trial (all rats)")
    xlims!(0,500)
    yticks!(0:0.5:1)
    

    # l = @layout [a b c{0.5w} d; e f g{0.5w} h; i j]
    # l1 = @layout [[_ _; a{0.999h} b{0.999h}; _ _] c{0.5w} [_; d{0.999h}; _]]
    l1 = @layout [a{0.55h}; b{0.15w} c e{0.56w}]
    p1 = plot(ex_plots[3],ex_plots[1],ex_plots[2],ex_plots[5],layout=l1)
    yticks!(p1[2],[0,0.5,1])
    ylims!(p1[2],0,1)
    plot!(p1[1],legend=false,title="example model weights",xrotation=45,xlims=[0.55,3.45],ylims=[-2,3])
    xlims!(p1[4],-10,510)
    plot!(p1[4],title="expected state given trial (ex. rat)",yticks=[0,0.5,1],legend=false)


    # yticks!(p1[4],[0,0.5,1])
    # l2 = @layout [e{0.1w} f g{0.6w} h{0.17w}]
    p2 = plot(sum_plots[3],sum_plots[1],sum_plots[2],avgplot,layout=l1)
    yticks!(p2[2],[0,0.5,1])
    ylims!(p2[2],0,1)
    plot!(p2[1],legend=false,title="population model weights",xrotation=45,xlims=[0.55,3.45],ylims=[-2,3])
    xlims!(p2[6],-10,510)
    # title!(p2[6],"expected state given trial (all rats)")


    # yticks!(p2[5],[0,0.5,1])
    # ylims!(p2[5],0,1)
    # l3 = @layout [i j]
    # p3 = plot(ex_plots[5],avgplot,layout=l3)
    # xlims!(p3[1],0,500)
    l = @layout [a b]
    # yticks!(p3[1],[0,0.5,1])
    p = plot(p1,p2,layout=l,size=(1600,500),framestyle=:box)
    plot!(margin=5mm,left_margin=2mm,right_margin=2mm)#,top_margin=3mm,left_margin=5mm,right_margin=5mm,bottom_margin=12mm)#,right_margin=0mm,top_margin=2mm)
    # plot!(p[4],right_margin=8mm)
    # plot!(p[end],right_margin=8mm)

    plot!(ylabelfontsize=16,
        ytickfontsize=14,
        xlabelfontsize=16,
        xtickfontsize=14,
        legendfontsize=10,
        titlefontsize=18)
        
    # plot!(p[1],left_margin=12mm)
    # plot!(p[5],left_margin=12mm)
    # plot!(p[6],yguidefontrotation=-90)
    # annotate!(p[5],-0.1,0.5,text(1))
    ylabel!("")
    plot!(p[4],ylabel="prob.")
    plot!(p[10],ylabel="prob.")
    plot!(p[6],right_margin=6mm)
    plot!(p[7],yticks=([0.5],[1]),bottom_margin=-5mm)#,top_margin=10mm)
    annotate!(p[7],0.25,0,text(0,10;rotation=90))
    annotate!(p[7],0.25,1,text(1,10;rotation=90))
    plot!(p[8],yticks=([0.5],[2]),top_margin=0mm,bottom_margin=-5mm)
    annotate!(p[8],0.25,0,text(0,10;rotation=90))
    annotate!(p[8],0.25,1,text(1,10;rotation=90))
    plot!(p[9],yticks=([0.5],[3]),top_margin=0mm)
    annotate!(p[9],0.25,0,text(0,10;rotation=90))
    annotate!(p[9],0.25,1,text(1,10;rotation=90))
    xlabel!("")
    title!("")


    return p,df_all
end

p = plot_all_fits(ex_plots,sum_plots,all_agents,all_datas)[1]
savefig("/Users/sarah/Library/CloudStorage/Dropbox/Princeton/Manuscripts/MoA-HMM/figures/fig7sup_phys_ex_and_summary_3state_fits.svg")



