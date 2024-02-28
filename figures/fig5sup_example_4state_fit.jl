using MixtureAgentsModels
using Bootstrap,StatsBase,Plots,StatsPlots,DataFrames,HypothesisTests
using StatsPlots: violinoffsets
include("dirty_plotting.jl")

ex_rat1 = 7
ex_rat2 = 10
fit_i = 10
fldr = "model_fits_prior_mean"
sort_args = Dict(:method=>"trans",:a=>1)
sort_states = true

ex_plots1 = plot_model_fit(ex_rat1,fit_i;return_plots=true,data_fldr=fldr,sort_states=sort_states,sort_args...)[1];
ex_plots2 = plot_model_fit(ex_rat2,fit_i;return_plots=true,data_fldr=fldr,sort_states=sort_states,sort_args...)[1];


function plot_4state_fits(ex_plots1,ex_plots2)
    l1 = @layout [a{0.55h}; b{0.15w} c e{0.56w}]
    p1 = plot(ex_plots1[3],ex_plots1[1],ex_plots1[2],ex_plots1[5],layout=l1)
    yticks!(p1[2],[0,0.5,1])
    ylims!(p1[2],0,1)
    plot!(p1[1],legend=false,title="example model weights",xrotation=45,xlims=[0.55,4.45],ylims=[-2,3])
    xlims!(p1[4],-10,510)
    plot!(p1[4],title="expected state given trial (ex. rat)",yticks=[0,0.5,1],legend=false)


    # yticks!(p1[4],[0,0.5,1])
    # l2 = @layout [e{0.1w} f g{0.6w} h{0.17w}]
    p2 = plot(ex_plots2[3],ex_plots2[1],ex_plots2[2],ex_plots2[5],layout=l1)
    yticks!(p2[2],[0,0.5,1])
    ylims!(p2[2],0,1)
    plot!(p2[1],legend=false,title="example model weights",xrotation=45,xlims=[0.55,4.45],ylims=[-2,3])
    xlims!(p2[4],-10,260)
    plot!(p2[4],title="expected state given trial (ex. rat)",yticks=[0,0.5,1],legend=false)

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
    plot!(p[end],ylabel="prob.")
    plot!(p[6],right_margin=6mm)
  

    xlabel!("")
    title!("")
    # plot!(p[end-1],left_margin=10mm)
    # plot!(p[end],left_margin=5mm)

    # plot!(p[7],margin=0mm)


    return p
end

p = plot_4state_fits(ex_plots1,ex_plots2)
savefig("/Users/sarah/Library/CloudStorage/Dropbox/Princeton/Manuscripts/MoA-HMM/figures/fig5sup_ex_4state_fits.svg")


