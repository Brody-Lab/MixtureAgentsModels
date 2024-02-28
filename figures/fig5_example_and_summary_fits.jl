using MixtureAgentsModels
using Bootstrap,StatsBase,Plots,StatsPlots,DataFrames,HypothesisTests,NaNStatistics,MixedModels
using StatsPlots: violinoffsets
include("dirty_plotting.jl")

sort_args = Dict(:method=>"agent",:a=>1)
all_models,all_agents,all_mops,all_aops,all_datas = load_model_fits(2;data_fldr="model_fits_prior_mean",sort_states=true,sort_args...)

ex_rat = 7
fit_i = 2
fldr = "model_fits_prior_mean"
sort_args = Dict(:method=>"agent",:a=>1)
sort_states = true

ex_plots,model,agents,data = plot_model_fit(ex_rat,fit_i;return_plots=true,sessions=4:6,data_fldr=fldr,sort_states=sort_states,sort_args...);
sum_plots,all_models,all_agents,all_datas=plot_model_fit_summary(fit_i;return_plots=true,data_fldr=fldr,sort_states=sort_states,sort_args...);

# plot_1state_weights()
# plot!(framestyle=:box,legend=false,xtickfontsize=14,ytickfontsize=14,ylabelfontsize=16,size=(500,300))
# savefig("/Users/sarah/Library/CloudStorage/Dropbox/Princeton/Conferences/Cosyne2024/1state_moa_fit.svg")

# stats
na = 5
ns = 3
nread = length(all_models)
all_betas = cat(map((x)->x.β,all_models)...,dims=3)
rat_inds = cat(map((x)->zeros(Int,na,ns).+x,collect(1:nread))...,dims=3)
agent_inds = repeat(repeat(1:na,1,ns),1,1,nread)
state_inds = repeat(permutedims(repeat(1:ns,1,na)),1,1,nread)
df = DataFrame(rat=string.(rat_inds[:]),state=string.(state_inds[:]),agent=agent_inds[:],beta=all_betas[:])
# data_grp = groupby(data,[:agent,:state])
# data_mean = combine(data_grp, :beta=>mean, :beta=>sem)

df_a = groupby(df,:agent)[4]
combine(groupby(df_a,:state),:beta=>median)
df_a = vcat(groupby(df_a,:state)[[2,3]]...)

form = @formula beta ~ 1 + state + (1 + state|rat)
lmm = fit(MixedModel,form,df_a)
pvalue(SignedRankTest(Array(groupby(df_a,:state)[1][!,:beta]) .- Array(groupby(df_a,:state)[2][!,:beta])))

all_As = cat(map((x)->x.A,all_models)...,dims=3)
rat_inds = cat(map((x)->zeros(Int,ns,ns).+x,collect(1:nread))...,dims=3)
state1_inds = repeat(permutedims(hcat([repeat([n],ns) for n in 1:ns]...)),1,1,nread)
state2_inds = repeat(permutedims(repeat(1:ns,1,ns)),1,1,nread)
df = DataFrame(rat=string.(rat_inds[:]),state1=state1_inds[:],state2=string.(state2_inds[:]),A=all_As[:])
df_a = groupby(df,:state1)[1]
combine(groupby(df_a,:state2),:A=>median)
df_a = vcat(groupby(df_a,:state2)[[2,3]]...)
form = @formula A ~ 1 + state2 + (1 + state2|rat)
lmm = fit(MixedModel,form,df_a)
pvalue(SignedRankTest(Array(groupby(df_a,:state2)[1][!,:A]) .- Array(groupby(df_a,:state2)[2][!,:A])))


function plot_all_fits(ex_plots,sum_plots,all_agents,all_datas)
    df_all = DataFrame()
    for (model,agents,data,rat_i) in zip(all_models,all_agents,all_datas,1:length(all_models))
        _,_,_,df = plot_gammas(model,agents,data)
        df[!,:rat] .= rat_i
        df_all = vcat(df_all,df)

    end
    df_grp = groupby(df_all,[:state,:trial])
    df_avg = combine(df_grp,:gamma_mean=>mean,:gamma_mean=>sem=>:gamma_mean_sem)
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
    # plot!(p[end-1],left_margin=10mm)
    # plot!(p[end],left_margin=5mm)

    # plot!(p[7],margin=0mm)


    return p,df_all
end

p = plot_all_fits(ex_plots,sum_plots,all_agents,all_datas)[1]
savefig("/Users/sarah/Library/CloudStorage/Dropbox/Princeton/Manuscripts/MoA-HMM/figures/fig5_ex_and_summary_3state_fits.svg")



function nansem(x)
    return nanstd(x)/sqrt(sum((!).(ismissing.(x))))
end

function plot_learning_rates(ex_plots,sum_plots)
    p1 = ex_plots[4]
    p2 = sum_plots[4] 

    l = @layout [a b]
    p = plot(p1,p2,layout=l,size=(800,500),framestyle=:box,legend=false)
    return p
end

function plot_extras(ex_plots,model,agents,data,all_models,all_agents,all_datas)

    p3 = ex_plots[6]
    xcors_state = zeros(101,3,length(data.sess_inds_free))
    xcors_choice = zeros(101,length(data.sess_inds_free))
    lag_lab = zeros(101,length(data.sess_inds_free))
    session_lab = zeros(101,length(data.sess_inds_free))
    df = plot_gammas(model,agents,data)[3]
    lags = -50:50
    for (i,inds) in enumerate(data.sess_inds_free)
        z1 = groupby(df,:state)[1].gamma[inds]
        z2 = groupby(df,:state)[2].gamma[inds]
        z3 = groupby(df,:state)[3].gamma[inds]
        rp = data.leftprobs[data.forced .!= 1][inds]
        cp = data.choices[data.forced .!= 1][inds] .== 1

        # lags = -50:50
        xcors_state[:,:,i] = [crosscor(rp,z1,lags) crosscor(Int.(rp),z2,lags) crosscor(Int.(rp),z3,lags)]
        lag_lab[:,i] = lags
        session_lab[:,i] .= i 
        xcors_choice[:,i] = crosscor(rp,cp,lags)
    end

    df_cors = DataFrame(lag=lag_lab[:],session=session_lab[:],z1=xcors_state[:,1,:][:],z2=xcors_state[:,2,:][:],z3=xcors_state[:,3,:][:],choice=xcors_choice[:])
    df_avg = combine(groupby(df_cors,:lag),:choice=>nanmean,:choice=>nansem)
    peak = argmax(abs.(df_avg[!,:choice_nanmean]))
    p4 = plot(df_avg[!,:lag],df_avg[!,:choice_nanmean],ribbon=df_avg[!,:choice_nansem],c=:gray)
    for (i,state) in enumerate([:z1,:z2,:z3])
        df_avg = combine(groupby(df_cors,:lag),state=>nanmean,state=>nansem)
        plot!(df_avg[!,:lag],df_avg[!,Symbol(string(state)*"_nanmean")],ribbon=df_avg[!,Symbol(string(state)*"_nansem")],c=i)
    end
    hline!([0],c=:black)
    vline!([lags[peak]],c=:black,s=:dash)


    df_all = []
    for r = 1:20
        xcors_state = zeros(101,3,length(all_datas[r].sess_inds_free))
        xcors_choice = zeros(101,length(all_datas[r].sess_inds_free))
        lag_lab = zeros(101,length(all_datas[r].sess_inds_free))
        session_lab = zeros(101,length(all_datas[r].sess_inds_free))
        df = plot_gammas(all_models[r],all_agents[r],all_datas[r])[3]
        lags = -50:50
        for (i,inds) in enumerate(all_datas[r].sess_inds_free)
            z1 = groupby(df,:state)[1].gamma[inds]
            z2 = groupby(df,:state)[2].gamma[inds]
            z3 = groupby(df,:state)[3].gamma[inds]
            rp = all_datas[r].leftprobs[all_datas[r].forced .!= 1][inds]
            if all_datas[r].p_congruent > 0.5
                rp = all_datas[r].leftprobs[all_datas[r].forced .!= 1][inds]
            else
                rp = all_datas[r].leftprobs[all_datas[r].forced .!= 1][inds] .== 0
            end
            cp = all_datas[r].choices[all_datas[r].forced .!= 1][inds] .== 1

            # lags = -50:50
            xcors_state[:,:,i] = [crosscor(rp,z1,lags) crosscor(Int.(rp),z2,lags) crosscor(Int.(rp),z3,lags)]
            lag_lab[:,i] = lags
            session_lab[:,i] .= i 
            xcors_choice[:,i] = crosscor(rp,cp,lags)
        end

        df_cors = DataFrame(lag=lag_lab[:],session=session_lab[:],z1=xcors_state[:,1,:][:],z2=xcors_state[:,2,:][:],z3=xcors_state[:,3,:][:],choice=xcors_choice[:])
        df_avg = combine(groupby(df_cors,:lag),:choice=>nanmean=>:choice,:z1=>nanmean=>:z1,:z2=>nanmean=>:z2,:z3=>nanmean=>:z3)
        df_avg[!,:rat] .= r
        df_all = vcat(df_all,df_avg)
    end
    df_all = reduce(vcat,df_all)
    df_avg = combine(groupby(df_all,:lag),:choice=>nanmean,:choice=>nansem)
    peak = argmax(abs.(df_avg[!,:choice_nanmean]))
    p5 = plot(df_avg[!,:lag],df_avg[!,:choice_nanmean],ribbon=df_avg[!,:choice_nansem],c=:gray)
    @df df_all plot!(:lag,:choice,group=:rat,alpha=0.3,c=:gray)
    for (i,state) in enumerate([:z1,:z2,:z3])
        df_avg = combine(groupby(df_all,:lag),state=>nanmean,state=>nansem)
        plot!(df_avg[!,:lag],df_avg[!,Symbol(string(state)*"_nanmean")],ribbon=df_avg[!,Symbol(string(state)*"_nansem")],c=i)
        plot!(df_all[!,:lag],df_all[!,Symbol(state)],group=df_all[!,:rat],alpha=0.3,c=i)
    end
    hline!([0],c=:black)
    vline!([lags[peak]],c=:black,s=:dash)


#    l = @layout [a; b c]
#    p = plot(p3,p4,p5,layout=l,size=(800,500),framestyle=:box,legend=false)
   return p3,p4,p5
end

sort_args2 = Dict(:method=>"agent",:a=>5,:pos_to_neg=>false)
sum_plots2,all_models2,all_agents2,all_datas2=plot_model_fit_summary(fit_i;return_plots=true,data_fldr=fldr,sort_states=sort_states,sort_args2...);
# plot_all_fits(ex_plots,sum_plots2,all_agents2,all_datas2)[1]


# p = deepcopy(ex_plots[4])

p1 = plot_learning_rates(ex_plots,sum_plots)
plot!(p1[1],ylims=[0,1.01])
plot!(p1[2],ylims=[0,1.01])
p2,p3,p4 = plot_extras(ex_plots,model,agents,data,all_models2,all_agents,all_datas)
l = @layout [a{0.3h}; b{0.3h}; c d]
p = plot(p1,p2,p3,p4,layout=l,size=(1000,800),framestyle=:box,ylabel="",xlabel="",title="",legend=false)
ylims!(p[4],-0.25,0.65)
ylims!(p[5],-0.25,0.65)
plot!(xtickfontsize=12,ytickfontsize=12)
savefig("/Users/sarah/Library/CloudStorage/Dropbox/Princeton/Manuscripts/MoA-HMM/figures/fig5sup_learning_rates_and_bias.svg")


