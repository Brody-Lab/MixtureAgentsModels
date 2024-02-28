using MixtureAgentsModels
using Bootstrap,StatsBase,Plots,StatsPlots,DataFrames,HypothesisTests
using Measures


#fldr1 = "C:/Users/Sarah/Dropbox/julia/data/orig_agents_comparison"
#fldr2 = "C:/Users/Sarah/Dropbox/julia/data/ID_agents_comparison"
function plot_agent_comparison()

    preset = 2
    nfold = 3
    path = "/Users/sarah/Library/CloudStorage/Dropbox/julia/data"
    model_ops,agent_ops,folder = get_presets(preset)
    fldr = joinpath(path,folder)

    rats = readdir(fldr)
    deleteat!(rats,rats.==".DS_Store")
    deleteat!(rats,rats.=="rat20")

    nrats = length(rats)
    nmodels = length(model_ops)
    train_lls = Array{Float64}(undef,nmodels,nrats)
    test_lls =  Array{Float64}(undef,nmodels,nrats)
    model_idx = Array{Int64}(undef,nmodels,nrats)
    fold_idx = Array{Int64}(undef,nmodels,nrats,nfold)
    rat_idx = Array{String}(undef,nmodels,nrats)

    model_fnames = make_fname.(model_ops,agent_ops,nfold=nfold,ext="")
    for rat in rats
        rat_i = parse(Int,rat[4:end])
        if rat_i==21
            rat_i=20
        end
        files = readdir(joinpath(fldr,rat),join=true)
        fnames = readdir(joinpath(fldr,rat))

        for (file,fname) in zip(files,fnames)
        
            #for file in files
            model_i = findfirst(contains.(fname,model_fnames))
            # fname_base = split(model_fnames[model_i],'.')[1]
            if !isnothing(model_i) & !contains(fname,r"n\d")
                # n = parse(Int,match(r"n\d",fname).match[2])
                dat = loadvars(file)
                # train_lls[model_i,rat_i,n] = exp(dat["ll_train"]/dat["train_trials"])
                # test_lls[model_i,rat_i,n] = exp(dat["ll_test"]/dat["test_trials"])
                train_lls[model_i,rat_i] = dat["ll_train"]
                test_lls[model_i,rat_i] = dat["ll_test"]

                model_idx[model_i,rat_i] = model_i
                # fold_idx[model_i,rat_i,n] = n
                rat_idx[model_i,rat_i] = rat
            end
        end
    end

    plot_models = 8:12
    plot_labels = ["-MBr","-MBc","-MFr","-MFc","-Bias"]
    ref_model = 7
    model_ticks = plot_models.-ref_model

    test_lls_diff = copy(test_lls[plot_models,:] .- transpose(test_lls[ref_model,:])) .* 100
    # test_lls_diff = permutedims((permutedims(test_lls[plot_models,:,:],(2,3,1)) .- test_lls[ref_model,:,:]) .* 100,(3,1,2))

    df = DataFrame(rat=string.(rat_idx[plot_models,:][:]), model=model_idx[plot_models,:][:].-ref_model, ll=test_lls_diff[:])
    # df = DataFrame(rat=string.(rat_idx[plot_models,:,:][:]), n=fold_idx[plot_models,:,:][:],model=string.(model_idx[plot_models,:,:][:].-ref_model), ll=test_lls_diff[:])
    p_models = 1:5
    pvals = zeros(length(p_models))
    # form = @eval @formula ll ~ 1 + model + ((1 + model)|rat)
    for (i,m) in enumerate(p_models)
        # df_model = vcat(groupby(df,:model)[1],groupby(df,:model)[m])
        # lmm = fit(MixedModel,form,df_model)
        # pvals[i] = lmm.pvalues[2]
        df_model = groupby(df,:model)[m]
        pvals[i] = pvalue(SignedRankTest(Array(df_model[!,:ll])))
    end

    df_grp = groupby(df,:model)
    function ci(x)
        med_bs = bootstrap(median,x,BasicSampling(10000))
        med_ci = confint(med_bs,BCaConfInt(0.95))
        return (med_ci[1][1]-med_ci[1][2],med_ci[1][3]-med_ci[1][1])
    end
    df_avg = combine(df_grp,:ll=>median,:ll=>ci)
    print(df_avg)
    # print(df_avg)
    # p = @df df violin(:model,:ll,side=:left,color=:gray53,alpha=0.5,label="1-state")
    # @df df dotplot!(:model,:ll,side=:left,color=:gray53,label="")
    # @df df_avg scatter!(:model.-0.15,:ll_median,yerror=:ll_ci,color=:black,markersize=7,lw=3,label="")
   
    #1-state
    # p = @df df violin(:model,:ll,color=:gray53,alpha=0.5,label="1-state")
    # @df df dotplot!(:model,:ll,color=:gray53,label="")
    # @df df_avg scatter!(:model,:ll_median,yerror=:ll_ci,color=:black,markersize=7,lw=3,label="")

   
    plot!([1-0.5,length(plot_models)+0.5],zeros(2),linecolor=:Black,linewidth=2,label="")
    plot!(ylabelfontsize=16,xtickfontsize=16,ytickfontsize=12)
    plot!(ylabel="change in norm. LL (%)",ylabelfontsize=16,yticks=([-10,-6,-2,0,2]))
    plot!(xticks=(collect(model_ticks),plot_labels),xtickfontrotation=45)
    plot!(size=(650,380),margin=5mm,framestyle=:box,xlim=[1-0.5,length(plot_models)+0.5])
    title!("agent cross-validation",titlefontsize=18)
    for (m,pval) in zip(p_models,pvals)
        annotate!(m-0.4,0.5,text(string(round(pval,digits=4)),8,:left))
    end
    ylims!(-7,1)
    # savefig("/Users/sarah/Library/CloudStorage/Dropbox/Princeton/Manuscripts/MoA-HMM/figures/fig3_agents_comparison.svg")





    ### 3-state
    preset = 9
    nfold = 3
    path = "/Users/sarah/Library/CloudStorage/Dropbox/data/julia"
    model_ops,agent_ops,folder = get_presets(preset)
    fldr = joinpath(path,folder)

    rats = readdir(fldr)
    deleteat!(rats,rats.==".DS_Store")
    deleteat!(rats,rats.=="rat6")
    deleteat!(rats,rats.=="rat18")
    deleteat!(rats,rats.=="rat19")




    nrats = length(rats)
    nmodels = length(model_ops)
    train_lls = Array{Float64}(undef,nmodels,nrats)
    test_lls =  Array{Float64}(undef,nmodels,nrats)
    model_idx = Array{Int64}(undef,nmodels,nrats)
    rat_idx = Array{Any}(undef,nmodels,nrats)

    model_fnames = make_fname.(model_ops,agent_ops,nfold=nfold)
    for (rat_i,rat) in enumerate(rats)
        # rat_i = parse(Int,rat[4:end])
        # if rat_i==21
        #     rat_i=20
        # end
        files = readdir(joinpath(fldr,rat),join=true)
        fnames = readdir(joinpath(fldr,rat))

        for (file,fname) in zip(files,fnames)
        
            #for file in files
            model_i = findfirst(model_fnames .== fname)
            if !isnothing(model_i)
                dat = loadvars(file)
                train_lls[model_i,rat_i] = dat["ll_train"]
                test_lls[model_i,rat_i] = dat["ll_test"]
                model_idx[model_i,rat_i] = model_i
                rat_idx[model_i,rat_i] = rat
            end
        end
    end

    plot2_models = 2:6
    # plot_labels = ["-MBr","-MBc","-MFr","-MFc","-Bias"]
    ref2_model = 1

    test_lls_diff = copy(test_lls[plot2_models,:] .- transpose(test_lls[ref2_model,:])) .* 100
    df2 = DataFrame(rat=rat_idx[plot2_models,:][:], model=model_idx[plot2_models,:][:].-ref2_model, ll=test_lls_diff[:])

    # 3-state pvals
    p_models = 1:5
    pvals2 = zeros(length(p_models))
    # form = @eval @formula ll ~ 1 + model + ((1 + model)|rat)
    for (i,m) in enumerate(p_models)
        # df_model = vcat(groupby(df,:model)[1],groupby(df,:model)[m])
        # lmm = fit(MixedModel,form,df_model)
        # pvals[i] = lmm.pvalues[2]
        df_model = groupby(df2,:model)[m]
        pvals2[i] = pvalue(SignedRankTest(Array(df_model[!,:ll])))
    end

    df2_grp = groupby(df2,:model)
    function ci(x)
        med_bs = bootstrap(median,x,BasicSampling(10000))
        med_ci = confint(med_bs,BCaConfInt(0.95))
        return (med_ci[1][1]-med_ci[1][2],med_ci[1][3]-med_ci[1][1])
    end
    df2_avg = combine(df2_grp,:ll=>median,:ll=>ci)
    print(df2_avg)

    # both
    p = @df df violin(:model,:ll,side=:left,color=:gray53,alpha=0.5,label="")
    @df df dotplot!(:model,:ll,side=:left,color=:gray53,label="")
    @df df_avg scatter!(:model.-0.15,:ll_median,yerror=:ll_ci,color=:black,markersize=7,lw=3,label="")
    plot!([plot_models[1]-0.5,plot_models[end]+0.5],zeros(2),linecolor=:Black,linewidth=2,label="")
    plot!(ylabelfontsize=16,xtickfontsize=16,ytickfontsize=12)
    plot!(ylabel="change in norm. LL (%)",ylabelfontsize=16,yticks=([-10,-6,-2,0,2]))
    plot!(xticks=(collect(plot_models),plot_labels),xtickfontrotation=45)
    plot!(size=(650,380),margin=5mm,framestyle=:box,xlim=[plot_models[1]-0.5,plot_models[end]+0.5])
    title!("agent cross-validation",titlefontsize=18)
    ylims!(-7,0.5)

    @df df2 violin!(p,:model,:ll,side=:right,color=3,alpha=0.5,label="3-state")
    @df df2 dotplot!(p,:model,:ll,side=:right,color=3,label="")
    @df df2_avg scatter!(p,:model.+0.15,:ll_median,yerror=:ll_ci,color=:black,markersize=7,lw=3,label="")
    plot!([plot_models[1]-0.5,plot_models[end]+0.5],zeros(2),linecolor=:Black,linewidth=2,label="")
    plot!(xticks=(collect(plot_models),plot_labels),xtickfontrotation=45)
    plot!(size=(650,380),margin=5mm,framestyle=:box,xlim=[plot_models[1]-0.5,plot_models[end]+0.5])
    title!("agent cross-validation",titlefontsize=18)
    ylims!(-7,0.5)

    plot!(ylabelfontsize=16,xtickfontsize=16,ytickfontsize=12)
    plot!(ylabel="change in norm. LL",ylabelfontsize=16,yticks=([-6,-2,0],["-6%", "-2%", "0%"]))

    plot!(xticks=(collect(model_ticks),plot_labels),xtickfontrotation=45)
    plot!(size=(650,380),margin=5mm,framestyle=:box,xlim=[1-0.5,length(plot_models)+0.5])
    title!("agent cross-validation",titlefontsize=18)
    # plot!(size=(650,380))
    for (m,pval,pval2) in zip(p_models,pvals,pvals2)
        annotate!(m+0.1,0.5,text(string(round(pval2,digits=4)),8,:left))
        annotate!(m-0.35,0.5,text(string(round(pval,digits=4)),8,:left))
    end
    ylims!(-7,1)
    

    return p
# ylims!(-7,0.5)
end

p = plot_agent_comparison()
savefig("/Users/sarah/Library/CloudStorage/Dropbox/Princeton/Manuscripts/MoA-HMM/figures/fig4_agents_comparison_w3state.svg")


# pvalue(SignedRankTest(Array(df_grp[3][!,:ll])))
# pvalue(SignedRankTest(Array(df2_grp[3][!,:ll])))
