using MixtureAgentsModels
using Bootstrap,StatsBase,Plots,StatsPlots,DataFrames,HypothesisTests
using Measures


#fldr1 = "C:/Users/Sarah/Dropbox/julia/data/orig_agents_comparison"
#fldr2 = "C:/Users/Sarah/Dropbox/julia/data/ID_agents_comparison"
function load_agent_comp_data()

    preset = 2
    nfold = 3
    path = "/Users/sarah/Library/CloudStorage/Dropbox/data/julia"
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

    plot_models = vcat(collect(2:7),collect(13:14))
    ref_model = 1
    model_ticks = 1:length(plot_models) #plot_models.-ref_model
    model_idx_shift = repeat(model_ticks,nrats)
    test_lls_diff = copy(test_lls[plot_models,:] .- transpose(test_lls[ref_model,:])) .* 100
    # test_lls_diff = permutedims((permutedims(test_lls[plot_models,:,:],(2,3,1)) .- test_lls[ref_model,:,:]) .* 100,(3,1,2))

    df = DataFrame(rat=string.(rat_idx[plot_models,:][:]), model=model_idx_shift[:], ll=test_lls_diff[:])
   return df
end
   
df = load_agent_comp_data()

function plot_agent_comparison(df)
    # df = DataFrame(rat=string.(rat_idx[plot_models,:,:][:]), n=fold_idx[plot_models,:,:][:],model=string.(model_idx[plot_models,:,:][:].-ref_model), ll=test_lls_diff[:])
    plot_models = 1:8
    model_ticks = 1:8
    plot_labels = ["-MB","-MF","-NP","Persev","-Bias","new","+NP","+Persev"]
    c = [:royalblue :royalblue :royalblue :royalblue :royalblue :grey53 :firebrick :firebrick]
    p_models = 1:6
    pvals = zeros(8)
    # form = @eval @formula ll ~ 1 + model + ((1 + model)|rat)
    for (i,m) in enumerate(p_models)
        # df_model = vcat(groupby(df,:model)[1],groupby(df,:model)[m])
        # lmm = fit(MixedModel,form,df_model)
        # pvals[i] = lmm.pvalues[2]
        df_model = groupby(df,:model)[m]
        pvals[m] = pvalue(SignedRankTest(Array(df_model[!,:ll])))
    end

    p_models = 7:8
    # pvals2 = zeros(length(p_models))
    df_comp = groupby(df,:model)[6]
    # form = @eval @formula ll ~ 1 + model + ((1 + model)|rat)
    for (i,m) in enumerate(p_models)
        # df_model = vcat(groupby(df,:model)[1],groupby(df,:model)[m])
        # lmm = fit(MixedModel,form,df_model)
        # pvals[i] = lmm.pvalues[2]
        df_model = groupby(df,:model)[m]
        pvals[m] = pvalue(SignedRankTest(Array(df_comp[!,:ll]),Array(df_model[!,:ll])))
    end

    df_grp = groupby(df,:model)
    function ci(x)
        med_bs = bootstrap(median,x,BasicSampling(10000))
        med_ci = confint(med_bs,BCaConfInt(0.95))
        return (med_ci[1][1]-med_ci[1][2],med_ci[1][3]-med_ci[1][1])
    end
    df_avg = combine(df_grp,:ll=>median,:ll=>ci)
    p = @df df violin(:model,:ll,group=:model,color=c,alpha=0.5,label="1-state")
    @df df dotplot!(:model,:ll,group=:model,color=c,label="")
    @df df_avg scatter!(:model,:ll_median,yerror=:ll_ci,color=:black,markersize=7,lw=3,label="")
    plot!([1-0.5,length(plot_models)+0.5],zeros(2),linecolor=:Black,linewidth=2,label="")
    plot!([1-0.5,length(plot_models)+0.5],[df_avg[6,:ll_median],df_avg[6,:ll_median]],linecolor=:Black,s=:dash,linewidth=2,label="")
    plot!(ylabelfontsize=16,xtickfontsize=16,ytickfontsize=12)
    plot!(ylabel="change in norm. LL (%)",ylabelfontsize=16,yticks=([-10,-6,-4,-2,0,2]))
    plot!(xticks=(collect(model_ticks),plot_labels),xtickfontrotation=45)
    plot!(size=(1000,380),margin=5mm,framestyle=:box,xlim=[1-0.5,length(plot_models)+0.5],legend=false)
    title!("comparison to old agents",titlefontsize=18)
    for (m,pval) in enumerate(pvals)
        annotate!(m,3,text(string(round(pval,digits=4)),8,:left))
    end
    ylims!(-7,3.5)
    # savefig("/Users/sarah/Library/CloudStorage/Dropbox/Princeton/Manuscripts/MoA-HMM/figures/fig3_agents_comparison.svg")    
    yticks!([-10,-6,-4,-2,0,2],["-10%","-6%","-4%","-2%","0%","2%"])
    return p
# ylims!(-7,0.5)
end

p = plot_agent_comparison(df)
savefig("/Users/sarah/Library/CloudStorage/Dropbox/Princeton/Manuscripts/MoA-HMM/figures/fig4sup_agents_comparison_full.svg")


pvalue(SignedRankTest(Array(df_grp[3][!,:ll])))
pvalue(SignedRankTest(Array(df2_grp[3][!,:ll])))
