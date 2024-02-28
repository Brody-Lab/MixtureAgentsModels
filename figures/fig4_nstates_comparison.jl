using MixtureAgentsModels
using Parameters
using HypothesisTests
using DataFrames
using StatsPlots, Measures
using Bootstrap,StatsBase,Plots,StatsPlots,DataFrames
using StatsPlots: violinoffsets
using MixtureAgentsModels: nfold_sessions


data_fldr = "/Users/sarah/Library/CloudStorage/Dropbox/data/julia/nstates_comparison_val"
nfold = 6
agent_ops = venditto2023()
rat = "rat1"
nstates=1
model_ops = ModelOptionsHMM(nstates=nstates)
fname_search = make_fname(model_ops,agent_ops,nfold=nfold)
files = filter(x->contains(x,fname_search),readdir(joinpath(data_fldr,rat),join=true))
vars = loadvars(files[1])
model_check = dict2model(vars["model"])[1]


function ci(x)
    med_bs = bootstrap(median,x,BasicSampling(10000))
    med_ci = confint(med_bs,BCaConfInt(0.95))
    return (med_ci[1][1]-med_ci[1][2],med_ci[1][3]-med_ci[1][1])
end

function load_nstates_data()
    #data_fldr = "C:/Users/Sarah/Dropbox/julia/data/nstates_comparison"
    data_fldr = "/Users/sarah/Library/CloudStorage/Dropbox/data/julia/nstates_comparison"
    nfold = 6
    agent_ops = venditto2023()

    # nstates_max = 4
    rats = readdir(data_fldr)
    deleteat!(rats,rats .== ".DS_Store")
    deleteat!(rats,rats .== "rat15")
    # deleteat!(rats,rats .== "rat21")
    nrats = length(rats)
    # model_lls = zeros(nrats,nstates_max,2)
    model_lls = Array{Any}(undef,nrats)
    nfold_lls = Array{Any}(undef,nrats)
    ll_plots = Array{Any}(undef,nrats)
    fold_plots = Array{Any}(undef,nrats)
    yacc_all = Array{Any}(undef,nrats)



    for (rat_i,rat) in enumerate(rats)
        # rat = "rat1"
        rat_dat_i = parse(Int,rat[4:end])
        dat_file = "data/MBB2017_behavioral_dataset.mat"
        varname = "dataset"
        # rat = 1
        data = load_twostep(dat_file,rat_dat_i)


        nstates = 1
        model_ops = ModelOptionsHMM(nstates=nstates)
        fname_search = make_fname(model_ops,agent_ops,nfold=nfold)
        state_files = filter(x->contains(x,fname_search[2:end]),readdir(joinpath(data_fldr,rat)))

        nstates_all = map(x->parse(Int,x[1]),state_files)
        model_lls[rat_i] = zeros(length(nstates_all),2)
        nfold_lls[rat_i] = zeros(length(nstates_all),nfold,2)
        yacc_all[rat_i] = zeros(length(nstates_all))


        for nstates in nstates_all
            model_ops = ModelOptionsHMM(nstates=nstates)
            fname_search = make_fname(model_ops,agent_ops,nfold=nfold)
            files = filter(x->contains(x,fname_search),readdir(joinpath(data_fldr,rat),join=true))

            vars = loadvars(files[1])
            @unpack ll_test,ll_train,method,seed,nfold = vars
            println(rat," ",nstates," states")
            model_lls[rat_i][nstates,1] = ll_train
            model_lls[rat_i][nstates,2] = ll_test

            train_set,test_set = nfold_sessions(data;method=method,nfold=nfold,seed=seed)

            ypred = []
            for n = 1:nfold
                fname_search = make_fname(model_ops,agent_ops,nfold=nfold,num=n)
                files = filter(x->contains(x,fname_search),readdir(joinpath(data_fldr,rat),join=true))
                vars = loadvars(files[1])
                @unpack ll_train,train_trials,ll_test,test_trials = vars
                nfold_lls[rat_i][nstates,n,1] = exp(ll_train/train_trials)
                nfold_lls[rat_i][nstates,n,2] = exp(ll_test/test_trials)

                @unpack model,agents = vars
                _,data_test = split_data(data,train_set[n],test_set[n])
                # ypred = vcat(ypred,choice_accuracy(dict2model(model)[1],dict2agents(agents)[1],data_test))
            end

            # yacc_all[rat_i][nstates] = mean(ypred)
        end
        ll_plots[rat_i] = plot(nstates_all,model_lls[rat_i],
            xticks=nstates_all,
            xlabel="nstates",ylabel="norm. log. li.")
        #plot(nstates_all,model_lls[rat_i])

        fold_lls = nfold_lls[rat_i][:,:,2] .- nfold_lls[rat_i][1,:,2]'
        fold_inds = permutedims(repeat(1:nfold,1,length(nstates_all)))
        state_inds = repeat(nstates_all,1,nfold)

        df = DataFrame(fold=fold_inds[:],nstate=state_inds[:],ll=fold_lls[:])
        fold_plots[rat_i] = @df df violin(:nstate,:ll,color=:gray,label="")
        @df df plot!(:nstate,:ll,group=:fold,marker=5,
            xticks=nstates_all,
            xlabel="nstates",
            ylabel="norm. log. li.",
            title="foldwise log. li.")
    end

    return model_lls,yacc_all,rats
end

model_lls,yacc_all,rats = load_nstates_data()

function plot_nstates_comparison(model_lls,rats)
    nrats = length(rats)
    one_lls = [100 .* dat_i[1,2] for dat_i in model_lls]
    p1 = violin(ones(size(one_lls)),one_lls,color=1,alpha=0.5,label="")
    dotplot!(ones(size(one_lls)),one_lls,mc=:black,alpha=0.5,label="")
    scatter!([1],[median(one_lls)],yerror=[ci(one_lls)],color=:black,fc=:gray33,markersize=7,lw=3,label="")
    plot!(ylims=[55,75],yticks=([55,65,75],["55%","65%","75%"]),ytickfontsize=12,xticks=[1],xtickfontsize=12,framestyle=:box)
    xlabel!("single state",xlabelfontsize=16)
    ylabel!("norm. LL",ylabelfontsize=16)


    nstates_max = maximum(size.(model_lls,1))
    lls_train = [100*(dat_i[2:end,1] .- dat_i[1,1]) for (i,dat_i) in zip(1:nrats,model_lls)]
    rat_inds = [repeat([rat],length(ll)) for (rat,ll) in zip(rats,lls_train)]
    state_inds = [collect(2:length(ll)+1) for ll in lls_train]
    lls_test = [100*(dat_i[2:end,2] .- dat_i[1,1]) for (i,dat_i) in zip(1:nrats,model_lls)]

    df = DataFrame(rat=vcat(rat_inds...),nstate=vcat(state_inds...),ll_train=vcat(lls_train...),ll_test=vcat(lls_test...))
    df_stack = stack(df,[:ll_train,:ll_test],variable_name=:type,value_name=:ll)
    p2 = @df df violin(:nstate,:ll_test,c=vcat(repeat([2],192),repeat([3],192),repeat([4],192),repeat([5],192),repeat([6],192),repeat([7],192),repeat([7],192)),alpha=0.5,label="",legend=false)
    plot!([1.5,7.5],[0,0],c=:black,lw=2,xlims=[1.5,7.5])

    df_grp = groupby(df,:nstate)
    offsets = vcat(permutedims.([violinoffsets(0.2,grp.ll_test) for (i,grp) in enumerate(df_grp)])...)[:]
    @df df scatter!(:nstate .+ offsets,:ll_test,mc=:black,label="",alpha=0.5)
    @df df plot!(:nstate .+ offsets,:ll_test,group=:rat,c=:black,label="", alpha=0.5)
    
    df_grp = groupby(df,:nstate)
    println(median(df_grp[2][!,:ll_test] .- df_grp[1][!,:ll_test])) # median difference b/w 2 and 3
    println(median(df_grp[3][!,:ll_test] .- df_grp[2][!,:ll_test])) # median difference b/w 3 and 4

    df_avg = combine(df_grp,:ll_train=>median,:ll_train=>ci,:ll_test=>median,:ll_test=>ci)
    print(df_avg)
    @df df_avg scatter!(:nstate,:ll_test_median,yerror=:ll_test_ci,color=:black,fc=:gray33,markersize=7,lw=3,label="")

    pvals = zeros(nstates_max-1)
    for i in 1:nstates_max-1
        if i == 1
            df_model = Array(groupby(df,:nstate)[i][!,:ll_test])
        else
            df_model = Array(groupby(df,:nstate)[i][!,:ll_test]) .- Array(groupby(df,:nstate)[i-1][!,:ll_test])
        end
        pvals[i] = pvalue(SignedRankTest(df_model))
    end
    for (m,p) in zip(2:nstates_max,pvals)
        annotate!(m,2.4,text(string(round(p,digits=5)),8))
    end


    # @df df_stack groupedboxplot!(:nstate,:ll,group=reverse(:type),lw=2,color=[:black :gray33],fillalpha=0.5,label="")
    xlabel!("number of states",xlabelfontsize=16)
    ylabel!("change in norm. LL",ylabelfontsize=16)
    title!("hidden state cross-validation",titlefontsize=18)
    plot!(ylims=[-0.5,2.6],yticks=([0,1,2],["0%","1%","2%"]),ytickfontsize=12,xtickfontsize=12,legendfontsize=12,framestyle=:box,size=(700,380),margin=5mm)
    # savefig("/Users/sarah/Library/CloudStorage/Dropbox/Princeton/Manuscripts/MoA-HMM/figures/fig3_nstates_comparison.svg")
    layout = @layout [a{0.125w} b]
    return plot(p1,p2,layout=layout,margin=5mm),df
end

plot_nstates_comparison(model_lls,rats)[1]
savefig("/Users/sarah/Library/CloudStorage/Dropbox/Princeton/Manuscripts/MoA-HMM/figures/fig4_nstates_comparison.svg")

