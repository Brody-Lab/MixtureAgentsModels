using MixtureAgentsModels
using UnPack
using DataFrames
using Plots
using StatsPlots
using Statistics, StatsBase
using Measures
using Plots
using MultivariateStats
using Bootstrap
using LogExpFunctions
using LaTeXStrings
using StatsPlots: violinoffsets
using RollingFunctions
using MixtureAgentsModels: marginal_likelihood

function load_model_fit(rat,fit_i;num=nothing,iter=nothing,nfold=nothing,nstates=3,use_exact=false,data_path="/Users/sarah/Library/CloudStorage/Dropbox/data/julia",data_fldr="model_fits",ext=".mat",sort_states=true,sort_args...)
    presets = get_presets(1)
    m_ops = presets[1][fit_i]
    a_ops = presets[2][fit_i]
    if isnothing(nfold)
        fname = make_fname(m_ops,a_ops;num=num,iter=iter,nfold=nfold,ext=ext)
    elseif (nfold == 6) & (fit_i == 2)
        m_ops = Parameters.reconstruct(m_ops,nstates=nstates)
        fname = make_fname(m_ops,a_ops;num=num,iter=iter,nfold=nfold,ext=ext)
    else
        error("unsupported nfold + fit_i combination")
    end
    fname_parts = split(fname,['_','.'])

    if typeof(rat) <: Int
        rat = string("rat",rat)
    end

    fpath = joinpath(data_path,data_fldr,rat)

    if use_exact
        files = filter(x->contains(x,fname), readdir(fpath))
    else
        files = filter(x->(all(contains.(x,fname_parts))), readdir(fpath))
    end


    if !isempty(files)
        lls = zeros(length(files))
        for (i,file) in enumerate(files)
            dat = loadvars(joinpath(fpath,file))
            if isnothing(nfold)
                @unpack model,agents,ratdata = dat
                data = deepcopy(ratdata)
            else
                @unpack model,agents,ratname = dat
                data = load_twostep("data/MBB2017_behavioral_dataset.mat",ratname)
            end
            if typeof(model) <: Dict
                model,model_ops = dict2model(model)
                agents,agent_ops = dict2agents(agents)
                # if typeof(ratdata) <: Dict
                if typeof(data) <: Dict
                    try
                        data = dict2ratdata(ratdata)
                    catch 
                        data = load_twostep("data/MBB2017_behavioral_dataset.mat",data["ratname"])
                    end
                end
                # else
                #     data = deepcopy(ratdata)
                # end
            end
            lls[i] = marginal_likelihood(model,agents,model_ops,agent_ops,data)
            # lls[i] = 1
            # lls[i] = model_likelihood(model,agents,agent_ops,data)

        end
        file_use = files[argmax(lls)]
        dat = loadvars(joinpath(fpath,file_use))
        if isnothing(nfold)
            @unpack model,agents,ratdata = dat
            data = deepcopy(ratdata)
        else
            @unpack model,agents,ratname = dat
            data = load_twostep("data/MBB2017_behavioral_dataset.mat",ratname)
        end
        if typeof(model) <: Dict
            model,model_ops = dict2model(model)
            agents,agent_ops = dict2agents(agents)
            if typeof(data) <: Dict
                try
                    data = dict2ratdata(ratdata)
                catch
                    data = load_twostep("data/MBB2017_behavioral_dataset.mat",data["ratname"])
                end
            end
        end

        if sort_states
            sort_model!(model;sort_args...)

            if size(agents,2) > 1
                agents = agents[:,zord]
            end
        end


        return model,agents,model_ops,agent_ops,data
    else
        error("no file found")
    end
end

function load_model_fits(fit_i;rats_use=nothing,data_path="/Users/sarah/Library/CloudStorage/Dropbox/data/julia",data_fldr="model_fits",kwargs...)
    rats = readdir(joinpath(data_path,data_fldr))
    deleteat!(rats,rats .== ".DS_Store")
    nrats = length(rats)
    all_models = Array{Any}(undef,nrats)
    all_agents = Array{Any}(undef,nrats)
    all_mops = Array{Any}(undef,nrats)
    all_aops = Array{Any}(undef,nrats)
    all_datas = Array{Any}(undef,nrats)
    bads = falses(nrats)

    for (r,rat) in enumerate(rats)
        # println(rat)
        rat_i = parse(Int,rat[4:end])
        if !isnothing(rats_use)
            if rat_i ∉ rats_use
                bads[r] = true
                continue
            end
        end
        # try
        all_models[r],all_agents[r],all_mops[r],all_aops[r],all_datas[r] = load_model_fit(rat,fit_i;data_path=data_path,data_fldr=data_fldr,kwargs...)
        # catch
            # print(rat)
            # bads[r] = true
        # end
    end
    deleteat!(all_models,bads)
    deleteat!(all_agents,bads)
    deleteat!(all_mops,bads)
    deleteat!(all_aops,bads)
    deleteat!(all_datas,bads)
    return all_models,all_agents,all_mops,all_aops,all_datas
end


function plot_model_fit(rat,fit_i=2;num=nothing,iter=nothing,return_plots=false,plot_example=true,sessions=nothing,data_path="/Users/sarah/Library/CloudStorage/Dropbox/data/julia",data_fldr="model_fits",sort_states=true,sort_args...)
    

    # file = "data/MBB2017_behavioral_dataset.mat"
    # varname = "dataset"
    # data = load_twostep(file,rat)
    
    # presets = get_presets(1)
    # m_ops = presets[1][fit_i]
    # a_ops = presets[2][fit_i]
    # fname = make_fname(m_ops,a_ops;num=num,iter=iter)

    # if typeof(rat) <: Int
    #     rat = string("rat",rat)
    # end
    # fpath = joinpath(data_fldr,rat)
    # if isnothing(num)
    #     files = filter(x->(contains(x,fname[1:end-5]) && contains(x,".mat")), readdir(fpath))
    # else
    #     files = filter(x->(contains(x,fname[1:end-3]) && contains(x,".mat")), readdir(fpath))
    # end
    # if !isempty(files)
    #     lls = zeros(length(files))
    #     for (i,file) in enumerate(files)
    #         dat = loadvars(joinpath(fpath,file))
    #         @unpack model,agents = dat
    #         model,_ = dict2model(model)
    #         agents,agent_ops = dict2agents(agents)
    #         _,_,lls[i] = compute_posteriors(model,agents,data)
    #         # lls[i] = model_likelihood(model,agents,agent_ops,data)
    #     end
    #     file_use = files[argmax(lls)]
    #     dat = loadvars(joinpath(fpath,file_use))
    #     @unpack model,agents = dat
    #     model,_ = dict2model(model)
    #     sort_model!(model;sort_args...)
    #     agents,_ = dict2agents(agents)

    model,agents,_,a_ops,data = load_model_fit(rat,fit_i;num=num,iter=iter,data_path=data_path,data_fldr=data_fldr,sort_states=sort_states,sort_args...)
    return plot_model(model,agents,a_ops,data;return_plots=return_plots,plot_example=plot_example,sessions=sessions,sort_states=sort_states,sort_args...),model,agents,data


end



function plot_model_fit_scatter(rat,fit_i=2;data_path="/Users/sarah/Library/CloudStorage/Dropbox/data/julia",data_fldr="model_fits",ext=".mat",sort_args...)
    presets = get_presets(1)
    m_ops = presets[1][fit_i]
    a_ops = presets[2][fit_i]
    ns = m_ops.nstates
    na = size(a_ops.agents,1)
    fname = make_fname(m_ops,a_ops)
    if typeof(rat) <: Int
        rat = string("rat",rat)
    end
    fpath = joinpath(data_path,data_fldr,rat)
    files = filter(x->contains(x,fname[1:end-5])&contains(x,ext)&contains(x,r"i\d"), readdir(fpath))
    nread = length(files)
    all_models = Array{θmodelHMM}(undef,nread)
    all_agents = Array{Any}(undef,nread)
    all_lls = Array{Float64}(undef,nread)
    data = nothing

    for (file_i,file) in enumerate(files)
        model,model_ops,agents,agent_ops,dat,ll = loadfit(joinpath(fpath,file))
        data = dat
        ll = marginal_likelihood(model,agents,model_ops,agent_ops,dat)
        # _,_,ll = compute_posteriors(model,agents,dat)
        # ll = model_likelihood(model,agents,agent_ops,dat)

        all_lls[file_i] = ll #exp(ll / dat.nfree)
        # model,_ = dict2model(dat["model"])
        # agents,_ = dict2agents(dat["agents"])
        
        sort_model!(model;sort_args...)
        all_models[file_i] = model
        all_agents[file_i] = agents

    end

    all_betas = cat(map((x)->x.β,all_models)...,dims=3)
    fit_inds = cat(map((x)->zeros(Int,na,ns).+x,collect(1:nread))...,dims=3)
    # ll_inds = cat(map((x)->zeros(Int,na,ns).+x,all_lls)...,dims=3)
    agent_inds = repeat(repeat(1:na,1,ns),1,1,nread)
    state_inds = repeat(permutedims(repeat(1:ns,1,na)),1,1,nread)
    df = DataFrame(fit=fit_inds[:],state=state_inds[:],agent=agent_inds[:],beta=all_betas[:])
    df_grp = groupby(df,[:agent,:state])
    df_mean = combine(df_grp, :beta=>mean, :beta=>sem)
    βplot = @df df_mean groupedbar(:agent, :beta_mean, group=:state, yerror=:beta_sem, lw=2)
    @df df groupeddotplot!(:agent, :beta, group=:state, color=permutedims(palette(:default)[1:ns]), label=nothing)
    xticks!(1:na,βtitle.(all_agents[1]))
    ylabel!("weight")
    title!("model fit scatter")

    all_pis = cat(map((x)->round.(x.π,digits=4),all_models)...,dims=2)
    fit_inds = cat(map((x)->zeros(Int,ns).+x,collect(1:nread))...,dims=2)
    state_inds = repeat(1:ns,1,nread)
    df = DataFrame(fit=fit_inds[:],state=state_inds[:],pi=all_pis[:])
    df_grp = groupby(df, :state)
    df_avg = combine(df_grp, :pi=>mean, :pi=>sem)
    πplot = @df df_avg bar(:state, :pi_mean, color=palette(:default)[1:ns],yerror=:pi_sem,legend=false,lw=2)
    try
        @df df dotplot!(:state, :pi, color=:black, legend=false)
    catch
        @df df scatter!(:state, :pi, color=:black, legend=false)
    end
    xticks!(1:ns)
    xlabel!("hidden state")
    ylabel!("probability")

    llplot = violin(ones(size(all_lls)),100*(all_lls .- maximum(all_lls)),color=:gray53,legend=false)
    dotplot!(ones(size(all_lls)),100*(all_lls .- maximum(all_lls)),color=:black,legend=false)
    ylabel!("diff from best ll (%)")

    all_params = cat(map((a)->get_params(a,a_ops),all_agents)...,dims=2)
    print(all_params)
    if !(typeof(all_params[1])<:String)
        np = size(all_params,1)
        fit_inds = cat(map((x)->zeros(Int,np).+x,collect(1:nread))...,dims=2)
        agent_inds = repeat(1:np,1,nread)
        df = DataFrame(fit=fit_inds[:],agent=agent_inds[:],param=all_params[:])
        df_grp = groupby(df, :agent)
        df_avg = combine(df_grp, :param=>mean, :param=>sem)
        cm = map((a)->a.color,a_ops.agents[a_ops.symb_inds])
        αplot = @df df_avg bar(:agent, :param_mean, yerror=:param_sem, color=cm, legend=false,lw=2)
        αplot = @df df dotplot!(:agent, :param, color=:black, legend=false)
        xticks!(1:np,αtitle.(a_ops.agents[a_ops.symb_inds]))
        ylabel!("learning rates")
    else 
        αplot = plot()
    end

    _, best_fit = findmax(all_lls)
    best_model = all_models[best_fit]
    best_agents = all_agents[best_fit]

    bestplot = plot_model(best_model,best_agents,a_ops,data;suptitle="best model fit",plot_example=true,sort_args...)


    l = @layout [a; b c d; e{0.75h}]
    return plot(βplot, αplot, πplot, llplot, bestplot, layout=l, framestyle=:box, size=(1500,1500), margin=5mm,legend=false),best_model,best_agents,a_ops,all_lls[best_fit]

end

function bootci(x)
    med_bs = bootstrap(median,x,BasicSampling(10000))
    med_ci = confint(med_bs,BCaConfInt(0.95))
    return (med_ci[1][1]-med_ci[1][2],med_ci[1][3]-med_ci[1][1])
end


function plot_model_fit_summary(fit_i=2;return_plots=false,data_path="/Users/sarah/Library/CloudStorage/Dropbox/data/julia",data_fldr="model_fits",rats_use=nothing,ex_rat=nothing,ext=".mat",sort_states=true,sort_args...)

    all_models,all_agents,all_mops,all_aops,all_datas = load_model_fits(fit_i;rats_use=rats_use,data_path=data_path,data_fldr=data_fldr,ext=ext,sort_states=sort_states,sort_args...)
    a_ops = all_aops[1]
    ns = all_mops[1].nstates
    na = size(a_ops.agents,1)

    nread = length(all_models)

    all_betas = cat(map((x)->x.β,all_models)...,dims=3)
    rat_inds = cat(map((x)->zeros(Int,na,ns).+x,collect(1:nread))...,dims=3)
    agent_inds = repeat(repeat(1:na,1,ns),1,1,nread)
    state_inds = repeat(permutedims(repeat(1:ns,1,na)),1,1,nread)
    data = DataFrame(rat=rat_inds[:],state=state_inds[:],agent=agent_inds[:],beta=all_betas[:])
    data_grp = groupby(data,[:agent,:state])
    data_mean = combine(data_grp, :beta=>median=>:beta_mean, :beta=>bootci=>:beta_sem)
    # βplot = @df data_mean groupedbar(:agent, :beta_mean, group=:state, lw=2, label="", alpha=0.5)
    βplot = @df data_mean groupedbar(:state, :beta_mean, group=:agent, lw=2, c=repeat(collect(1:ns),na)[:], label="", alpha=0.5)
    s = rollmean(collect(range(-0.4,0.4,na+1)),2)
    xt = collect(1:ns) .+ repeat(s,1,ns)'
    xl = permutedims(repeat(atick.(all_agents[1][:,1]),1,ns))
    xticks!(xt[:],xl[:])


    # data_grp = groupby(data,:agent)
    # for (i,grp) in enumerate(data_grp)
    #     if ns == 3
    #         @df grp plot!(repeat(i.+[-0.25,0,0.25],nread),:beta,group=:rat,color=:black,label="",lw=3,alpha=0.1)
    #     else
    #         @df grp plot!(repeat([i],nread),:beta,group=:rat,color=:black,label="",lw=3,alpha=0.1)
    #     end
    # end
    # @df data groupeddotplot!(:agent, :beta, group=:state, color=permutedims(palette(:default)[1:ns]), label=nothing, alpha=1, ms=3, msw=0)
   
   
    data_grp = groupby(data,:state)
    for (i,grp) in enumerate(data_grp)
        if ns > 1
            @df grp plot!(repeat(xt[i,:],nread),:beta,group=:rat,color=:black,label="",lw=3,alpha=0.1)
        else
            @df grp plot!(repeat([i],nread),:beta,group=:rat,color=:black,label="",lw=3,alpha=0.1)
        end
    end
    @df data groupeddotplot!(:state, :beta, group=:agent, color=c=repeat(collect(1:ns)',na*nread)[:], label=nothing, alpha=1, ms=3, msw=0)
    
    


    # data_grp = groupby(data_mean,:agent)
    # if !isnothing(ex_rat)
    #     data_rat = groupby(data,:rat)
    #     data_rat_grp = groupby(data_rat[ex_rat],:agent)
    # end
    # for (i,grp) in enumerate(data_grp)
    #     if ns == 3
    #         @df grp plot!(permutedims(i.+[-0.25,0,0.25]),permutedims(:beta_mean),yerror=permutedims(:beta_sem),color=:black,label="",lw=3)
    #         # @df grp scatter!(permutedims(i.+[-0.25,0,0.25]),permutedims(:beta_mean),color=:black,label="",alpha=0.5)
    #         if !isnothing(ex_rat)
    #             @df data_rat_grp[i] scatter!(permutedims(i.+[-0.25,0,0.25]),permutedims(:beta),color=:black,label="",markershape=:star5,markersize=10)
    #         end
    #     else
    #         @df grp plot!(i.+[0],:beta_mean,yerror=:beta_sem,color=:black,label="",lw=3)
    #         @df grp scatter!(i.+[0],:beta_mean,color=:black,label="",alpha=0.5)
    #         if !isnothing(ex_rat)
    #             @df data_rat_grp[i] scatter!(i,permutedims(:beta),color=:black,label="",markershape=:star5,markersize=10)
    #         end
    #     end
    # end

    data_grp = groupby(data_mean,:state)
    for (i,grp) in enumerate(data_grp)
        if ns > 1
            @df grp scatter!(xt[i,:],:beta_mean,yerror=:beta_sem,color=:black,label="",lw=3)
        else
            @df grp plot!(i.+[0],:beta_mean,yerror=(:beta_sem[1],:beta_sem[2]),color=:black,label="",lw=3)
            @df grp scatter!(i.+[0],:beta_mean,color=:black,label="",alpha=0.5)
            if !isnothing(ex_rat)
                @df data_rat_grp[i] scatter!(i,permutedims(:beta),color=:black,label="",markershape=:star5,markersize=10)
            end
        end
    end

    # @df data_mean groupedbar!(:agent, :beta_mean, group=:state, lw=2, label="", alpha=0.5)

    # xticks!(1:na,atick.(all_agents[1]))
    xlabel!("agent")
    ylabel!("weight")
    title!("model weights")


    all_pis = cat(map((x)->round.(x.π,digits=4),all_models)...,dims=2)
    rat_inds = cat(map((x)->zeros(Int,ns).+x,collect(1:nread))...,dims=2)
    state_inds = repeat(1:ns,1,nread)
    data = DataFrame(rat=rat_inds[:],state=state_inds[:],pi=all_pis[:])
    data_grp = groupby(data, :state)
    data_avg = combine(data_grp, :pi=>median=>:pi_mean, :pi=>bootci=>:pi_sem)
    # πplot = @df data_avg bar(:state, :pi_mean, color=palette(:default)[1:ns],legend=false,lw=2)
    πplot = @df data violin(:state, :pi, group=:state,legend=false,alpha=0.5)
    @df data dotplot!(:state, :pi, group=:state, c=:state, legend=false, ms=3, msw=0)
    # @df data_avg plot!(permutedims(:state),permutedims(:pi_mean),yerror=permutedims(:pi_sem),color=:black,legend=false,lw=3)
    # if !isnothing(ex_rat)
    #     data_rat = groupby(data,:rat)
    #     @df data_rat[ex_rat] dotplot!(:state, :pi, color=:black, markershape=:star5, markersize=10, legend=false)
    # end
    xticks!(1:ns)
    title!("init. state prob.")
    xlabel!("state")
    ylabel!("prob")

    all_As = cat(map((x)->x.A,all_models)...,dims=3)
    rat_inds = cat(map((x)->zeros(Int,ns,ns).+x,collect(1:nread))...,dims=3)
    state1_inds = repeat(permutedims(hcat([repeat([n],ns) for n in 1:ns]...)),1,1,nread)
    state2_inds = repeat(permutedims(repeat(1:ns,1,ns)),1,1,nread)
    df = DataFrame(rat=rat_inds[:],state1=state1_inds[:],state2=state2_inds[:],A=all_As[:])
    df_grp = groupby(df,:state1)
    prows = Array{Any}(undef,ns)
    for (p,grp) in enumerate(df_grp)
        prows[p] = plot()
        @df grp violin!(prows[p],:state2,:A,group=:state2,c=(1:ns)',label="",alpha=0.5)
        @df grp dotplot!(prows[p],:state2,:A,group=:state2,mc=:state2,label="",ms=3,msw=0)
        plot!(yticks=[0,1])
        # annotate!(0.4,0.5,text(p),10)
        ylims!(-0.1,1.1)
        for i in 1:ns-1
            plot!([i+0.5,i+0.5],[-0.1,1.1],c=:black,lw=1,label="")
        end

        if p < ns
            if p == 1
                title!("trans. probs.")
            elseif p == ns-1
                ylabel!("state t-1")
            end
            plot!(xticks=(1:ns,["" for i in 1:ns]))
        else
            plot!(xticks=1:ns,xtickfontsize=10,xlabel="state t") 
        end
    end
    l = @layout grid(ns,1)
    Aplot = plot(prows...,layout=l,framestyle=:box,size=(100*ns,100*ns),margin=0mm)

    all_params = cat(map((a)->get_params(a,a_ops),all_agents)...,dims=2)
    np = size(all_params,1)
    rat_inds = cat(map((x)->zeros(Int,np).+x,collect(1:nread))...,dims=2)
    agent_inds = repeat(1:np,1,nread)
    data = DataFrame(rat=rat_inds[:],agent=agent_inds[:],param=all_params[:])
    data_grp = groupby(data, :agent)
    data_avg = combine(data_grp, :param=>median=>:param_mean, :param=>sem=>:param_sem)
    cm = map((a)->a.color,a_ops.agents[a_ops.symb_inds])
    αplot = @df data_avg bar(:agent, :param_mean, yerror=:param_sem, color=cm, legend=false,lw=2, alpha=0.5)
    @df data dotplot!(:agent, :param, color=:black,legend=false)
    @df data_avg scatter!(:agent,:param_mean,yerror=:param_sem,color=:black,legend=false,lw=5)
    xticks!(1:np,atick.(a_ops.agents[a_ops.symb_inds]))
    title!("model l. rates")
    xlabel!("agent")
    ylabel!("l. rate")

    if return_plots
        return (πplot, Aplot, βplot, αplot), all_models,all_agents,all_datas
    else
        l = @layout [a; [b [c; d]]]
        return plot(βplot, Aplot, πplot, αplot, layout=l, framestyle=:box, size=(700,700), margin=5mm), all_models, all_agents, all_datas, df
    end
end

function plot_state_times(rat_i,session=1::Int;fit_i=2,sort_args...)
    _,model,agents,data = plot_model_fit(rat_i,fit_i;sort_args...);


    inds = data.sess_inds[session]
    starts = data.trial_start[inds]
    ends = data.trial_end[inds]
    rews = data.rewards[inds]
    trans = data.trans_commons[inds]


    # tplot = plot_gammas(model,agents,data;sessions=session,xax="time")[1]
    # p = twinx(tplot)
    # plot!(p,([starts.-starts[1] ends.-starts[1]]./60)',[collect(inds).-(inds[1]-1) collect(inds).-(inds[1]-1)]',lw=8,c=:gray,legend=false)
    # #xlabel!(tplot,"time (minutes)")
    # ylabel!(p,"trial count")

    lplot = plot_gammas(model,agents,data;sessions=session,xax="time")[1]
    # plot!(lplot,alpha=0.5)
    # p = twinx(lplot)
    
    smth_trials = 10
    std_trials = 20

    p11 = plot_gammas(model,agents,data;sessions=session,xax="time")[1]
    # xlabel!("")
    p1 = twinx(p11)
    times = log10.(ends.-starts)
    t = (starts.-starts[1])./60
    plot!(p1,t,smooth(times,smth_trials),ribbon=smthstd(times,std_trials),lw=3,color=:gray33,ms=2,msw=0,label="")
    ylims!(p1,0.1,2)
    ylabel!(p1,L"duration ($log_{10}(s)$)")
    title!("trial duration")


    # scatter!(p,t,times,lw=2,color=:gray33,ms=3,msw=0,label="trial")
    p33 = plot_gammas(model,agents,data;sessions=session,xax="time")[1]
    p3 = twinx(p33)
    times = [log10.(starts[2:end].-ends[1:end-1])...,0]
    i = rews .== 1
    plot!(p3,t[i],smooth(times[i],smth_trials),ribbon=smthstd(times[i],std_trials),lw=3,c=:firebrick,label="",ms=2,msw=0,legend=:top)
    ylims!(p3,0.1,2)
    ylabel!(p3,L"duration ($log_{10}(s)$)")
    title!("post-reward ITI")



    # scatter!(p,t[i],smooth(times[i],smth_trials),lw=2,c=:firebrick,label="ITI(reward)",ms=3,msw=0,legend=:top)
    p22 = plot_gammas(model,agents,data;sessions=session,xax="time")[1]
    # xlabel!("")
    p2 = twinx(p22)
    i = rews .== -1
    plot!(p2,t[i],smooth(times[i],smth_trials),ribbon=smthstd(times[i],std_trials),lw=3,c=:slateblue1,ms=3,msw=0,label="")
    ylims!(p2,0.1,2)
    ylabel!(p2,L"duration ($log_{10}(s)$)")
    title!("post-omission ITI")


    # scatter!(p,t[i],smooth(times[i],smth_trials),lw=2,c=:slateblue1,ms=3,msw=0,label="ITI(omision)")
    
    # times = log10.(ends.-starts)
    # t = (starts.-starts[1])./60
    # plot!(p,t,smooth(times,smth_trials),ribbon=smthstd(times,std_trials),lw=3,color=:gray33,ms=2,msw=0,label="")


    # ylabel!(p2,L"duration ($log_{10}(s)$)")
    # ylims!(p,0.1,1.5)
    # xlabel!("")
    # title!(lplo,"behavioral response times")

    ys = ones(size(collect(inds).-(inds[1]-1)))
    tplot = plot(((starts.-starts[1])./60)',ys',c=:black,shape=:vline,ms=100,yticks=nothing,legend=false)
    ylims!(0.8,1.2)
    # xlabel!(tplot,"time (minutes)")
    # ylabel!("trial times")
    title!("trial times")

    # layout = @layout [A{0.95h};B]
    # return plot(lplot,tplot,layout=layout,framestyle=:box,size=(800,600),margin=10mm)

    # layout = @layout [A{0.025h};B;C;D]
    # return plot(tplot,p11,p22,p33,layout=layout,framestyle=:box,size=(700,1000),left_margin=10mm,right_margin=10mm)

    layout = @layout [B C D]
    return plot(p11,p22,p33,layout=layout,framestyle=:box,size=(1200,200),bottom_margin=10mm,left_margin=10mm,right_margin=10mm)


end

function pca_plots(input)

    _,all_models,_,rat_nums = plot_model_fit_summary(2;method="agent",a=1)
    nread=18
    na = 5
    ns = 3
    all_betas = cat(map((x)->x.β,all_models)...,dims=3)[:,2:3,1:end-1]


    c = distinguishable_colors(nread)
    plotlyjs()
    p = plot([-2,2],[0,0],[0,0],c=:black,label=nothing)
    plot!(p,[0,0],[-2,2],[0,0],c=:black,label=nothing)
    plot!(p,[0,0],[0,0],[-2,2],c=:black,label=nothing)
    xlabel!(p,"PC1")
    ylabel!(p,"PC2")
    zlabel!(p,"PC3")

    if input == 1 || input == 2
        betas_mat = reshape(all_betas,5,:)
        M = fit(PCA,betas_mat;maxoutdim=3)
        proj = Array{Float64}(undef,3,2,nread)

        for r = 1:nread
            proj[:,:,r] = predict(M,all_betas[:,:,r])    
        end

        if input == 1
            for r = 1:nread
                scatter!(p,proj[1,:,r],proj[2,:,r],proj[3,:,r],c=c[r],label=string(rat_nums[r]))
            end

        elseif input == 2
            proj_diff = dropdims(abs.(diff(proj,dims=2)),dims=2)
            for r = 1:nread
                scatter!(p,[proj_diff[1,r]],[proj_diff[2,r]],[proj_diff[3,r]],c=c[r],label=rat_nums[r])
            end
        end

    elseif input == 3
        all_betas_diff = diff(all_betas,dims=2)
        betas_mat = reshape(all_betas_diff,5,:)
        M = fit(PCA,betas_mat;maxoutdim=3)
        proj = Array{Float64}(undef,3,nread)

        for r = 1:nread
            proj[:,r] = predict(M,all_betas_diff[:,:,r])
        end

        for r = 1:nread
            scatter!(p,[proj[1,r]],[proj[2,r]],[proj[3,r]],c=c[r],label=rat_nums[r])
        end
    end
    return plot(p,size=(700,700)),M
end

function plot_1state_weights()

    # gr()
    data_fldr="/Users/sarah/Library/CloudStorage/Dropbox/data/julia/model_fits_prior"
    rats_use=nothing
    fit_i=7

    file = "data/MBB2017_behavioral_dataset.mat"

    presets = get_presets(1)
    m_ops = presets[1][fit_i]
    a_ops = presets[2][fit_i]
    ns = m_ops.nstates
    na = size(a_ops.agents,1)
    fname = make_fname(m_ops,a_ops)


    rats = readdir(data_fldr)
    deleteat!(rats,rats .== ".DS_Store")
    nrats = length(rats)
    all_models = Array{Any}(undef,nrats)
    all_agents = Array{Any}(undef,nrats)

    bads = falses(nrats)
    rat_nums = collect(1:length(rats))
    for (r,rat) in enumerate(rats)
        rat_i = parse(Int,rat[4:end])
        if !isnothing(rats_use)
            if rat_i ∉ rats_use
                bads[rat_i] = true
                continue
            end
        end
        data = load_twostep(file,rat_i)

        fpath = joinpath(data_fldr,rat)
        files = filter(x->contains(x,fname[1:end-5]), readdir(fpath))


        if !isempty(files)
            lls = zeros(length(files))
            for (i,file) in enumerate(files)
                dat = loadvars(joinpath(fpath,file))
                @unpack model,agents = dat
                model,_ = dict2model(model)
                agents,_ = dict2agents(agents)
                # _,_,lls[i] = compute_posteriors(model,agents,data)
                _,_,lls[i] = compute_posteriors(model,agents,data)

            end
            file_use = files[argmax(lls)]
            dat = loadvars(joinpath(fpath,file_use))
            @unpack model,agents = dat
            model,_ = dict2model(model)
            agents,_ = dict2agents(agents)


            all_models[r] = model
            all_agents[r] = agents
        else
            bads[r] = true
        end
    end
    deleteat!(all_models,bads)
    deleteat!(all_agents,bads)
    deleteat!(rat_nums,bads)
    nread = nrats-sum(bads)

    all_betas = cat(map((x)->x.β,all_models)...,dims=3)
    all_betas = dropdims(all_betas,dims=2)

    #βplot = @df data_mean violin(:agent, :beta_mean)

    #permutedims(βtitle.(all_agents[1]))
    # violin([1 2 3 4 5], all_betas', legend=false, fillcolor=:gray,alpha=0.5)
    bar([1 2 3 4 5], median(all_betas,dims=2)', legend=false, fillcolor=:gray,alpha=0.5)

    dotplot!([1 2 3 4 5], all_betas', marker=:gray, label="")
    plot!([0.5,5.5],[0,0],c=:black)
    ylabel!("weight")
    xticks!(1:na,atick.(all_agents[1]))

    meds = median(all_betas,dims=2)
    meds_ci = zeros(5,2)
    for agent_i = 1:5
        med_bs = bootstrap(median,all_betas[agent_i,:], BasicSampling(10000))
        med_ci = confint(med_bs,BCaConfInt(0.95))
        meds_ci[agent_i,1] = meds[agent_i] - med_ci[1][2]
        meds_ci[agent_i,2] = med_ci[1][3] - meds[agent_i]
    end 
    scatter!(meds,yerror=(meds_ci[:,1],meds_ci[:,2]),seriescolor=:black,markersize=6,markerstrokewidth=3,legend=false)

end


function compare_phys(phys_i)

    file_b = "data/MBB2017_behavioral_dataset.mat"
    varname_b = "dataset"

    file_p = "data/ofc_physdata.mat"
    varname_p = "ratdatas"

    phys_rats = [["M055",8],["M064",14],["M070",17],["M071",18]]

    rat_b = phys_rats[phys_i][2]
    data_b = load_twostep(file_b,rat_b)

    rat_p = phys_rats[phys_i][1]
    data_p = load_twostep(file_p,rat_p)

    nstates = 1
    maxiter = 100
    nstarts = 5
    tol = 1E-5
    model_options = modeloptionsHMM(nstates=nstates,tol=tol,maxiter=maxiter,nstarts=nstarts)

    agent_options = venditto2023()
    model_b,agents_b,ll_b = optimize(data_b,model_options,agent_options);
    model_p,agents_p,ll_p = optimize(data_p,model_options,agent_options);

    p1 = compare_β(model_b,model_p,agents_b;labels=["behav." "phys"])
    

    agent_options = twostep_glm()
    model_b,agents_b,ll_b = optimize(data_b,model_options,agent_options);
    model_p,agents_p,ll_p = optimize(data_p,model_options,agent_options);

    p2 = compare_tr(model_b,model_p,agents_b)
    title!(p2,"two-step glm")

    layout = @layout [a; b]
    return plot(p1,p2,layout=layout, size=(600,600))
end


