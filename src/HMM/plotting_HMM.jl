function plot_model(model::ModelHMM,agents::AbstractArray{A},options::AgentOptions;suptitle=nothing,return_plots=false,sort_states=true,sort_args...) where A<:Agent
    if sort_states
        sort_model!(model; sort_args...)
    end

    if any(typeof.(agents) .<: TransReward) | any(typeof.(agents) .<: Choice) | any(typeof.(agents) .<: Reward)
        nottr = (!).((typeof.(agents) .<: TransReward) .| (typeof.(agents) .<: Choice) .| (typeof.(agents) .<: Reward))
        βplot,_ = plot_tr(model,agents)
    else
        nottr = 0
        βplot = plot_β(model,agents)
    end
    if !isnothing(suptitle)
        title!(suptitle)
    end

    πplot = plot_π(model)

    @unpack fit_params = options
    if !isnothing(fit_params)

        αplot = plot_params(agents,options)
    else 
        if sum(nottr) > 1
            αplot = plot_β(model.β[nottr,:],agents[nottr,1];ylim=(min(model.β...),max(model.β...)))
        elseif sum(nottr) == 1
            αplot = plot_β(reshape(model.β[nottr,:],1,:),agents[nottr,1];ylim=(min(model.β...),max(model.β...)))
        else
            αplot = nothing
        end
    end

    Aplot = plot_A(model)

    if return_plots
        return πplot,Aplot,βplot,αplot
    else
        if !isnothing(αplot)
            l = @layout [a b c{0.5w} d]
            return plot(πplot,Aplot,βplot,αplot, layout=l, framestyle=:box, size=(1700,275), margin=5mm)
        else
            l = @layout [a b c{0.7w}]
            return plot(πplot,Aplot,βplot, layout=l, framestyle=:box, size=(1700,275), margin=5mm)
        end
    end

end

function plot_model(model::ModelHMM,agents::AbstractArray{A},options::AgentOptions,data::D;return_plots=false,plot_example=false,sessions=nothing,sort_args...) where {A<:Agent,D<:RatData}
    πplot,Aplot,βplot,αplot = plot_model(model,agents,options;return_plots=true,sort_args...)
    explot,avgplot,_ = plot_gammas(model,agents,data;sessions=sessions)

    l = @layout [grid(3,1)]
    if return_plots
        return πplot,Aplot,βplot,αplot,avgplot,explot
    elseif plot_example
        if isnothing(αplot)
            l = @layout [a b{0.85w}; d e{0.8w}; f]
            return plot(πplot,βplot,Aplot,avgplot,explot,layout=l,framestyle=:box, size=(900,750), margin=5mm)
        else
            l = @layout [a b{0.6w} c{0.2w}; d e{0.8w}; f]
            return plot(πplot,βplot,αplot,Aplot,avgplot,explot,layout=l,framestyle=:box, size=(900,750), margin=5mm)
        end
    else
        if isnothing(αplot)
            l = @layout [a b{0.85w}; d e{0.8w}]
            return plot(πplot,βplot,Aplot,avgplot,layout=l,framestyle=:box, size=(900,500), margin=5mm)
        else
            l = @layout [a b{0.6w} c{0.2w}; d e{0.8w}]
            return plot(πplot,βplot,αplot,Aplot,avgplot,layout=l,framestyle=:box, size=(900,500), margin=5mm)
        end
    end
end


function plot_1state_model(model::ModelHMM,agents::AbstractArray{A},options::AgentOptions;suptitle=nothing,return_plots=false) where A<:Agent

    if any(typeof.(agents) .<: TransReward)
        nottr = (!).(typeof.(agents) .<: TransReward)
        if sum(nottr) > 1
            βplot = plot_β(model.β[nottr,:],agents[nottr,1])
        elseif sum(nottr) == 1
            βplot = plot_β(reshape(model.β[nottr,:],1,:),agents[nottr,1])
        else
            βplot = nothing
        end
        _,trplot = plot_tr(model,agents)
    else
        βplot = plot_β(model,agents)
        trplot = nothing
    end
    if !isnothing(suptitle)
        title!(suptitle)
    end

    ntop = 1
    if !isnothing(trplot)
        ntop += length(trplot)
        if !isnothing(βplot)
            topplots = [βplot,trplot...]
        else
            topplots = trplot
        end
    else
        topplots = [βplot]
    end

    @unpack fit_params = options
    if !isnothing(fit_params)
        αplot = plot_params(agents,options)
        topplots = [topplots...,αplot]
    else 
        αplot = nothing
    end

    if return_plots
        return βplot,αplot,trplot
    else
        l = @layout [grid(1,ntop+1)]
        return plot(topplots..., layout=l, framestyle=:box, size=(600,275), margin=5mm)
    end
end


function plot_β(model::ModelHMM,agents)
    return plot_β(model.β,agents)
end

function plot_β(β::AbstractMatrix,agents;ylim=nothing)
    ns = size(β,2)
    na = size(β,1)

    agent_inds = repeat(1:na,1,ns)
    state_inds = permutedims(repeat(1:ns,1,na))
    data = DataFrame(state=state_inds[:],agent=agent_inds[:],beta=β[:])
    if ns == 1
        βplot = @df data bar(:agent, :beta, color=:gray53, bar_width=1)
        xticks!(βplot,1:na,atick.(agents[:,1]))
        if !isnothing(ylim)
            ylims!(βplot,ylim)
        end
    else
        βplot = @df data groupedbar(:state, :beta, group=:agent, c=repeat(collect(1:ns)',na)[:], legend=false)
        if !isnothing(ylim)
            ylims!(βplot,ylim)
        end
        s = rollmean(collect(range(-0.4,0.4,na+1)),2)
        xt = collect(1:ns) .+ repeat(s,1,ns)'
        xl = permutedims(repeat(atick.(agents[:,1]),1,ns))
        xticks!(xt[:],xl[:])
        plot!([xt[1]-0.25,xt[end]+0.25],[0,0],color=:black,label=nothing,lw=2)
    end

    # xticks!(βplot,1:na,atick.(agents[:,1]))
    # plot!(legend=false)
    plot!(xrotation=45)
    title!(βplot,"agent weights")
    # ylabel!(βplot,"weight")
    # xlabel!(βplot,"agent")

    return βplot
end

function plot_β(β::AbstractVector,agents)
    na = length(β)
    ns = 1

    agent_inds = repeat(1:na,1,ns)
    state_inds = permutedims(repeat(1:ns,1,na))
    data = DataFrame(state=state_inds[:],agent=agent_inds[:],beta=β[:])
    βplot = @df data groupedbar(:agent, :beta, group=:state, color=:gray53, xticks=atick.(agents))
    plot!(xrotation=45)
    title!(βplot,"agent weights")
    ylabel!(βplot,"weight")

    return βplot
end

function compare_β(model1::ModelHMM,model2::ModelHMM,agents;labels=["1" "2"],sort_args...)
    sort_model!(model1; sort_args...)
    sort_model!(model2; sort_args...)

    β1 = model1.β
    β2 = model2.β
    ns = size(β1,2)
    na = size(β2,1)

    β = [β1 β2]
    agent_inds = repeat(1:na,2,ns)
    state_inds = permutedims(repeat(1:ns,2,na))
    model_inds = [permutedims(repeat([1],ns,na)) permutedims(repeat([2],ns,na))]
    df = DataFrame(state=state_inds[:],agent=agent_inds[:],model=model_inds[:],beta=β[:])
    if ns > 1
        df_g = groupby(df,:state)
        p = Array{Any}(undef,ns)
        for (g,dg) in enumerate(df_g)
            p[g] = @df dg groupedbar(:agent, :beta, group=:model, label=labels)
        end
        xticks!(p[end],1:na,atick.(agents))
        title!(p[1],"agent weights")

        layout = @layout grid(ns,1)
        βplot = plot(p...,layout=layout)

    else
        βplot = @df df groupedbar(:agent, :beta, group=:model, label=labels)
        xticks!(1:na,atick.(agents))
        title!("agent weights")
    end

    return βplot
end


function plot_params(agents,ops)
    params = get_params(agents,ops)
    np = size(params,1)
    cm = map((a)->a.color,agents[ops.symb_inds])
    αplot = bar(1:np, params, color=cm, legend=false)
    xticks!(αplot,1:np,atick.(agents[ops.symb_inds]))
    ylabel!(αplot,"l. rate")
    title!(αplot,"model l. rates")
    xlabel!(αplot,"agent")
    ylims!(αplot,0,1)
    return αplot
end

function plot_π(model)
    ns = length(model.π)
    πplot = bar(1:ns, model.π, color=palette(:default)[1:ns],legend=false)
    xticks!(πplot,1:ns)
    xlabel!(πplot,"hidden state")
    ylabel!(πplot,"prob.")
    title!(πplot,"init. state prob.")
    return πplot
end

function plot_A(model)
    ns = size(model.A,1)
    img_colors = palette(:default)[1:ns]
    img_plt = Array{Any}(undef,ns,ns)
    for i = 1:ns
        for j = 1:ns
            img_plt[j,i] = RGBA(img_colors[i],model.A[j,i])
        end
    end
    Aplot = plot(RGBA.(img_plt), xticks=(collect(1:ns)), yticks=(collect(1:ns)), grid=false, ylabel="state t-1",xlabel="state t")#,title = "trans. mat.")
    for i = 1:ns
        for j = 1:ns
            annotate!(j,i,text(round(model.A[i,j],digits=2),10))
        end
    end
    plot!(xlims=[0.5,ns+0.5],ylims=[0.5,ns+0.5],aspect_ratio=:equal)
    title!(Aplot,"trans. matrix")
    return Aplot
end

function plot_gammas(model,agents,data;sessions::S=nothing,xax="trial") where S <: Union{Nothing,Int,AbstractVector}
    gammas,_,_ = compute_posteriors(model,agents,data)
    ns = length(model.π)
    @unpack sess_inds_free,new_sess_free,forced,leftprobs = data
    trial_inds = []
    sess_inds = []
    for (i,inds) in enumerate(sess_inds_free)
        ntrials = length(inds)

        trial_inds = vcat(trial_inds,collect(1:ntrials))
        sess_inds = vcat(sess_inds,repeat([i],ntrials))    
    end
    state_inds = permutedims(cat(map((x)->zeros(Int,length(trial_inds)).+x,1:ns)...,dims=2))
    trial_inds = permutedims(repeat(trial_inds,1,ns))
    sess_inds = permutedims(repeat(sess_inds,1,ns))

    df = DataFrame(session=sess_inds[:],trial=trial_inds[:],state=state_inds[:],gamma=gammas[:])
    df_grp = groupby(df,[:state,:trial])
    df_avg = combine(df_grp,:gamma=>mean,:gamma=>sem)

    replace!(df_avg.gamma_sem,NaN=>0)
    df_z = groupby(df_avg,:state)
    avgplot = plot()
    for z = 1:ns
        @df df_z[z] plot!(avgplot,:trial,:gamma_mean,ribbon=1.96 .* :gamma_sem, label = "state "*string(z), lw=3, legend=:right)#, legend=false)
    end
    title!(avgplot,"avg. session state prob.")
    xlabel!(avgplot,"trial")
    ylabel!(avgplot,"prob.")

    if isnothing(sessions)
        mid = round(Int,length(sess_inds_free)/2)
        plot_inds = sess_inds_free[max(1,mid-1):min(length(sess_inds_free),mid+2)]
        
    else
        plot_inds = sess_inds_free[sessions]
    end
    plot_inds = reduce(vcat,plot_inds)
    if xax == "trial"
        explot = plot()
        plot!(explot,leftprobs[plot_inds],color=:gray53,style=:dash,lw=3,label="reward flips")
        plot!(explot,gammas[:,plot_inds]',lw=3,c=collect(1:ns)',label="")
        plot!(explot,new_sess_free[plot_inds],color=:black, legend=:outertop,lw=3,label="session break")

        xlabel!(explot,"trial")
        title!(explot,"example state prob.")

    elseif xax == "time"
        times = copy(trial_start) ./ 60
        deleteat!(times,forced)
        explot = plot(times[plot_inds].-times[plot_inds[1]],smooth(gammas[:,plot_inds],10,2)',lw=3,legend=false)
        plot!(explot,times[plot_inds].-times[plot_inds[1]],new_sess_free[plot_inds],color=:black)
        xlabel!(explot,"time (minutes)")
    end
    ylabel!("probability")
    return explot,avgplot,df,df_avg
end

function plot_tr(model,agents;err=nothing)
    ns = length(model.π)
    p =  Array{Any}(undef,ns)
    TR_types = [CR,UR,CO,UO,Reward,Choice]
    i = 1
    ys = (min(model.β...),max(model.β...))

    for type in TR_types
        TR = typeof.(agents) .<: type
        if sum(TR) == 0
            continue
        end
        TR_agents = agents[TR]
        nback = get_param.(TR_agents,:nback)
        TR_β = model.β[TR,:]

        for z = 1:ns
            if i == 1
                p[z] = plot(nback,zeros(size(nback)),color=:black,label=nothing,lw=2)
            end
            # if i==1
            #     if isnothing(err)
            #         p[z] = plot(nback,TR_β[:,z],label=type,color=TR_agents[1].color,lw=2,linestyle=TR_agents[1].line_style,title=string(z))
            #     else
            #         p[z] = plot(nback,TR_β[:,z],label=type,color=TR_agents[1].color,lw=2,linestyle=TR_agents[1].line_style,title=string(z),ribbon=err[TR],fillalpha=0.2)
            #     end
            # else
            if isnothing(err)
                plot!(p[z],nback,TR_β[:,z],label=type,color=TR_agents[1].color,lw=2,linestyle=TR_agents[1].line_style,title=string(z),ylims=ys)
            else
                plot!(p[z],nback,TR_β[:,z],label=type,color=TR_agents[1].color,lw=2,linestyle=TR_agents[1].line_style,title=string(z),ribbon=err[TR],fillalpha=0.2,ylims=ys)
            end

            # end
        end
        i += 1
    end
    l = @layout [grid(1,length(p))]
    return plot(p...,layout=l,xlabel="nback"),p
end


function compare_tr(model1,model2,agents)
    ns = length(model1.π)
    p =  Array{Any}(undef,ns)
    TR_types = [θCR,θUR,θCO,θUO]
    for(i,type) in enumerate(TR_types)
        TR = typeof.(agents) .<: type
        TR_agents = agents[TR]
        nback = get_param.(TR_agents,:nback)
        TR_β1 = model1.β[TR,:]
        TR_β2 = model2.β[TR,:]
        for z = 1:ns
            if i==1
                p[z] = plot(nback,zeros(size(nback)),label=nothing,color=:black,lw=2,title=string(z))
                plot!(p[z],nback,TR_β1[:,z],label=string(type,1),color=TR_agents[1].color,lw=2,linestyle=TR_agents[1].line_style,title=string(z))
                plot!(p[z],nback,TR_β2[:,z],label=string(type,2),color=TR_agents[1].color_lite,lw=2,linestyle=TR_agents[1].line_style,title=string(z))
            else
                plot!(p[z],nback,TR_β1[:,z],label=string(type,1),color=TR_agents[1].color,lw=2,linestyle=TR_agents[1].line_style,title=string(z))
                plot!(p[z],nback,TR_β2[:,z],label=string(type,2),color=TR_agents[1].color_lite,lw=2,linestyle=TR_agents[1].line_style,title=string(z))
            end
        end
    end
    return plot(p...,xlabel="nback")
end


function sort_model!(model::ModelHMM;method="trans",a=nothing,pos_to_neg=true)

    init = model.π
    ns = length(init)
    zord = zeros(Int,ns)
    zord[1] = sortperm(init)[end]
    if method=="trans"
        for z = 2:ns
            zi = setdiff(1:ns,zord[1:z-1])
            zord[z] = zi[argmax(model.A[zord[z-1], zi])]
        end
    elseif method=="agent"
        if isnothing(a)
            ab = model.β[end,:]
        elseif length(a) == 2
            zord[2] = findall(1:ns .!= zord[1])[argmax(model.β[a[1], 1:ns .!= zord[1]])]
            ab = setdiff(1:ns,zord[1:2])[sortperm(model.β[a[2],setdiff(1:ns,zord[1:2])])]
            if pos_to_neg
                zord[3:end] = reverse(ab)
            else
                zord[3:end] = ab
            end
        else
            ab = model.β[a,:]
            if pos_to_neg
                zord[2:end] = reverse(sortperm(ab)[sortperm(ab) .!= zord[1]])
            else
                zord[2:end] = sortperm(ab)[sortperm(ab) .!= zord[1]]
            end
        end

    elseif method=="agent-trans"
        zord[2] = findall(1:ns .!= zord[1])[argmax(model.β[a, 1:ns .!= zord[1]])]
        zrem = setdiff(1:ns,zord[1:2])[sortperm(model.A[zord[2],setdiff(1:ns,zord[1:2])])]
        zord[3:end] = reverse(zrem)
    end
    model.β .= model.β[:,zord]
    model.A .= model.A[zord,zord]
    model.π .= model.π[zord]  

end

function unique_titles(title_str)

    unique_str = unique(title_str)
    if length(unique_str) < length(title_str)
        for str in unique_str
            str_i = findall(title_str .== str)
            if length(str_i) > 1
                for (i,s) in enumerate(str_i)
                    title_str[s] *= "-"*string(i)
                end
            end
        end
    end
    return title_str
end

function plot_recovery(α_recovery,β_recovery,π_recovery,A_recovery,ll_recovery,agent_options;return_plots=false)
    @unpack agents,symb_inds = agent_options
    l = @layout [a ; b ; c d{0.5w} e]
    pa = plot_β_recovery(β_recovery,agents;return_plots=return_plots)
    pb = plot_α_recovery(α_recovery,agents[symb_inds];return_plots=return_plots)
    pc = plot_π_recovery(π_recovery)
    pd = plot_A_recovery(A_recovery;return_plots=return_plots)
    pe = plot_ll_recovery(ll_recovery)
    if return_plots 
        return pa,pb,pc,pd,pe
    else
        return plot(pa,pb,pc,pd,pe, layout = l, size=(550*length(agents),1800), margin=5mm, framestyle=:box)
    end
end

function plot_α_recovery(α_recovery,agents;return_plots=false)
    nα = size(α_recovery,1)
    lmin = 0
    lmax = 1
    p = [scatter(vec(α_recovery[p,1,:]),vec(α_recovery[p,2,:]),markercolor=:Gray,
    legend=false,title=αtitle(agent),
    xlabel="simulated",ylabel="recovered",
    xlabelfontsize=18,ylabelfontsize=18,titlefontsize=20,
    xtickfontsize=18,ytickfontsize=18,
    xlim=[lmin,lmax],ylim=[lmin,lmax],
    xticks=([0,0.5,1]),yticks=([0,0.5,1]),
    aspect_ratio=:equal) for (p,agent) in zip(1:nα,agents)]
    [plot!(p[i],[0,1],[0,1],color=:black) for i in 1:nα]
    if return_plots
        return p
    else
        return plot(p..., layout=(1,nα),size=(250*nα,250))
    end
end

function plot_β_recovery(β_recovery,agents;return_plots=false)
    nβ = size(β_recovery,1)
    lmin = minimum(β_recovery)
    lmax = maximum(β_recovery)
    p = [scatter(vec(β_recovery[p,:,1,:]),vec(β_recovery[p,:,2,:]),markercolor=:Gray,
    legend=false,title=βtitle(agent),
    xlabel="simulated",ylabel="recovered",
    xlabelfontsize=18,ylabelfontsize=18,titlefontsize=20,
    xtickfontsize=18,ytickfontsize=18,
    xlim=[lmin,lmax],ylim=[lmin,lmax],
    aspect_ratio=:equal) for (p,agent) in zip(1:nβ,agents)]
    [plot!(p[i],[lmin,lmax],[lmin,lmax],color=:black) for i in 1:nβ]
    if return_plots
        return p
    else
        return plot(p..., layout=(1,nβ),size=(250*nβ,250))
    end
end

function plot_π_recovery(π_recovery)
    lmin = 0
    lmax = 1
    pπ = scatter(vec(π_recovery[:,1,:]),vec(π_recovery[:,2,:]),markercolor=:Gray,
        legend=false,title="init. state prob.",
        xlabel="simulated",ylabel="recovered",
        xlim=[lmin,lmax],ylim=[lmin,lmax],
        xticks=([0,0.5,1]),yticks=([0,0.5,1]),
        xlabelfontsize=18,ylabelfontsize=18,titlefontsize=20,
        xtickfontsize=18,ytickfontsize=18,
        aspect_ratio=:equal,size=(250,250)) 
    plot!([0,1],[0,1],color=:black)
    return pπ
end

function plot_A_recovery(A_recovery;return_plots=false)
    ns = size(A_recovery,1)
    diag_A = (diag.(eachslice(A_recovery[:,:,1,:],dims=3)),diag.(eachslice(A_recovery[:,:,2,:],dims=3)))
    off_A = ([reduce(vcat,diag.(eachslice(A_recovery[:,:,1,:],dims=3),z)) for z=1:ns-1],[reduce(vcat,diag.(eachslice(A_recovery[:,:,2,:],dims=3),z)) for z=1:ns-1])

    lmin = 0
    lmax = 1
    pA_diag = scatter(reduce(vcat,diag_A[1]),reduce(vcat,diag_A[2]),
        markercolor=:Gray,
        legend=false,title="diag. A",
        xlabel="simulated",ylabel="recovered",
        xlim=[lmin,lmax],ylim=[lmin,lmax],
        xticks=([0,0.5,1]),yticks=([0,0.5,1]),
        xlabelfontsize=18,ylabelfontsize=18,titlefontsize=20,
        xtickfontsize=18,ytickfontsize=18,
        aspect_ratio=:equal) 
    plot!([0,1],[0,1],color=:black)

    pA_off = scatter(reduce(vcat,off_A[1]),reduce(vcat,off_A[2]),
        markercolor=:Gray,
        legend=false,title="off A",
        xlabel="simulated",ylabel="recovered",
        xlim=[lmin,lmax],ylim=[lmin,lmax],
        xticks=([0,0.5,1]),yticks=([0,0.5,1]),
        xlabelfontsize=18,ylabelfontsize=18,titlefontsize=20,
        xtickfontsize=18,ytickfontsize=18,
        aspect_ratio=:equal) 
    plot!([0,1],[0,1],color=:black)
    if return_plots
        return [pA_diag,pA_off]
    else
        plot(pA_diag,pA_off,size=(500,250))
    end
end

function plot_ll_recovery(ll_recovery)
    lmin = 0.5
    lmax = 1
    pll = scatter(vec(ll_recovery[1,:]),vec(ll_recovery[2,:]),markercolor=:Gray,
        legend=false,title="norm. log. li.",
        xlabel="simulated",ylabel="recovered",
        xlim=[lmin,lmax],ylim=[lmin,lmax],
        xticks=([0.5,0.75,1]),yticks=([0.5,0.75,1]),
        xlabelfontsize=18,ylabelfontsize=18,titlefontsize=20,
        xtickfontsize=18,ytickfontsize=18,
        aspect_ratio=:equal) 
    plot(pll)
    plot!(size=(250,250))
end