using MixtureAgentsModels
using Measures
using Plots
using Parameters
using StatsBase


function plot_state_rectangles!(p,z,z_1h,inds)
    rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
    for i in unique(z)
        o = findall(diff(z_1h[i,inds]) .== 1)
        f = findall(diff(z_1h[i,inds]) .== -1)
        if !isempty(f) || !isempty(o)
            if (length(f) > length(o)) 
                o = vcat(1,o)
            end
            if (length(o) > length(f)) 
                f = vcat(f,length(z[inds]))
            end
            if (o[1] > f[1]) && (o[end] > f[end])
                o = vcat(1,o)
                f = vcat(f,length(z[inds]))
            end
        end

        for r = 1:length(o)
            plot!(p,rectangle(f[r]-o[r],6,o[r],-3),alpha=0.2,linewidth=0,c=i,label="")
        end
    end
end

function get_plot_data()
    model_ops,agent_ops,_,sim_op,model_3s,agents_3s = get_presets(8)
    seed = 17040
    data_3s,z = simulate(sim_op,model_3s,agents_3s;return_z=true,seed=seed)
    model_3sr,agents_3sr,_ = optimize(data_3s,ModelOptionsHMM(nstates=3),agent_ops[1];disp_iter=20)
    model_1s,agents_1s,_ = optimize(data_3s,model_ops[1],agent_ops[1])

    return data_3s,z,model_3s,agents_3s,model_3sr,agents_3sr,model_1s,agents_1s,agent_ops
end

data_3s,z,model_3s,agents_3s,model_3sr,agents_3sr,model_1s,agents_1s,agent_ops = get_plot_data()

function plot_simulated_models(session,data_3s,z,model_3s,agents_3s,model_3sr,agents_3sr,model_1s,agents_1s,agent_ops)

    yrange = [-0.3,1.3]
    # 3-state model plot
    p11,p12,p13,p14 = plot_model(model_3s,agents_3s,agent_ops[1];return_plots=true,sort_states=false)

    plot!(p11,title="init. state",ylabel="probability")
    plot!(p13,title="3-state agent weights",ylims=yrange,yticks=[0,0.5,1])
    plot!(p14,title="3-state",ylabel="learning rate",yticks=[0,0.5,1])

    # model_3sr,agents_3sr,_ = optimize(data_3s,modeloptionsHMM(nstates=3),agent_ops[1];disp_iter=20)
    plot_model(model_3sr,agents_3sr,agent_ops[1],data_3s;sort_states=false)
    
    match_states!(model_3sr,agents_3sr,model_3s,agents_3s,data_3s)
    p111,p122,p133,p144 = plot_model(model_3sr,agents_3sr,agent_ops[1];return_plots=true,sort_states=false)
    plot!(p111,title="init. state",ylabel="probability")
    plot!(p133,title="3-state agent weights",ylims=yrange,yticks=[0,0.5,1])
    plot!(p144,title="3-state",ylabel="learning rate",yticks=[0,0.5,1])

    # simulated 3-state data
    # seed = 17040
    # data_3s,z = simulate_task(sim_op,model_3s,agents_3s;return_z=true,seed=seed)
 
    # 1-state model fit to 3-state simulated data
    # model_1s,agents_1s,_ = optimize(data_3s,model_ops[1],agent_ops[1])
    p21,p22 = plot_1state_model(model_1s,agents_1s,agent_ops[1];return_plots=true)
    plot!(p21,ylims=yrange,yticks=[0,0.5,1],legend=false,title="1-state")
    plot!(p22,title="1-state",ylabel="learning rate",yticks=[0,0.5,1])


    # session = 25
    # #16,25,33

    # agent values for example session
    inds = data_3s.sess_inds_free[session]
    y,x3 = initialize(data_3s,agents_3s)
    y,x1 = initialize(data_3s,agents_1s)
    y,x3r = initialize(data_3s,agents_3sr)

    # active agent weights for example session
    # g3s,_,_ = compute_posteriors(model_3s,agents_3s,data_3s)
    # z = argmax.(eachcol(g3s))
    z_1h = onehot(z)
    betas = zeros(size(x3))
    for zi = 1:3
        betas[:,z_1h[zi,:]] .= model_3s.β[:,zi]
        # betasr[:,z_1h[zi,:]] .= model_3sr.β[:,zi]
    end
    p31 = plot()
    plot_state_rectangles!(p31,z,z_1h,inds)

    # effective agent values for example session
    # plot!(p31,zeros(length(inds)),color=:black,lw=1,label="")
    plot!(p31,betas[3,inds] .* x3[3,inds],color=:gray,lw=3,label="Q(Bias)",ylabel="value",xlabel="trial",title="3-state effective values (β x Q)")
    plot!(p31,betas[2,inds] .* x3[2,inds],color=agents_3s[2].color,lw=3,label="Q(MF)",ylabel="value",xlabel="trial")
    plot!(p31,betas[1,inds] .* x3[1,inds],color=agents_3s[1].color,lw=3,label="Q(MB)")
    plot!(p31,sum(betas[:,inds] .* x3[:,inds],dims=1)',color=:black,s=:dot,lw=3,label="Q(MB)")
    yrange2 = 3.3
    # xrange
    ylims!(p31,-yrange2,yrange2)
    xlims!(p31,-0.5,101.5)
    # xlims!(1,length(inds))
    # plot!(legend=:bottomright)
    plot!(legend=false)

    l = @layout [a b{0.86w}]
    p2 = plot(p14,p31,layout=l)

    p32 = plot()
    # plot_state_rectangles!(p32,z,z_1h,inds)

    # plot!(p32,zeros(length(inds)),color=:black,lw=1,label="")
    plot!(p32,model_1s.β[3] .* x1[3,inds],color=:gray,lw=3,label="Q(Bias)",ylabel="value",xlabel="trial",title="1-state effective values (β x Q)")
    plot!(p32,model_1s.β[2] .* x1[2,inds],color=agents_3s[2].color,lw=3,label="Q(MF)",ylabel="value",xlabel="trial")
    plot!(p32,model_1s.β[1] .* x1[1,inds],color=agents_3s[1].color,lw=3,label="Q(MB)")
    plot!(p32,sum(model_1s.β .* x1[:,inds],dims=1)',color=:black,s=:dot,lw=3,label="Q(MB)")
    ylims!(p32,-yrange2,yrange2)
    xlims!(p32,-0.5,101.5)

    # xlims!(1,length(inds))
    plot!(legend=false)
    # plot!(twiny(),RGBA.(img_plt),grid=false,ylims=(0,1),xaxis=false)
    l = @layout [a b{0.86w}]
    p3 = plot(p22,p32,layout=l)

    # recovered state
    gs,_,_ = compute_posteriors(model_3sr,agents_3sr,data_3s)
    zr = argmax.(eachcol(gs))
    zr_1h = onehot(zr)
    betasr = zeros(size(x3))
    for zi = 1:3
        betasr[:,zr_1h[zi,:]] .= model_3sr.β[:,zi]
    end

    p33 = plot()
    plot_state_rectangles!(p33,zr,zr_1h,inds)

    plot!(p33,betasr[3,inds] .* x3r[3,inds],color=:gray,lw=3,label="Q(Bias)",ylabel="value",xlabel="trial",title="3-state effective values (β x Q)")
    plot!(p33,betasr[2,inds] .* x3r[2,inds],color=agents_3s[2].color,lw=3,label="Q(MF)",ylabel="value",xlabel="trial")
    plot!(p33,betasr[1,inds] .* x3r[1,inds],color=agents_3s[1].color,lw=3,label="Q(MB)")
    plot!(p33,sum(betasr[:,inds] .* x3r[:,inds],dims=1)',color=:black,s=:dot,lw=3,label="Q(MB)")

    ylims!(p33,-yrange2,yrange2)
    xlims!(p33,-0.5,101.5)

    # xlims!(1,length(inds))
    plot!(legend=false)
    # plot!(twiny(),RGBA.(img_plt),grid=false,ylims=(0,1),xaxis=false)
    l = @layout [a b{0.86w}]
    p4 = plot(p144,p33,layout=l)


    layout = @layout [a b{0.60w}; [[[_ c{0.55w}]; d{0.85h}] e{0.13w} f{0.64275w}]; g h{0.60w}]
    plot!(p11,title="",yticks=[0,0.4],xlabel="",xticks=[],ylabel="")
    plot!(p12,title="",xlabel="",ylabel="")
    p = plot(p13,p2,p11,p12,p21,p3,p133,p4,layout=layout,size=(1500,750),framestyle=:box)
    plot!(margin=5mm,bottom_margin=8mm)
    plot!(p[5],margin=0mm,bottom_margin=2mm)
    plot!(p[4],margin=2mm)
    ylabel!("")
    title!("")
    plot!(xtickfontsize=12,ytickfontsize=12,xlabelfontsize=14,ylabelfontsize=14)   

    return p

end


p = plot_simulated_models(6,data_3s,z,model_3s,agents_3s,model_3sr,agents_3sr,model_1s,agents_1s,agent_ops)
savefig("/Users/sarah/Library/CloudStorage/Dropbox/Princeton/Manuscripts/MoA-HMM/figures/fig3_simulated_model.svg")

