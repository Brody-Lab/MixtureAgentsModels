function get_presets(preset::Int;seed=nothing)

    model_ops,agent_ops,folder,sim_ops,model_sim,agents_sim = eval(Symbol(string("preset",preset)))(;seed=seed)

    if isnothing(sim_ops)
        return model_ops,agent_ops,folder
    else
        return model_ops,agent_ops,folder,sim_ops,model_sim,agents_sim
    end

end


function preset1(;kwargs...)
    folder = "model_fits"

    nstates = 3
    maxiter = 300
    nstarts = 5
    tol = 1E-5
    model_options = ModelOptionsHMM(nstates=nstates,maxiter=maxiter,tol=tol,nstarts=nstarts)

    # 1. shared LR
    agents = [MBrewardB(), MBchoiceB(), MFrewardB(), MFchoiceB(), Bias()]
    fit_symbs = [:α,:α]
    fit_params = [1,1,2,2,0]
    agent_ops = AgentOptions(agents,fit_symbs,fit_params)

    # 2. split LR
    agents = [MBrewardB(), MBchoiceB(), MFrewardB(), MFchoiceB(), Bias()]
    fit_symbs = [:α,:α,:α,:α]
    fit_params = [1,2,3,4,0]
    agent_ops = vcat(agent_ops,AgentOptions(agents,fit_symbs,fit_params))

    # 3. minus MFr
    agents = [MBrewardB(), MBchoiceB(), MFchoiceB(), Bias()]
    fit_symbs = [:α,:α,:α]
    fit_params = [1,2,3,0]
    agent_options = AgentOptions(agents,fit_symbs,fit_params)
    agent_ops = vcat(agent_ops,AgentOptions(agents,fit_symbs,fit_params))

    # 4. latent alpha (don't use)
    agents = [MBrewardB(), MBchoiceB(), MFrewardB(), MFchoiceB(), Bias()]
    agents = repeat(agents,1,nstates)
    fit_symbs = [:α,:α,:α,:α]
    fit_params = [1,2,3,4,0]
    agent_options = AgentOptions(agents,fit_symbs,fit_params)
    agent_ops = vcat(agent_ops,AgentOptions(agents,fit_symbs,fit_params))

    # 5. reward regressors
    nback = 1:5
    agents = vcat(Bias(),TransReward(nback))
    agent_ops = vcat(agent_ops,AgentOptions(agents))

    # 6. sim agents
    agents = [MBbellmanB(), TD1B(), Bias()]
    fit_symbs = [:α,:α]
    fit_params = [1,2,0]
    agent_ops = vcat(agent_ops,AgentOptions(agents,fit_symbs,fit_params))

    model_ops = repeat([model_options],length(agent_ops))

    # 7. 1-state
    agents = [MBrewardB(), MBchoiceB(), MFrewardB(), MFchoiceB(), Bias()]
    fit_symbs = [:α,:α,:α,:α]
    fit_params = [1,2,3,4,0]
    agent_ops = vcat(agent_ops,AgentOptions(agents,fit_symbs,fit_params))

    nstates = 1
    maxiter = 300
    nstarts = 10
    tol = 1E-5
    model_ops = vcat(model_ops,ModelOptionsHMM(nstates=nstates,nstarts=nstarts,maxiter=maxiter,tol=tol))

    # 8. reward regressors (no bias)
    nback = 1:5
    agents = TransReward(nback)
    agent_ops = vcat(agent_ops,AgentOptions(agents))
    model_ops = vcat(model_ops,model_options)

    # 9. 1-state with extra MBchoiceB
    agents = [MBrewardB(), MBchoiceB(), MFrewardB(), MFchoiceB(), MFchoiceB(), Bias()]
    fit_symbs = [:α,:α,:α,:α,:α]
    fit_params = [1,2,3,4,5,0]
    agent_ops = vcat(agent_ops,AgentOptions(agents,fit_symbs,fit_params))

    nstates = 1
    maxiter = 300
    nstarts = 10
    tol = 1E-5
    model_ops = vcat(model_ops,ModelOptionsHMM(nstates=nstates,nstarts=nstarts,maxiter=maxiter,tol=tol))

    # 10. 4-state
    agents = [MBrewardB(), MBchoiceB(), MFrewardB(), MFchoiceB(), Bias()]
    fit_symbs = [:α,:α,:α,:α]
    fit_params = [1,2,3,4,0]
    agent_ops = vcat(agent_ops,AgentOptions(agents,fit_symbs,fit_params))

    nstates = 4
    maxiter = 300
    nstarts = 5
    tol = 1E-5
    model_ops = vcat(model_ops,ModelOptionsHMM(nstates=nstates,nstarts=nstarts,maxiter=maxiter,tol=tol))


    return model_ops,agent_ops,folder,nothing,nothing,nothing
end

function preset2(;kwargs...)
    folder = "orig_ID_IDfull"

    nstates = 1
    maxiter = 100
    nstarts = 20
    tol = 1E-5
    # L2_penalty = true
    model_options = ModelOptionsHMM(nstates=nstates,maxiter=maxiter,tol=tol,nstarts=nstarts)

    agents = [MBbellman(), TD1(), NoveltyPref(), Persev(), Bias()]
    fit_symbs = [:α,:α]
    fit_params = [1,2,0,0,0]
    agent_options = AgentOptions(agents,fit_symbs,fit_params)
    model_ops,agent_ops = agents_comparison(model_options,agent_options)


    agents = [MBrewardB(), MBchoiceB(), MFrewardB(), MFchoiceB(), Bias()]
    fit_symbs = [:α,:α,:α,:α]
    fit_params = [1,2,3,4,0]
    agent_options = AgentOptions(agents,fit_symbs,fit_params)
    model_op,agent_op = agents_comparison(model_options,agent_options)
    model_ops = vcat(model_ops,model_op)
    agent_ops = vcat(agent_ops,agent_op)

    agents = [MBrewardB(), MBchoiceB(), MFrewardB(), MFchoiceB(), NoveltyPref(), Bias()]
    fit_symbs = [:α,:α,:α,:α]
    fit_params = [1,2,3,4,0,0]
    agent_op = AgentOptions(agents,fit_symbs,fit_params)
    model_ops = vcat(model_ops,model_options)
    agent_ops = vcat(agent_ops,agent_op)

    agents = [MBrewardB(), MBchoiceB(), MFrewardB(), MFchoiceB(), Persev(), Bias()]
    fit_symbs = [:α,:α,:α,:α]
    fit_params = [1,2,3,4,0,0]
    agent_op = AgentOptions(agents,fit_symbs,fit_params)
    model_ops = vcat(model_ops,model_options)
    agent_ops = vcat(agent_ops,agent_op)

    return model_ops,agent_ops,folder,nothing,nothing,nothing
end

function preset3(;kwargs...)
    folder = "absorbed_alpha"

    nstates = 1
    maxiter = 250
    nstarts = 5
    tol = 1E-4
    model_options = ModelOptionsHMM(nstates=nstates,maxiter=maxiter,tol=tol,nstarts=nstarts)
    model_ops = nstates_comparison(model_options,1:3)
    model_ops = repeat(model_ops,2)

    agents = [MBreward(), MBchoice(), MFreward(), MFchoice(), Bias()]
    fit_symbs = [:α,:α,:α,:α]
    fit_params = [1,2,3,4,0]
    agent_options = AgentOptions(agents,fit_symbs,fit_params)
    
    agents = [MBrewardB(), MBchoiceB(), MFrewardB(), MFchoiceB(), Bias()]
    fit_symbs = [:α,:α,:α,:α]
    fit_params = [1,2,3,4,0]
    agent_options_B = AgentOptions(agents,fit_symbs,fit_params)

    agent_ops = [repeat([agent_options],3);repeat([agent_options_B],3)]

    return model_ops,agent_ops,folder,nothing,nothing,nothing

end

function preset4(;seed=nothing,kwargs...)
    folder = "1state_recovery"

    nsess = 25
    ntrials = 10000
    sim_op = TwoStepSim(nsess=nsess,ntrials=ntrials)

    nstarts = 5
    maxiter = 300
    model_op = ModelOptionsHMM(nstates=1,nstarts=nstarts,maxiter=maxiter)

    agents = [MBbellmanB(), TD1B(), Bias()]
    fit_symbs = [:α,:α]
    fit_params = [1,2,0]
    agent_op = AgentOptions(agents,fit_symbs,fit_params)

    data,model_sim,agents_sim = simulate_task(sim_op,model_op,agent_op;seed=seed)


    comp_states = 1:4
    model_ops = nstates_comparison(model_op,comp_states)
    agent_ops = repeat([agent_op],length(model_ops))

    return model_ops,agent_ops,folder,data,model_sim,agents_sim

end

function preset5(;seed=nothing,kwargs...)
    folder = "2state_recovery"

    nsess = 25
    ntrials = 10000
    sim_op = TwoStepSim(nsess=nsess,ntrials=ntrials)

    nstarts = 5
    maxiter = 300
    model_op = ModelOptionsHMM(nstates=2,nstarts=nstarts,maxiter=maxiter)

    agents = [MBbellmanB(), TD1B(), Bias()]
    fit_symbs = [:α,:α]
    fit_params = [1,2,0]
    agent_op = AgentOptions(agents,fit_symbs,fit_params)

    data,model_sim,agents_sim = simulate_task(sim_op,model_op,agent_op;seed=seed)

    comp_states = 1:4
    model_ops = nstates_comparison(model_op,comp_states)
    agent_ops = repeat([agent_op],length(model_ops))

    return model_ops,agent_ops,folder,data,model_sim,agents_sim

end

function preset6(;seed=nothing,kwargs...)
    folder = "3state_recovery"

    nsess = 25
    ntrials = 10000
    sim_op = TwoStepSim(nsess=nsess,ntrials=ntrials)

    nstarts = 5
    maxiter = 300
    model_op = ModelOptionsHMM(nstates=3,nstarts=nstarts,maxiter=maxiter)

    agents = [MBbellmanB(), TD1B(), Bias()]
    fit_symbs = [:α,:α]
    fit_params = [1,2,0]
    agent_op = AgentOptions(agents,fit_symbs,fit_params)

    data,model_sim,agents_sim = simulate_task(sim_op,model_op,agent_op;seed=seed)


    comp_states = 1:4
    model_ops = nstates_comparison(model_op,comp_states)
    agent_ops = repeat([agent_op],length(model_ops))

    return model_ops,agent_ops,folder,data,model_sim,agents_sim

end

function preset7(;seed=nothing,kwargs...)
    folder = "4state_recovery"

    nsess = 25
    ntrials = 10000
    sim_op = TwoStepSim(nsess=nsess,ntrials=ntrials)

    nstarts = 5
    maxiter = 300
    model_op = ModelOptionsHMM(nstates=4,nstarts=nstarts,maxiter=maxiter)

    agents = [MBbellmanB(), TD1B(), Bias()]
    fit_symbs = [:α,:α]
    fit_params = [1,2,0]
    agent_op = AgentOptions(agents,fit_symbs,fit_params)

    data,model_sim,agents_sim = simulate_task(sim_op,model_op,agent_op;seed=seed)

    comp_states = 1:4
    model_ops = nstates_comparison(model_op,comp_states)
    agent_ops = repeat([agent_op],length(model_ops))

    return model_ops,agent_ops,folder,data,model_sim,agents_sim
end

function preset8(;kwargs...)
    folder = "3state_example"
    β0 = [1 0.05 0.1; 0.05 1 -0.1; 0.05 -0.05 0.5]
    # β0 = [-1 -0.05 0.1; 0.05 1 0.1; 0.05 -0.05 0.5]

    π0 = [0.5,0.3,0.2] 
    π0 ./= sum(π0)
    # A0 = [0.994 0.006 0.; 0. 0.996 0.004; 0. 0. 1.]
    # A0 = [0.97 0.02 0.01; 0.02 0.97 0.01; 0.02 0.02 0.96]
    A0 = [0.98 0.01 0.01; 0.01 0.97 0.02; 0.02 0.02 0.96]

    nstarts = 5
    maxiter = 300
    model_sim = ModelHMM(β=β0,π=π0,A=A0)
    model_op = ModelOptionsHMM(nstates=1,nstarts=nstarts,maxiter=maxiter)

    agents_sim = [MBrewardB(α=0.35), MFchoiceB(α=0.65), Bias()]
    # agents_sim = [MBchoiceβ(α=0.35), MBrewardβ(α=0.65), Bias()]
    fit_symbs = [:α,:α]
    fit_params = [1,2,0]
    agent_op = AgentOptions(agents_sim,fit_symbs,fit_params)

    nsess = 50 # 25
    ntrials = 5000 # 10000
    sim_op = TwoStepSim(nsess=nsess,ntrials=ntrials)

    comp_states = 1:4
    model_ops = nstates_comparison(model_op,comp_states)
    agent_ops = repeat([agent_op],length(model_ops))

    return model_ops,agent_ops,folder,sim_op,model_sim,agents_sim

end

function preset9(;kwargs...)
    folder = "agent_comparison_3state"

    nstates = 3
    nstarts = 5
    model_options = ModelOptionsHMM(nstates=nstates,nstarts=nstarts)

    agents = [MBrewardB(), MBchoiceB(), MFrewardB(), MFchoiceB(), Bias()]
    fit_symbs = [:α,:α,:α,:α]
    fit_params = [1,2,3,4,0]
    agent_options = AgentOptions(agents,fit_symbs,fit_params)
    model_ops,agent_ops = agents_comparison(model_options,agent_options)

    return model_ops,agent_ops,folder,nothing,nothing,nothing
end



# function preset5()
#     folder = "ID_share_v_split_LR"

#     nstates = 1
#     maxiter = 1000
#     nstarts = 5
#     tol = 1E-5
#     L2_penalty = true
#     model_options = ModelOptionsHMM(nstates=nstates,maxiter=maxiter,tol=tol,nstarts=nstarts,L2_penalty=L2_penalty)
#     model_ops = repeat([model_options],4)

#     agents = [MBrewardB(), MBchoiceB(), MFrewardB(), MFchoiceB(), Bias()]
#     fit_symbs = [:α,:α]
#     fit_params = [1,1,2,2,0]
#     agent_ops = AgentOptions(agents,fit_symbs,fit_params)


#     fit_symbs = [:α,:α,:α]
#     fit_params = [1,2,3,3,0]
#     agent_ops = vcat(agent_ops,AgentOptions(agents,fit_symbs,fit_params))

#     fit_symbs = [:α,:α,:α]
#     fit_params = [1,1,2,3,0]
#     agent_ops = vcat(agent_ops,AgentOptions(agents,fit_symbs,fit_params))

#     fit_symbs = [:α,:α,:α,:α]
#     fit_params = [1,2,3,4,0]
#     agent_ops = vcat(agent_ops,AgentOptions(agents,fit_symbs,fit_params))


#     return model_ops,agent_ops,folder,nothing,nothing,nothing
# end

