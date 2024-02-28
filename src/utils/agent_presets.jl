"""
Agents used in Miller et al 2017
"""
function miller2017(nstates=1)
    agents = [MBbellman(),NoveltyPref(),Persev(),Bias()]
    if nstates > 1
        agents = repeat(agents,1,nstates)
    end
    fit_symbs = [:α]
    fit_params = [1,0,0,0]

    return AgentOptions(agents,fit_symbs,fit_params)
end

"""
Agents used in Venditto et al 2023
"""
function venditto2023(nstates=1)
    agents = [MBrewardB(),MBchoiceB(),MFrewardB(),MFchoiceB(),Bias()]
    if nstates > 1
        agents = repeat(agents,1,nstates)
    end
    fit_symbs = [:α,:α,:α,:α]
    fit_params = [1,2,3,4,0]
    return AgentOptions(agents,fit_symbs,fit_params)
end

"""
Two-step task GLM agents
"""
function twostep_glm()
    nback = 1:5
    agents = vcat(Bias(),TransReward(nback))
    return AgentOptions(agents)
end

