function plot_model(model::ModelDrift,agents::AbstractArray{A};label_add="") where A<:Agent
    for (a,agent) in zip(1:length(agents),agents)
        if a == 1
            plot(model.β[a,:],lc=agent.color,label=string(βtitle(agent),label_add))
        else
            plot!(model.β[a,:],lc=agent.color,label=string(βtitle(agent),label_add))
        end
    end
    plot!(xlabel="trials",ylabel="weight")
    plot!(size=(1000,500),margin=5mm)
end

function plot_model(model::ModelDrift,agents::AbstractArray{A},data::D;label_add="") where {A<:Agent,D<:RatData}
    for (a,agent) in zip(1:length(agents),agents)
        if a == 1
            plot(model.β[a,:],lc=agent.color,label=string(βtitle(agent),label_add))
        else
            plot!(model.β[a,:],lc=agent.color,label=string(βtitle(agent),label_add))
        end
    end
    plot!(xlabel="trials",ylabel="weight")
    plot!(size=(1000,500),margin=5mm)
end


function plot_model_lite(model::ModelDrift,agents::AbstractArray{A};label_add="") where A<:Agent
    for (a,agent) in zip(1:length(agents),agents)
        if a == 1
            plot(model.β[a,:],lc=agent.color_lite,label=string(βtitle(agent),label_add))
        else
            plot!(model.β[a,:],lc=agent.color_lite,label=string(βtitle(agent),label_add))
        end
    end
    plot!(xlabel="trials",ylabel="weight")
    plot!(size=(1000,500),margin=5mm)
end


function plot_model!(model::ModelDrift,agents::AbstractArray{A};label_add="") where A<:Agent
    for (a,agent) in zip(1:length(agents),agents)
        plot!(model.β[a,:],lc=agent.color,label=string(βtitle(agent),label_add))
    end
    plot!(xlabel="trials",ylabel="weight")
    plot!(size=(1000,500),margin=5mm)
end

function plot_model_lite!(model::ModelDrift,agents::AbstractArray{A};label_add="") where A<:Agent
    for (a,agent) in zip(1:length(agents),agents)
        plot!(model.β[a,:],lc=agent.color_lite,label=string(βtitle(agent),label_add))
    end
    plot!(xlabel="trials",ylabel="weight")
    plot!(size=(1000,500),margin=5mm)
end

function plot_parameter_recovery(model_sim::ModelDrift,model_fit::ModelDrift,agents::AbstractArray{A}) where A<:Agent

    plot_model_lite(model_sim,agents;label_add="sim")
    plot_model!(model_fit,agents;label_add="fit")
    plot!(xlabel="trials",ylabel="weight")
    plot!(size=(1000,500))

end

# function plot_parameter_recovery(model_sim::ModelDrift,model_fit::ModelDrift,agents::AbstractArray{A}) where A<:Agent

#     for (a,agent) in zip(1:length(agents),agents)
#         if a == 1
#             plot(model_sim.β[a,:],lc=agent.color_lite,label=string(βtitle(agent),"sim"))
#         else
#             plot!(model_sim.β[a,:],lc=agent.color_lite,label=string(βtitle(agent),"sim"))
#         end
#         plot!(model_fit.β[a,:],lc=agent.color,label=string(βtitle(agent),"rec"))
#     end
#     plot!(xlabel="trials",ylabel="weight")
#     plot!(size=(1000,500))

# end