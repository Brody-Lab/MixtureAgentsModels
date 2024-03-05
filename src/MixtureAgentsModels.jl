module MixtureAgentsModels

using MAT, Distributions, Distributed, Optim, ForwardDiff, Parameters, UnPack, CSV
using DataFrames, Plots, StatsPlots, Statistics, StatsBase, Measures, NaNStatistics
using Flatten, LineSearches, LinearAlgebra, SparseArrays, SuiteSparse, SparseDiffTools, SparsityDetection
using FiniteDiff, LogExpFunctions, CategoricalArrays, Plots, Distances, Random, InvertedIndices
using Setfield, Kaleido, Accessors, StatsPlots, JLD2, JSON3, StaticArrays
using RollingFunctions


##-- agents --##
export Agent, AgentOptions
export MBreward, MBchoice, MFreward, MFchoice
export MBrewardB, MBchoiceB, MFrewardB, MFchoiceB
export MBbellman, TD1, TD0, MBbellmanB, TD1B
export TransReward, CR, CO, UR, UO
export Persev, NoveltyPref, Bias, Gambler
export DeltaClicks, DeltaClicksZ, Intercept
# agent functions
export agents_mean, miller2017, venditto2023, twostep_glm
export get_params, get_param, initialize_agent, initialize
export initialize_data, initialize_y, initialize_x
export Q_values, init_Q, next_Q, next_Q!
# agent utils
export agent_strings, βtitle, αtitle, atick, agent_color, agent_color_lite

##-- tasks --##
export RatData, TwoStepData, PClicksData, GenericData
export SimOptions, TwoStepSim, GenericSim
# task functions
export load_twostep, load_pclicks, load_generic
export simulate, split_data, ratdata_tasks

##-- models --##
export ModelOptions, ModelOptionsHMM, ModelOptionsDrift
export MixtureAgentsModel, ModelHMM, ModelDrift
# model functions
export modeltype, mean_model, initialize, get_presets, match_latent_states!, match_states!, sort_model!, compute_evidence
export population_priors, choice_accuracy
# model fitting
export optimize, compute_posteriors, choice_likelihood, model_likelihood
# model analysis
export parameter_recovery, cross_validate, model_compare, nstates_comparison, agents_comparison
# plotting
export plot_model, plot_model!, plot_1state_model, plot_β, compare_β, plot_tr, compare_tr, plot_gammas, plot_model_lite, plot_model_lite!
export plot_recovery, plot_α_recovery, plot_β_recovery, plot_πA_recovery

# conversion functions
export ratdata2dict, model2dict, agents2dict, options2dict, prior2dict
export dict2ratdata, dict2model, dict2agents, dict2options, dict2prior, fit2options
# io functions
export savevars, loadvars, make_fname, savefit, loadfit
# helper functions
export smooth, smthstd, onehot, fill_inverse_σ, DT_Σ_D, βT_x
export deleterow,deletecol,deleterows,deletecols

# debug
# export initialize_y, initialize_x, update_θHMM, vectorize, negloglikelihood, extract, update_agents!, get_fit_params, get_fit_agents, initialize_hypers, leave_out_sessions, get_priors, estimate_prior, moment_α, update_agent,ex_loglikelihood, update_priors!,simulate_model,simulate_agents, marginal_likelihood, get_scales, initialize_data, nfold_sessions, agents2string



abstract type Agent end
abstract type MixtureAgentsModel end
abstract type ModelOptions end
abstract type RatData end
abstract type SimOptions end

include("agent_options.jl")
include("model_options.jl")

include("tasks/generic_task.jl")
include("tasks/twostep_task.jl")
include("tasks/task_utils.jl")
include("tasks/pclicks_task.jl")

include("agents/NoveltyPref.jl")
include("agents/Bias.jl")
include("agents/Intercept.jl")
include("agents/Persev.jl")
include("agents/Gambler.jl")
include("agents/MBreward.jl")
include("agents/MBrewardB.jl")
include("agents/MBchoice.jl")
include("agents/MBchoiceB.jl")
include("agents/MFreward.jl")
include("agents/MFrewardB.jl")
include("agents/MFchoice.jl")
include("agents/MFchoiceB.jl")
include("agents/MBbellman.jl")
include("agents/MBbellmanB.jl")
include("agents/TD0.jl")
include("agents/TD1.jl")
include("agents/TD1B.jl")
include("agents/TransReward.jl")
include("agents/DeltaClicks.jl")
include("agents/DeltaClicksZ.jl")

include("agent_functions.jl")
include("model_priors.jl")

include("HMM/agents_model_HMM.jl")
include("HMM/EM_funcs_HMM.jl")
include("HMM/parameter_recovery_HMM.jl")
include("HMM/plotting_HMM.jl")


include("drift/agents_model_drift.jl")
include("drift/EM_funcs_drift.jl")
include("drift/plotting_drift.jl")


include("utils/io_functions.jl")
include("utils/helper_functions.jl")
include("utils/conversion_functions.jl")
include("utils/cross_validation.jl")
include("utils/model_comparison.jl")
include("utils/model_presets.jl")
include("utils/agent_presets.jl")


end
