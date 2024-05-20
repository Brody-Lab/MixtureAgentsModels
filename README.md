# MixtureAgentsModels
Fit a Mixture-of-Agents Hidden Markov Model (MoA-HMM) described in the preprint <a href=https://www.biorxiv.org/content/10.1101/2024.02.28.582617v1>Dynamic reinforcement learning reveals time-dependent shifts in strategy during reward learning.</a>

Documentation is a work in progress

For questions, email sjvenditto@gmail.com

## Installation
### Option 1: Clone and setup project folder (i.e. julia environment)
From the terminal/command line, clone the repository in the current directory with:
```
git clone https://github.com/Brody-Lab/MixtureAgentsModels/
```
Or <a href=https://github.com/Brody-Lab/MixtureAgentsModels/archive/refs/heads/main.zip>download ZIP</a> and extract in desired directory.

After cloning, also from the terminal/command line, `cd` into the `MixtureAgentsModels` directory and start julia by specifying the current folder as the project location:
```
julia --project=.
```
or, if outside of the `MixtureAgentsModels` directory,
```
julia --project=PATH/TO/YOUR/MixtureAgentsModels
```

Once in julia, enter the package manager by pressing `]` and type the following command to install the package in the project folder:
```julia
pkg > instantiate
```
or, without using the package manager:
```julia
julia > using Pkg
julia > Pkg.instantiate()
```
This will only install the package within the project folder, so to use the package via `using MixtureAgentsModels`, you must first start julia by specifying the project folder as shown above. If you cloned the repository, using the command `git pull` will update the package.

### Option 2: Install as global package
To install the package globally, start julia and enter the package manager by pressing `]`. To install, enter the following command:
```julia
pkg > add https://github.com/Brody-Lab/MixtureAgentsModels/
```
or, without using the package manager:
```julia
julia > using Pkg
julia > Pkg.add(PackageSpec(url="https://github.com/Brody-Lab/MixtureAgentsModels/"))
```
To update the package, enter the package manager using `]` and type the command 
```julia
pkg > update
```

## Getting started
### Fitting the model
See the `example_fit_HMM.jl` script in the <a href=https://github.com/Brody-Lab/MixtureAgentsModels/tree/main/examples>examples</a> directory for example model MoA-HMM fits to the two-step task and command descriptions. 
> NOTE: `example_fit_drift.jl` uses an experimental model that combines a MoA with psytrack (https://github.com/nicholas-roy/psytrack) and should not be used in a serious capacity.

### Loading your own task via `GenericData`
The task data struct `GenericData` contained in `generic_task.jl` in the <a href=https://github.com/Brody-Lab/MixtureAgentsModels/tree/main/src/tasks>tasks</a> directory contains the minimum features necessary to work with model-free agents. The example script `example_load_data.jl` points to two example files (a `.csv` and `.mat`) that can be used as skeletons for loading in your own data, as well as listing compatible agents with the GenericData struct. Additional fields can be added for compatibility with other agents. You may want to fork the repository first if you want to easily commit changes.

### Adding a new agent or task
Documentation for adding your own task or agent/agents is a work in progress. You may want to fork the repository first if you want to easily commit changes.

To add your own agent, see the documentation of `EXAMPLE_agent.jl` in the <a href=https://github.com/Brody-Lab/MixtureAgentsModels/tree/main/src/agents>agents</a> directory for requisite struct fields and functions. Save your new agent as its own julia script in the same `agents` directory.

To add your own task, see the documentation of `generic_task.jl` in the <a href=https://github.com/Brody-Lab/MixtureAgentsModels/tree/main/src/tasks>tasks</a> directory for requisite struct fields and loading examples. Save your new task as its own julia script in the same `tasks` directory.

Any agent or task needs to be exported and its file added to the module definition script `MixtureAgentsModels.jl`. To export a new agent or task, add a line near the existing agents and/or tasks.
```julia
export EXAMPLEAgent # for your agent struct
export EXAMPLEData # for your task struct
```

If you have a custom loading function for your task struct, similarly export that function:
```julia
export load_example_task # or however you named the function
```

Finally, add the scripts containing the code for your new agents and/or tasks near the end of module script near existing agents and tasks:
```julia
include("agents/EXAMPLE_agent.jl")
include("tasks/EXAMPLE_task.jl")
```




