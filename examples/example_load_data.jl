using MixtureAgentsModels
# You can load in your own data either from a .csv or .mat file. See the contents of "data/example_task_data.csv" and "data/example_task_data.mat" for examples of how to format your data for the GenericData struct.
# This struct assumes you have at least the following fields:
# - `choices`: vector of integers representing the choices made on each trial (1=primary choice, 2=secondary choice)
# - `rewards`: vector of integers representing the rewards received on each trial (1=reward, -1=no reward)
# - `new_sess`: vector of booleans representing whether each trial is the start of a new session

# The agents that are compatible with the GenericData struct are:
# - `Intercept`
# - `Bias` (2X the same as Intercept, equivalent to definition used in MoA-HMM manuscript)
# - `MFreward` and `MFrewardB` 
# - `MFchoice` and `MFchoiceB`
# - `TD1` and `TD1B` (equivalent to `MFchoice` and `MFchoiceB`)
# - `Gambler`
# - `Persev`
# - `Reward`
# - `Choice`
# Agents with a second version (e.g. `MFrewardB`) differ by what the learning rate is acting on, leading to differences in value scaling, but are functionally equivalent.
# Explicitly, all 'B' versions only use the learning rate to decay previous values, and the learning rate on reward is absorbed into the agent weight to reduce interactions between the two parameters.

# The example data also contains an additional field that is not being read in. This serves as an example of how to add in additional information to the GenericData struct for your own task.
# To add your own fields, you will need to modify `generic_task.jl` to include the field in the `GenericData` struct, and then modify `load_generic_csv` and/or `load_generic_mat` to read in the field from your data file.

# You can use the `load_generic` function to load in your data, which will work for both .csv and .mat files.
# .csv example load
file = "data/example_task_data.csv"
data = load_generic(file)
# .mat example load
file = "data/example_task_data.mat"
data = load_generic(file)