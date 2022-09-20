# Master thesis: Chaotic dynamical systems prediction

The codebase of my master thesis project, based on the research id participated in during my internships at Rice University. The project deals with predicting chaotic dynamical systems, specifically Lorenz 96, using ESN and LSTM network models. It also includes an examination of the impact of using the reduced precision arithmetic for training and evaluating mentioned models.

The master thesis itself (in Serbian language) can bi found in the file `Master thesis - Chaotic dynamical systems prediction.pdf`.

Various reference papers and project presentations can be found in `misc/`.

## Project files, usage and parametrization:

### Model class scripts:

- `script/model.py`: Abstract class representing a model

- `script/model_esn.py`: Class responsible for esn training and prediction

- `script/model_lstm.py`: Class responsible for lstm training and prediction

- `script/model_lstm1.py`: Class responsible for modified LSTM training and prediction, not used in the final project

- `script/model_ensemble_1.py`: Class responsible for ensemble training and prediction, implemented ensemble is the basic averaging model, used ony as a template


### Default parameters:

- `default_esn.json`:
  - File containing default parameters for esn model
  - Loaded by scripts to avoid hard-coding model parameters
  - Useful for keeping different model parameters on different machines, depending on their hardware capability

- `default_lstm.json`:
  - File containing default parameters for lstm model

- `default_ensemble.json`:
  - File containing default parameters for template ensemble model
  - Contains descriptions of all models considered by the ensemble

- `default_run.json`:
  - File containing parameters not specific to one model and not needed for training
  - Contains information on which subset of ground-truth data should be used of validation of a model


### Data sets:

- `data_sets/`:
  - Folder used by system to look for ground truth data-sets
  - All analysis scripts load data using the command: data_set = DataSet('lorenz_96', 200, 1)
  - data_set = DataSet(`<data-set name>`, `<number of time steps per MTU>`, `<st_skp_factor>`)

- `data_sets/lorenz_n/generate_lorenz_n.py`:
  - A simple script for generating a ground truth lorenz dataset

- `lorenz_96, 200, 1`:
  - Folder where all data related to the specific data set is stored
  - Data includes:
    - Dumps of any model trained on the data-set, together with any prediction set generated using it
    - Dumps of analysis scripts generated using given models and data-set
    - Each dump is inside a separate folder, containing the json files describing models used to generate them


### Analysis scripts:

The following scripts are used to generate predictions, statistics and visualizations. All scripts contain a main function, inside which execution parameters are defined.

- `pred_horizon_stats`:
  - Checks if model with requested parameters exist and if not, trains it
  - Checks if prediction data for requested prediction parameters exist and if not, executes the prediction
  - Produces analysis of a prediction
  - Parametrization and usage:
    - Select data set using data_set = DataSet()
    - Make sure default parameters of desired model are set correctly inside `default_<model>.json`
    - Make sure default prediction parameters are set correctly inside `default_run.json`
    - Load default model parameters of the desired model type, uncommenting correct line:
      - `# model_params = Model.default_model_params('esn')`
    - Create the object of desired model, uncommenting correct line:
      - `# model = ModelEsn(model_params, data_set.path_prefix())`
        Run the script
    - Rename generated pred_horizon_### folder to a descriptive name
  - Snipped of ESN Prediction horizons
    </br><img src="lorenz_96%2C%20200%2C%201/pred_horizon_esn/pred_horizons_cut.png" alt="ESN prediction" height="300" style="margin:10px"/>

- `combine_graphs`:
  - Produces comparative analysis from two or more models
  - Parametrization and usage:
    - Select data set using data_set = DataSet()
    - Make sure default parameters of desired models are set correctly inside `default_<model>.json`
    - Make sure default prediction parameters are set correctly inside `default_run.json`
    - Pick from predefined model sets, or create a new one using existing template
    - Select it by assigning it to chosen_models (chosen_models = models1)
    - Run the script
    - Rename generated `combined_graphs_###` folder to a descriptive name
  - ESN-LSTM prediction horizon comparison
    </br><img src="lorenz_96%2C%20200%2C%201/combined_graphs_esn_-_lstm/pred_horizons_cut.png" alt="ESN prediction" height="300" style="margin:10px"/>
  - ESN-ensemble model prediction horizon comparison
    </br><img src="lorenz_96%2C%20200%2C%201/combined_graphs_esn_-_ensemble/pred_horizons_cut.png" alt="ESN prediction" height="300" style="margin:10px"/>

- `color_map`:
  - Produces graphic representation of a prediction from one initial condition for each system variable
  - Iterates over initial conditions and produces animation
  - Parametrization and usage:
    - Select data set using data_set = DataSet()
    - Make sure default parameters of desired model are set correctly inside `default_<model>.json`
    - Make sure default prediction parameters are set correctly inside `default_run.json`
    - Select the model type by setting model_name (model_name = 'esn')
    - Select initial conditions to be drawn (in relation to the prediction parameters):
      - initial_condition = 0 - offset from a first initial condition inside the prediction
      - num_initial_conditions = 10
      - step = 5
    - Run the script
    - Rename generated `color_map_###` folder to a descriptive name
  - Color mapped ESN prediction
    </br><img src="lorenz_96%2C%20200%2C%201/color_map_esn/animation.gif" alt="ESN prediction" height="300" style="margin:10px"/>
  - Color mapped ensemble model prediction
    </br><img src="lorenz_96%2C%20200%2C%201/color_map_ensemble/animation.gif" alt="Model prediction comparison" height="300" style="margin:10px"/>

- `esn_parameter_sweep`:
  - Run esn model training and possibly prediction sets of parameter values to loop through
  - Parametrization and usage:
    - Select data set using data_set = DataSet()
    - Make sure default parameters of desired model are set correctly inside `default_<model>.json`
    - Make sure default prediction parameters are set correctly inside `default_run.json`
    - Initialize the sets of parameters to be varied from the default ones
    - Decided if prediction should be executed after the training (do_predict = True)
    - Run the script

- `ESN slider demo - all variables.ipynb`:
  - Jupyter notebook for interactive prediction horizon demonstration

### Arithmetic precision analysis

See the readme inside the `precision/`.