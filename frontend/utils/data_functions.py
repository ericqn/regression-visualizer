import numpy as np
import pandas as pd
import streamlit as st

def process_csv(csv_file):
    """
    Handles csv file and buttons associated with csv file. If the csv file is valid, will
    prompt the user if they want to display the data using streamlit UI.
    """
    if not csv_file:
        return
    
    try:
        data = pd.read_csv(csv_file)
        
        if len(data.columns) != 2:
            st.error(f'Your data must only include 2 columns. Currently you have {len(data.columns)} columns.')
        elif data.isnull().any().any():
            st.error(f'Some points in the dataset are invalid.')
        else:
            st.success('Data successfully uploaded!')
            if st.button('Display uploaded data?', width='stretch'):
                st.session_state['current_data'] = {
                    'X': data.iloc[:,0],
                    'y': data.iloc[:,1],
                    'added_noise': [0] * len(data.iloc[:,0])
                }
    except Exception as e:
        st.error(e)
        st.error("Error loading csv file.")


def generate_and_update_data(n_samples:int, problem_type:str='linear', keep_noise:bool = False):
    """
    Generates n_samples data points with associated labels used for problem_type regression. 
    Updates st.session_state['current_data'] with the new data and reruns the frontend.
    """
    data_object = generate_regression_data(
        n_samples = n_samples,
        problem_type = problem_type
    )

    if keep_noise and 'current_data' in st.session_state:
        current_noise_values = st.session_state['current_data']['added_noise']

    st.session_state['current_data'] = {
        'X': data_object.get('data'), 
        'y': data_object.get('labels'),
        'added_noise': [0] * len(data_object.get('labels')),
        'problem_type': problem_type
    }

    if keep_noise:
        st.session_state['current_data']['added_noise'] = current_noise_values

    st.rerun()


def function_map(input_x, parameters: list, problem_type: str):
    """
    Maps given data to labels given by the problem type. For example, f(x) = ax^2 + bx + c for
    the quadratic problem type.

    Args
        input_x: The input data points
        parameters (list): List of parameters to incorporate into generated function (i.e. for linear, label y = parameters[0] * x + parameters[1])
        problem_type (str): The type of regression dataset to create (valid args are 'linear', 'quad', 'exp', and 'log')
    Returns:
        The mapping of all the data points (will be the same length as input_x)
    """
    if problem_type == 'linear':
        labels = parameters[0] * input_x + parameters[1]
    elif problem_type == 'quad':
        labels = parameters[0] * (input_x ** 2) + parameters[1] * input_x + parameters[2]
    elif problem_type == 'exp':
        labels = parameters[0] * np.exp(input_x + parameters[1]) + parameters[2]
    elif problem_type == 'log':
        labels = parameters[0] * np.log(parameters[1] * input_x) + parameters[2]
    else:
        raise ValueError(f'Input problem_type of {problem_type} is not supported.')
    
    adjusted_labels = np.nan_to_num(labels, nan=0.0, posinf=1e10, neginf=-1e10)
    return adjusted_labels

def generate_regression_data(
        n_samples: int, 
        problem_type: str = 'linear', 
        bias_term: bool = False
    ):
    """
    Generates a regression problem specified by the user.

    Args:
        n_samples (int): Number of samples to generate
        problem_type (str): A string input specifying linear, quad, exp, or log regression problem.
        bias_term (bool): Whether to add a constant bias term. If true, will generate a random scalar value.

    Returns:
       (dict) Dictionary with keys 'data', 'labels'
    """
    problem_type = problem_type.lower()
    supported_problems = {
        'linear',
        'quad',
        'exp',
        'log'
    }

    if problem_type not in supported_problems:
        raise ValueError('Problem type invalid. Please choose a valid problem type.')
    
    data_points = np.random.uniform(low=-10, high=10, size=n_samples)
    parameters = []
    
    if problem_type == 'linear':
        num_params = 1
    else:
        num_params = 2

    for _ in range(num_params):
        param = np.random.uniform(low=-5, high=5)
        parameters.append(param)
    bias = 0 if not bias_term else np.random.uniform(low=-10, high=10)
    parameters.append(bias)

    # special settings for logarithmic since domain is limited
    if problem_type == 'log':
        data_points = np.random.uniform(low=0.1, high=10, size=n_samples)
        parameters[0] = np.random.uniform(low=-15, high=15)
        parameters[1] = np.random.uniform(low=0.01, high=0.25)

    labels = function_map(data_points, parameters, problem_type)

    return {'data': data_points, 'labels': labels}
    
def add_noise(labels, noise:int=0):
    """
    Returns noise constants generated from a uniform distribution with boundaries determined by the noise variable.
    """
    noise_constants = [np.random.uniform(low=-noise, high=noise) for _ in labels]
    adjusted_noise_constants = np.nan_to_num(noise_constants, nan=0.0, posinf=1e10, neginf=-1e10)

    return adjusted_noise_constants


def fit_data(X:np.ndarray, y:np.ndarray, degree:int = 1):
    """
    Fit X data with a polynomial. Default degree is 1 (linear regression model)

    Returns:
        (np.ndarray) An array comprising of the predicted y values of the model given the X inputs
    """
    model_coefs = np.polyfit(X, y, degree)[::-1]
    y_pred = np.zeros(len(y))
    print(f'Raw X values: {X}')
    for deg in range(degree + 1):
        print(f'DEGREE {deg} X values = {model_coefs[deg] * np.pow(X, deg)}')
        y_pred = y_pred + model_coefs[deg] * np.pow(X, deg)
        print(f'y_pred = {y_pred}')

    return y_pred


# --- Calculate loss in Python ---
# Fit linear regression manually using numpy
def avg_residual_error(y: np.ndarray, y_pred: np.ndarray):
    error = np.mean(np.abs(y - y_pred))
    return np.round(error, decimals=6)


# TODO:
def generate_regression_df():
    pass