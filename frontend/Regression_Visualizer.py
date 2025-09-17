import streamlit as st
import numpy as np

import pandas as pd
import altair as alt

from utils.data_functions import (
    process_csv,
    generate_and_update_data,
    add_noise,
    compute_avg_loss
)

from utils.api import send_get_request, send_post_request

hyperparams = {
    'N_SAMPLES': 100,
    'N_FEATURES': 2
}

def save_dataset(dataset_name):
    """
    Handles calling fastapi to save database to backend.
    """
    curr_dataset_details = send_get_request(endpoint='/get-dataset-names')
    current_dataset_names = set(curr_dataset_details.get('names'))

    if not dataset_name:
        st.error('Before saving your dataset, please name it! (Note that duplicate names are not allowed.)')
    elif dataset_name in current_dataset_names:
        st.error(f'[{dataset_name}] is already a name of a saved dataset')
    else:
        adjusted_y = st.session_state['current_data']['y'] + st.session_state['current_data']['added_noise']
        adjusted_y = np.nan_to_num(adjusted_y, nan=0.0, posinf=1e10, neginf=-1e10)
        adjusted_y = adjusted_y.tolist()
        
        data_payload = {
            'name': dataset_name,
            'problem_type': st.session_state['current_data']['problem_type'],
            'data': {
                'X': st.session_state['current_data']['X'].tolist(),
                'y': adjusted_y,
            }
        }
        dataset_details = send_post_request(endpoint='/add-dataset', payload=data_payload)
        name = dataset_details.get('dataset_name')
        
        st.success(f'Your new dataset has been successfully saved as [{name}]!')

def main():
    print('Running Script...')
    st.sidebar.header("Model")
    st.title('Regression Visualizer')
    st.sidebar.text('Model your data here! Adjust the noise setting to see how the model changes with noise!')
    st.set_page_config(
        page_title = 'Model'
    )

    if 'current_data' not in st.session_state:
        generate_and_update_data(hyperparams['N_SAMPLES'])
    
    if 'display_model' not in st.session_state:
        st.session_state['display_model'] = False


    uploaded_csv = st.file_uploader('File Uploader', type=['csv'])
    process_csv(uploaded_csv)

    col_1, col_2 = st.columns([3,1], gap='small', vertical_alignment='center')
    
    with col_1:
        noise_amplifier = st.slider(
            'Noise Setting', 
            min_value = 0.0,
            max_value = 10.0,
            step=0.1
        )

    X = st.session_state['current_data']['X']
    y = st.session_state['current_data']['y']
    
    with col_2:
        if st.button(f'Adjust Noise', width='stretch'):
            st.session_state['current_data']['added_noise'] = add_noise(y, noise = noise_amplifier * abs(max(X)) / 2.5)
    
    noise = st.session_state['current_data']['added_noise']
    adjusted_y = y + noise

    data = {
        'X' : X,
        'y' : adjusted_y
    }

    df = pd.DataFrame(data=data)

    graph = (alt.Chart(data=df)
             .mark_circle()
             .encode(
                 x = alt.X('X', scale = alt.Scale(domain=[-10, 10])), 
                 y = alt.Y('y', scale = alt.Scale(domain=[-25, 25]))
                 )
             .interactive()
             )
    
    regression_line = graph.transform_regression('X', 'y').mark_line(color='azure')
    loss = compute_avg_loss(df)
    

    if st.session_state['display_model']:
        model_data_text = ":green[Modeling Data]"
    else:
        model_data_text = "Model Data"
    
    if st.button(model_data_text, width='stretch'):
        st.session_state['display_model'] = not st.session_state['display_model']
        st.rerun()

    if st.session_state['display_model']:
        st.altair_chart(graph + regression_line)
        st.markdown(f"<h5 style='text-align: center;'>Current Loss: {loss}</h1>", unsafe_allow_html=True)
    else:
        st.altair_chart(graph)
    
    col_1_1, col_1_2 = st.columns(2, gap='small', vertical_alignment='center')
    with col_1_1:
        user_input = st.selectbox(
            'Choose a regression problem type',
            ('Linear', 'Quadratic', 'Exponential', 'Logarithmic')
        )
        user_input_to_function_input = {
            'Linear': 'linear', 
            'Quadratic': 'quad', 
            'Exponential': 'exp', 
            'Logarithmic': 'log'
        }
        problem_type = user_input_to_function_input[user_input]
        if st.button('Generate New Random Data', width='stretch'):
            generate_and_update_data(
                hyperparams['N_SAMPLES'], 
                problem_type=problem_type, 
                keep_noise=True
            )
    
    with col_1_2:
        dataset_name = st.text_input(
            'Name of the current dataset', 
            placeholder="Dataset Name", 
            label_visibility='hidden'
        )
        if st.button('Save Dataset', width='stretch'):
            save_dataset(dataset_name)

if __name__ == '__main__':
    main()