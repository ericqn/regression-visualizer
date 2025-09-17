# Regression Visualizer

A quick interactive tool to create, train, and visualize regression models using various pre-built algorithms built so that I could get my hands dirty with the tech stack provided below (pure-Python full stack framework).

## Features
- Visualize 2D datasets with interactive Streamlit graphs
- Apply and visualize different various regression curves to fit your data.
- Upload your own CSV data to explore and model
- Save datasets to compare across sessions
- Generate synthetic regression datasets
- Add different settings of noise to see how model performance changes
- Track loss across different regression models.

## Getting Started:
Clone the repository:

`$ git clone https://github.com/ericqn/regression-visualizer.git`

This repository uses uv to handle its dependncies. Install dependencies for from both frontend and backend folders:
```
$ cd frontend
$ uv install
```

```
$ cd backend
$ uv install
```

Run the app from the backend server, then the frontend:
```
$ cd backend
$ uv run server.py
```

```
$ cd frontend
$ streamlit run Regression_Visualizer.py
```

## Tech Stack
- Frontend: Streamlit, requests, altair, numpy, pandas
- Backend: FastAPI, uvicorn, SQLAlchemy, pydantic, pytest, 

See the pyproject.toml files in the frontend and backend folders for a more detailed breakdown of the project requirements.
