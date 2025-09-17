import requests

LOCAL_BACKEND = 'http://0.0.0.0:8000'

def use_api(endpoint: str, options: dict):
    pass

def send_get_request(endpoint, payload=None):
    try:
        route = LOCAL_BACKEND + endpoint
        if payload:
            result = requests.get(
                route,
                params = payload
            )
        else:
            result = requests.get(route)
            
        result.raise_for_status()
        return result.json()
    except requests.RequestException as e:
        return f'Request failed | Details: {e}'
    

def send_post_request(endpoint, payload):
    try:
        route = LOCAL_BACKEND + endpoint
        result = requests.post(
            route,
            json = payload
        )
        result.raise_for_status()
        return result.json()
    except requests.RequestException as e:
        return f'Request failed | Details: {e}'


def send_delete_request(endpoint, payload=None):
    try:
        route = LOCAL_BACKEND + endpoint
        if payload:
            result = requests.delete(
                route,
                params = payload
            )
        else:
            result = requests.delete(route)

        result.raise_for_status()
        return result.json()
    except requests.RequestException as e:
        return f'Request failed | Details: {e}'