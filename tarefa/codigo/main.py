import pickle
import pandas as pd
from fastapi import FastAPI, Response, status

app = FastAPI()
@app.post('/model')
## Coloque seu codigo na função abaixo
def titanic(Sex:int, Age:float, Lifeboat:int, Pclass:int, response:Response):
    with open('model/Titanic.pkl', 'rb') as fid: 
        titanic = pickle.load(fid)

    x_dict = {
                'sex': [Sex],
                'age': [Age],
                'lifeboat': [Lifeboat],
                'pclass': [Pclass]
            }
    
    X = pd.DataFrame(x_dict)
    results = float(titanic.predict(X)[0])

    survived = True if results == 0 else False
    response.status_code = status.HTTP_200_OK

    return {'survived': survived,
            'status': response.status_code,
            'message': 'SUCCESS'}
