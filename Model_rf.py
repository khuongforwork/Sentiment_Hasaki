from joblib import load
import pandas as pd
from Reviews import Reviews
from datetime import datetime


class Model_rf():
    def __init__(self):
        # Load the trained pipeline model
        self.model_rf = load('models/pipeline_rf_model.pkl')
    def transform(self, review: Reviews):
        # Prepare the input data as a DataFrame
        data = {
            'noi_dung_binh_luan_stopword': [review.content],
            'lenght': [review.lenght],
            'time_group': [review.hour_group],
            'day_group': [review.day_group],
            'setiment_label': 'neutral'
        }
        df = pd.DataFrame(data)
        
        # Make a prediction using the entire pipeline
        prediction = self.model_rf.predict(df)
        return prediction[0]  
