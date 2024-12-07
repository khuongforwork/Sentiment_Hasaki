from joblib import load
import pandas as pd
from Reviews import Reviews
from datetime import datetime
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


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
    def get_classification_report_df(self, X_test, y_test):
        y_pred = self.model_rf.predict(X_test)
        report_dict = classification_report(y_test, y_pred, output_dict=True)  # Get dictionary format
        report_df = pd.DataFrame(report_dict).transpose()
        confusion_matrix_rf = confusion_matrix(y_test, y_pred)
        fig, ax=plt.subplots(figsize=(5,5))
        sns.heatmap(confusion_matrix_rf,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        plt.show()
        return report_df, fig
