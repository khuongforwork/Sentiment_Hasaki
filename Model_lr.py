import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from Reviews import Reviews

class Model_lr():
    def __init__(self):
        # Initialize the Spark session
        spark = SparkSession.builder.appName("Review_Sentiment").getOrCreate()

        # Load the trained pipeline model
        model = PipelineModel.load('models/pipeline_lr_model')


    def transform(self, review: Reviews):

        # Prepare the data
        data = [(review.content, review.lenght, review.hour_group, review.day_group, 'neutral')]
        columns = ['noi_dung_binh_luan_stopword', 'lenght', 'time_group', 'day_group', 'sentiment_label']
        
        # Create a DataFrame
        df = spark.createDataFrame(data, columns)
        
        # Transform the data using the loaded model
        predictions_df = model.transform(df)
        
        # Extract the prediction result
        prediction = predictions_df.select('prediction').collect()[0][0]
        
        return prediction


