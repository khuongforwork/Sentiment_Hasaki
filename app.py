import streamlit as st 
from Reviews import Reviews
from datetime import datetime
from Model_rf import Model_rf
# import transform_with_lr as model_lr


model_rf = Model_rf()


st.title("Data Science Project")
st.write("## Sentiment Analysis")

menu = ["Business Objective", "Build Project", "New Prediction"]
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.write("""#### Thành viên thực hiện:
                 Nguyễn Vũ Khương & Lê Thị Vân Anh""")
st.sidebar.write("""#### Giảng viên hướng dẫn: Khuất Thùy Phương""")
st.sidebar.write("""#### Thời gian thực hiện: 12/2024""")


if choice == 'Business Objective':
    st.subheader("Business Objective")
    st.write("""
    ###### Sentiment Analysis is a natural language processing task that involves determining whether a piece of text is positive, negative, or neutral. It is widely used in various fields, such as social media analysis, customer feedback, and product reviews. In this project, we will focus on sentiment analysis using a dataset of customer reviews from an e-commerce platform.
    """)  
    st.write("""###### => Problem/Requirement: Analyze customer reviews and predict the sentiment of the reviews using machine learning algorithms.""")
    # st.image("sentiment_analysis.jpg")

elif choice == 'Build Project':
    st.subheader("Build Project")
    st.write("##### 1. Load and preprocess the dataset")
    # Load dataset
    reviews = Reviews()
    reviews.load_data()
    # show dataset
    data_root = reviews.data_root
    columns = ['id', 'ma_khach_hang', 'noi_dung_binh_luan', 'ngay_binh_luan', 'gio_binh_luan', 'so_sao', 'ma_san_pham']
    st.dataframe(data_root[columns].head(3))
    st.dataframe(data_root[columns].tail(3))  
    fig, ax = reviews.visualize_sentiment()
    st.pyplot(fig)



else:
    st.subheader("New Prediction")
    st.write("Enter your review and select the day and hours of your review:")
    text = st.text_area("Review")
    day = datetime.now()
    hours = int(day.hour)
    time = datetime.now()
    if text:
        review = Reviews(text, day=day, hours=hours)
        result = model_rf.transform(review)
        if result == 2:
            result = "positive"
        elif result == 1:
            result = "neutral"
        else:
            result = "negative"
        st.write(f"Predicted sentiment: {result}")
        st.write(f"Time predict: {datetime.now() - time}")





