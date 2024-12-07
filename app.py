import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime
from Reviews import Reviews
from Model_rf import Model_rf
from sklearn.preprocessing import LabelEncoder

# Load product and review datasets
products = pd.read_csv('GUI/product_list.csv')
reviews = Reviews()
reviews.load_data()
# Preprocess product names
products['ten_san_pham'] = products['ten_san_pham'].astype(str)
product_list = products['ten_san_pham'].values

# Initialize models
model_rf = Model_rf()

# Streamlit App Title
st.title("Data Science Project Sentiment Analysis")

# Sidebar Menu
menu = ["Business Objective", "Build Project", "New Prediction", "Analysis Products"]
choice = st.sidebar.selectbox('Menu', menu)

st.sidebar.write("#### Thành viên thực hiện:\nNguyễn Vũ Khương & Lê Thị Vân Anh")
st.sidebar.write("#### Giảng viên hướng dẫn: Khuất Thùy Phương")
st.sidebar.write("#### Thời gian thực hiện: 12/2024")

# Business Objective
if choice == 'Business Objective':
    st.subheader("Mục tiêu kinh doanh")
    st.write("""
    ###### Phân tích cảm xúc (Sentiment Analysis) là một nhiệm vụ trong xử lý ngôn ngữ tự nhiên (NLP) nhằm xác định xem một đoạn văn bản mang tính tích cực, tiêu cực hay trung lập. 
    Đây là một công cụ quan trọng và được ứng dụng rộng rãi trong nhiều lĩnh vực khác nhau, chẳng hạn như phân tích mạng xã hội, đánh giá phản hồi của khách hàng, hay phân tích các nhận xét về sản phẩm. 
    Trong dự án này, chúng tôi sẽ tập trung vào việc phân tích cảm xúc dựa trên một tập dữ liệu gồm các đánh giá của khách hàng từ một nền tảng thương mại điện tử.
    """)  
    st.write("""
    ###### => Vấn đề/Yêu cầu: Phân tích các đánh giá của khách hàng và dự đoán cảm xúc của những đánh giá này bằng cách sử dụng các thuật toán học máy. 
    Mục tiêu chính là giúp doanh nghiệp hiểu rõ hơn về ý kiến của khách hàng để cải thiện chất lượng sản phẩm, dịch vụ và tối ưu hóa trải nghiệm khách hàng. 
    Ngoài ra, việc phân tích cảm xúc còn hỗ trợ doanh nghiệp trong việc đưa ra các chiến lược kinh doanh phù hợp, từ việc quản lý khủng hoảng truyền thông đến phát triển sản phẩm mới dựa trên nhu cầu thực tế của người tiêu dùng.
    """)

# Build Project
elif choice == 'Build Project':
    st.subheader("Xây dựng mô hình")
    reviews = Reviews()
    reviews.load_data()
    data_root = reviews.data_root

    st.write("##### 1. Load the dataset")
    columns = ['id', 'ma_khach_hang', 'noi_dung_binh_luan', 'ngay_binh_luan', 'gio_binh_luan', 'so_sao', 'ma_san_pham']
    st.dataframe(data_root[columns].head(3))
    st.dataframe(data_root[columns].tail(3))

    st.write("##### 2. EDA the dataset")
    fig_sentiment = reviews.visualize_sentiment()
    fig_length = reviews.visualize_length()
    fig_day_group = reviews.visualize_day_group()
    fig_time_group = reviews.visualize_time_group()
    st.pyplot(fig_sentiment)
    st.pyplot(fig_length)
    st.pyplot(fig_day_group)
    st.pyplot(fig_time_group)

    st.write("##### 3. Modeling Random Forest - Sklearn")
    st.write("**Đầu vào:** noi_dung_binh_luan_stopword, length, time_group, day_group")
    st.write("**Đánh giá kết quả:** Classification report")

    data_processed = reviews.data_processed
    label_encoder = LabelEncoder()
    data_processed['indexedLabel'] = label_encoder.fit_transform(data_processed['sentiment_label'])
    X = data_processed[['noi_dung_binh_luan_stopword', 'lenght', 'time_group', 'day_group']]
    y = data_processed['indexedLabel']

    report_df, fig_classification_report = model_rf.get_classification_report_df(X, y)
    st.write("### Classification Report")
    st.dataframe(report_df)
    st.pyplot(fig_classification_report)

# New Prediction
elif choice == "New Prediction":
    st.subheader("New Prediction")
    st.write("Nhập đáng giá của bạn tại đây:")
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

# Analysis Products
# Analysis Products
elif choice == "Analysis Products":
    st.subheader("Analysis Products")
    search_query = st.text_input("Tìm kiếm sản phẩm:")
    if search_query:
        filtered_products = [p for p in product_list if search_query.lower() in p.lower()]
        if filtered_products:
            selected_product = st.selectbox("Chọn một sản phẩm:", filtered_products)
            if selected_product:
                product_detail = products[products['ten_san_pham'] == selected_product]
                if not product_detail.empty:
                    product_id = product_detail['ma_san_pham'].values[0]
                    st.dataframe(product_detail[['ma_san_pham', 'ten_san_pham', 'diem_trung_binh']])

                    # Show sentiment statistics
                    sentiment_counts = reviews.data_processed[reviews.data_processed['ma_san_pham'] == product_id].groupby('sentiment_label').size()
                    positive = sentiment_counts.get('positive', 0)
                    negative = sentiment_counts.get('negative', 0)
                    neutral = sentiment_counts.get('neutral', 0)
                    sentiment_data = pd.DataFrame({
                        'Sentiment': ['Positive', 'Negative', 'Neutral'],
                        'Count': [positive, negative, neutral]
                    })
                    st.bar_chart(sentiment_data.set_index('Sentiment'))

                    # Generate WordClouds
                    st.write("### WordClouds for Reviews")
                    positive_reviews = reviews.data_processed[(reviews.data_processed['ma_san_pham'] == product_id) & (reviews.data_processed['sentiment_label'] == 'positive')]
                    negative_reviews = reviews.data_processed[(reviews.data_processed['ma_san_pham'] == product_id) & (reviews.data_processed['sentiment_label'] == 'negative')]
                    neutral_reviews = reviews.data_processed[(reviews.data_processed['ma_san_pham'] == product_id) & (reviews.data_processed['sentiment_label'] == 'neutral')]

                    def generate_wordcloud(data, title):
                        words = ' '.join(data['noi_dung_binh_luan_stopword'])
                        wordcloud = WordCloud(width=800, height=400).generate(words)
                        plt.figure(figsize=(10, 5))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.title(title, fontsize=15)
                        plt.axis('off')
                        st.pyplot(plt)

                    if not positive_reviews.empty:
                        generate_wordcloud(positive_reviews, "Positive Reviews")
                    if not negative_reviews.empty:
                        generate_wordcloud(negative_reviews, "Negative Reviews")
                    if not neutral_reviews.empty:
                        generate_wordcloud(neutral_reviews, "Neutral Reviews")

                    # Display monthly purchase statistics
                    st.write("### Tần suất mua hàng theo tháng của khách hàng")
                    reviews.data_processed['ngay_binh_luan'] = pd.to_datetime(reviews.data_processed['ngay_binh_luan'], errors='coerce')
                    monthly_purchases = reviews.data_processed[reviews.data_processed['ma_san_pham'] == product_id].groupby(
                        reviews.data_processed['ngay_binh_luan'].dt.to_period('M')
                    ).size().reset_index(name='Purchase Count')
                    monthly_purchases['Month'] = monthly_purchases['ngay_binh_luan'].dt.strftime('%Y-%m')
                    st.line_chart(monthly_purchases.set_index('Month')['Purchase Count'])

                    
