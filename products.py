import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud






class Products():
    def __init__(self):
        # Đọc dữ liệu sản phẩm và review
        products = pd.read_csv('GUI/product_list.csv')
        reviews = pd.read_csv('GUI/review_final.csv')

        products['ten_san_pham'] = products['ten_san_pham'].astype(str)
        product_list = products['ten_san_pham'].values

        self.products = products
        self.products_list = product_list

    def show_product_detail(product_id, reviews, products):
        product_detail = products[products['ma_san_pham'] == product_id]
        count_sentiment = reviews[reviews['ma_san_pham'] == product_id].groupby('prediction_label').count()['diem_trung_binh']
        
        # Trả về tỉ lệ sentiment: Positive, Negative, Neutral
        return (count_sentiment.get('positive', 0), count_sentiment.get('negative', 0), count_sentiment.get('neutral', 0))
    
    def show_wordcloud_for_reviews_by_label(st, product_id, reviews):
        # Lọc các bình luận theo nhãn sentiment
        positive_words = reviews[(reviews['ma_san_pham'] == product_id) & (reviews['prediction_label'] == 'positive')]['binh_luan'].str.split(' ').explode().value_counts()[:10]
        negative_words = reviews[(reviews['ma_san_pham'] == product_id) & (reviews['prediction_label'] == 'negative')]['binh_luan'].str.split(' ').explode().value_counts()[:10]
        neutral_words = reviews[(reviews['ma_san_pham'] == product_id) & (reviews['prediction_label'] == 'neutral')]['binh_luan'].str.split(' ').explode().value_counts()[:10]

        # Tạo và hiển thị WordCloud cho từng loại bình luận
        if not positive_words.empty:
            st.subheader('Positive Words')
            positive_wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(positive_words)
            plt.figure(figsize=(8, 4))
            plt.imshow(positive_wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
        
        if not negative_words.empty:
            st.subheader('Negative Words')
            negative_wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(negative_words)
            plt.figure(figsize=(8, 4))
            plt.imshow(negative_wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
        
        if not neutral_words.empty:
            st.subheader('Neutral Words')
            neutral_wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(neutral_words)
            plt.figure(figsize=(8, 4))
            plt.imshow(neutral_wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

    # Hàm tìm kiếm sản phẩm
    def find(search_query):
        # Lọc danh sách sản phẩm dựa trên từ khóa tìm kiếm
        filtered_products = [p for p in product_list if search_query.lower() in p.lower()]
        return filtered_products

    # Hiển thị sản phẩm đã tìm kiếm
    def show(st, selected_product):
        product_detail = products[products['ten_san_pham'] == selected_product]
        if not product_detail.empty:
            product_id = product_detail['ma_san_pham'].values[0]
            st.dataframe(product_detail[['ma_san_pham', 'ten_san_pham', 'diem_trung_binh']], use_container_width=True)
            return product_id
        return None