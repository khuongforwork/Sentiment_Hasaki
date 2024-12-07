# import findspark
# findspark.init()
from pyspark.sql.functions import length, isnan, when, round, col, udf, lit, concat, asc, regexp_replace
from functools import reduce
import re
from gensim.models import Word2Vec
import numpy as np
import regex  
from underthesea import word_tokenize, pos_tag, sent_tokenize
from gensim.models import Word2Vec
import numpy as np
import ast
import unicodedata as ud
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pyspark.sql import functions as F

#LOAD EMOJICON
file = open('files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
#LOAD NEGATIVE IMOJICON

file = open('files/negative_emoji.txt', 'r', encoding="utf8")
negative_phrases = file.read().split('\n')

file.close()

#LOAD POSITIVE IMOJICON

file = open('files/positive_emoji.txt', 'r', encoding="utf8")
positive_phrases = file.read().split('\n')
file.close()

#LOAD NEGATIVE WORD

file = open('files/negative_VN.txt', 'r', encoding="utf8")
negative_words = file.read().split('\n')
file.close()

#LOAD POSITIVE WORD

file = open('files/positive_VN.txt', 'r', encoding="utf8")
positive_words = file.read().split('\n')
file.close()

#LOAD TEENCODE
file = open('files/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
###############
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open('files/english-vnmese.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()
################
#LOAD wrong words
file = open('files/wrong-word.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()
#################
#LOAD STOPWORDS
file = open('files/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()




def load_special_list(path):
    """
    Load danh sách từ đặc biệt cần dò trong bi_grams.txt và tri_grams.txt
    """
    with open(path, encoding='utf8') as f:
        words = f.read()
        words = ast.literal_eval(words)

    return words


bi_grams = load_special_list('files/bi_grams.txt')
tri_grams = load_special_list('files/tri_grams.txt')


def process_text(text):
    document = text.lower()
    document = document.replace("’",'')
    document = regex.sub(r'\.+', ".", document)
    new_sentence =''
    for sentence in sent_tokenize(document):
        # if not(sentence.isascii()):
        ###### CONVERT EMOJICON
        sentence = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence))
        ###### CONVERT TEENCODE
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())
        ###### DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern,sentence))
        # ...
        ###### DEL wrong words
        sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        new_sentence = new_sentence + sentence + ' '
    document = new_sentence
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    #...
    return document




# Chuẩn hóa unicode tiếng việt
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic



# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def covert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)



# Hàm để chuẩn hóa các từ có ký tự lặp
def normalize_repeated_characters(text):
    # Thay thế mọi ký tự lặp liên tiếp bằng một ký tự đó
    # Ví dụ: "lònggggg" thành "lòng", "thiệtttt" thành "thiệt"
    return re.sub(r'(.)\1+', r'\1', text)




def process_special_word_extend(text):
    # có thể có nhiều từ đặc biệt cần ráp lại với nhau
    new_text = ''
    text_lst = text.split()
    i= 0
    # không, chẳng, chả, khum, kh, k
    if 'không' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            #print(word)
            #print(i)
            if  word == 'không':
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text.strip()

def split_word(text):
    
    word = '\w+'
    non_word = '[^\w\s]'
    digits = '\d+([\.,_]\d+)+'
    
    patterns = []
    patterns.extend([word, non_word, digits])
    patterns = f"({'|'.join(patterns)})"
    
    text = ud.normalize('NFC', text)
    tokens = re.findall(patterns, text, re.UNICODE)
    
    return [token[0] for token in tokens]

def process_special_word(text):
    """
    Xử lý các từ đặc biệt cần ráp lại với nhau
    """    
    word = split_word(text)
    word_len = len(word)
    
    curr_id = 0
    word_list = []
    done = False
    
    while (curr_id < word_len) and (not done):
        curr_word = word[curr_id]
        
        if curr_id >= word_len - 1:
            word_list.append(curr_word)
            done = True
        
        else:
            next_word = word[curr_id + 1]
            pair_word = ' '.join([curr_word.lower(), next_word.lower()])
        
            if curr_id >= (word_len - 2):
                if pair_word in bi_grams:
                    word_list.append('_'.join([curr_word, next_word]))
                    curr_id += 2
        
                else:
                    word_list.append(curr_word)
                    curr_id += 1
        
            else:
                next_next_word = word[curr_id + 2]
                triple_word = ' '.join([pair_word, next_next_word.lower()])
        
                if triple_word in tri_grams:
                    word_list.append('_'.join([curr_word, next_word, next_next_word]))
                    curr_id += 3
        
                elif pair_word in bi_grams:
                    word_list.append('_'.join([curr_word, next_word]))
                    curr_id += 2
        
                else:
                    word_list.append(curr_word)
                    curr_id += 1
    
    text = ' '.join(word for word in word_list)

    return text

def process_postag_thesea(text):
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.','')
        ###### POS tag
        lst_word_type = ['N','Np','A','AB','V','VB','VY','R']
        # lst_word_type = ['A','AB','V','VB','VY','R']
        sentence = ' '.join( word[0] if word[1].upper() in lst_word_type else '' for word in pos_tag(process_special_word_extend(word_tokenize(sentence, format="text"))))
        new_document = new_document + sentence + ' '
    ###### DEL excess blank space
    new_document = regex.sub(r'\s+', ' ', new_document).strip()
    return new_document



def remove_stopword(text):
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords_lst else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document



def get_null_values(document):
    return document.filter(reduce(lambda x, y: x | y, (col(c).isNull() for c in document.columns)))



# Kiểm tra dữ liệu thiếu
def kiem_tra_du_lieu_thieu(document):

    for col in document.columns:
        missing = document.where(document[col].isNull()).count()
        missing_percent = ( missing / document.count() ) * 100
        print('{}: {}'.format(col, missing_percent))



def get_group_by_so_sao(document):
    grouped_by_image = document.groupBy("so_sao").agg(round(avg("ma_khach_hang"), 2).alias("so_kh"))

# Hàm phân tích cảm xúc

# Định nghĩa lại hàm UDF

def analyze_sentiment(text):
    text = text.lower()  # Chuyển thành chữ thường
    positive_score, negative_score = 0, 0

    # Đếm cụm từ tiêu cực
    for phrase in negative_phrases:
        if phrase in text:
            negative_score += text.count(phrase)
            text = text.replace(phrase, "")  # Loại bỏ cụm từ đã xử lý

    # Đếm cụm từ tích cực
    for phrase in positive_phrases:
        if phrase in text:
            positive_score += text.count(phrase)
            text = text.replace(phrase, "")  # Loại bỏ cụm từ đã xử lý

    # Đếm từ tích cực
    for word in positive_words:
        positive_score += len(re.findall(rf'\b{word}\b', text))

    # Đếm từ tiêu cực
    for word in negative_words:
        negative_score += len(re.findall(rf'\b{word}\b', text))

    # Xác định cảm xúc dựa trên điểm số
    if positive_score > negative_score:
        sentiment = "positive"
    elif negative_score > positive_score:
        sentiment = "negative"
    else:
        sentiment = 'unknown'
    
    # Trả về kết quả dưới dạng từ điển
    return {"sentiment_label": sentiment, 
            "positive_score": positive_score, 
            "negative_score": negative_score}





def find_positive_words(document):
    '''
    Đếm số từ/icon tích cực/tiêu cực có trong câu văn/đoạn văn
    '''
    document_lower = document.lower()
    count = 0
    word_list = []

    for word in positive_phrases:
        if word in document_lower:
            count += document_lower.count(word)
            word_list.append(word)

    for emoji in positive_words:
        if emoji in document_lower:
            count += document_lower.count(emoji)
            word_list.append(emoji)

    return count




def find_negative_words(document):
    '''
    Đếm số từ/icon tích cực/tiêu cực có trong câu văn/đoạn văn
    '''
    document_lower = document.lower()
    count = 0
    word_list = []

    for word in negative_phrases:
        if word in document_lower:
            count += document_lower.count(word)
            word_list.append(word)

    for emoji in negative_words:
        if emoji in document_lower:
            count += document_lower.count(emoji)
            word_list.append(emoji)

    return count


def assign_sentiment(data):
    """
    Hàm gán nhãn sentiment dựa trên các điều kiện cho trước.
    """
    data = data.withColumn(
            'sentiment_label',
            F.when((F.col('positive_score') > F.col('negative_score')) & (F.col('so_sao') > 3), 'positive')
            .when((F.col('positive_score') < F.col('negative_score')) & (F.col('so_sao') < 3), 'negative')
            .when((F.col('positive_score') == F.col('negative_score')) & (F.col('so_sao') > 3), 'positive')
            .when((F.col('positive_score') == F.col('negative_score')) & (F.col('so_sao') < 3), 'negative')
            .otherwise('neutral')
            )

    data = data.withColumn(
            'so_sao',
            F.when((F.col('sentiment_label') == 'neutral') & (F.col('so_sao') > 3), 3)
            .when((F.col('sentiment_label') == 'neutral') & (F.col('so_sao') < 3), 3)
            .otherwise(col('so_sao'))
            )
    return data


from gensim.models import Word2Vec
import numpy as np

# # Bước 1: Tokenization
# data['tokenized'] = data['binh_luan'].astype(str).apply(lambda x: x.lower().split())

# # Bước 2: Huấn luyện Word2Vec
# model = Word2Vec(data['tokenized'], vector_size=50, window=5, min_count=1, workers=4)

# Bước 3: Chuyển đổi câu thành vector
def sentence_to_vector(sentence, model, vector_size):
    """
    Chuyển đổi câu thành vector bằng cách lấy trung bình vector các từ trong câu.
    """
    tokens = sentence.lower().split()
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)  # Trung bình vector
    else:
        return np.zeros(vector_size)    # Trả về vector không nếu không có từ nào hợp lệ




def save_to_data_save(data, file_name):
    import os
    path = f"data_save/{file_name}.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data.write.csv(path, mode='overwrite', header=True)


def load_from_data_save(file_name, spark):
    data = spark.read.csv(f"data_save/{file_name}.csv", header=True, inferSchema=True)
    return data

def replace_non_numeric_with_zero(df, column_name):
    # Tạo cột mới, thay giá trị không phải số bằng 0
    df_with_replacement = df.withColumn(
        column_name,
        when(
            regexp_extract(col(column_name), r'^\d+(\.\d+)?$', 0) == "",
            0
        ).otherwise(col(column_name))
    )
    return df_with_replacement


def remove_null_rows(df, col):
    """
    Loại bỏ các dòng chứa giá trị NULL trong bất kỳ cột nào.
    
    Args:
        df (DataFrame): PySpark DataFrame.
    
    Returns:
        DataFrame: PySpark DataFrame sau khi loại bỏ các dòng NULL.
    """
    return df.dropna(subset=col)



def get_group_by_rate(document):
    # Tạo bảng tổng hợp số sao và số khách hàng
    grouped_by_image = document.groupBy("so_sao").agg(
        count("ma_khach_hang").alias("so_kh")
    )
    return grouped_by_image

def get_allTweetWords(document, col):

     Cleaned_Text = document.select(col).collect()
     Cleaned_Text_List = [row[col] for row in Cleaned_Text]

     wordList = []
     for CleanedText in Cleaned_Text_List:
          wordList.append(CleanedText.split(' '))

     allTweetWords = [word for subList in wordList for word in subList]
     '''
     Remove empty strings
     '''
     allTweetWords = list(filter(None, allTweetWords))
     allTweetWords=set(allTweetWords)

     return allTweetWords


from wordcloud import WordCloud

def getWordCloud(wordList,color,plt):

    allWords = ' '.join([word for word in wordList])
    wordCloud = WordCloud(background_color=color,
                          width=1600,
                          height=800,
                          random_state=21,
                          max_words=50,
                          max_font_size=200).generate(allWords)
    
    plt.figure(figsize=(15, 10))
    plt.axis('off')
    plt.imshow(wordCloud, interpolation='bilinear')
    return allWords
    





def show_wordcloud_for_reviews_by_label(reviews, col):

    reviews = reviews.toPandas()
    # Lọc các bình luận theo nhãn sentiment
    positive_words = reviews[(reviews['sentiment_label'] == 'positive')]['noi_dung_binh_luan_stopword'].str.split(' ').explode().value_counts()[:100]
    negative_words = reviews[(reviews['sentiment_label'] == 'negative')]['noi_dung_binh_luan_stopword'].str.split(' ').explode().value_counts()[:100]
    neutral_words = reviews[(reviews['sentiment_label'] == col)]['noi_dung_binh_luan_stopword'].str.split(' ').explode().value_counts()[:100]

    # Tạo và hiển thị WordCloud cho từng loại sentiment
    if not positive_words.empty:
        positive_wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(positive_words)
        plt.figure(figsize=(10, 6))
        plt.title('positive')
        plt.imshow(positive_wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    if not negative_words.empty:
        negative_wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(negative_words)
        plt.figure(figsize=(10, 6))
        plt.title('negative')
        plt.imshow(negative_wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    if not neutral_words.empty:
        neutral_wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(neutral_words)
        plt.figure(figsize=(10, 6))
        plt.title(col)
        plt.imshow(neutral_wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()