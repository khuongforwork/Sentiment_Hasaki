from datetime import datetime
import my_lib as functions  # Ensure that this module contains the appropriate functions.
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns 
class Reviews:
    
    def __init__(self, content=None, day=None, hours=None):
        self.data_processed = None
        self.data_root = None
        if content != None:
            self.content = content
            self.day = day
            self.hours = hours
            self.content_processed, self.lenght = self.process_text(self.content)
            self.day_group, self.hour_group = self.get_day_group(self.day, self.hours)

    
    def process_text(self, text):
        text = functions.process_text(text)
        text = functions.covert_unicode(text)
        text = functions.normalize_repeated_characters(text)
        text = functions.process_special_word(text)
        text = functions.process_special_word_extend(text)
        text = functions.remove_stopword(text)
        length = len(text.split(' '))
        return text, length

    def get_day_group(self, day, hour):
        # Ensure 'day' is a datetime object
        day_name = day.strftime('%A')

        if day_name in ('Monday', 'Saturday'):
            day_group = 1
        elif day_name in ('Wednesday', 'Friday'):
            day_group = 2
        else:
            day_group = 3

        if 1 <= hour <= 6:
            hour_group = 1
        elif 8 <= hour <= 23:
            hour_group = 2
        else:
            hour_group = 3
        return day_group, hour_group

    def load_data(self):
        # Use glob to find CSV files matching the pattern
        csv_path = 'data/review_final_4.csv/part-*.csv'
        file_list = glob.glob(csv_path)

        # Check if no files were found
        if not file_list:
            print("Error: No files found matching the pattern:", csv_path)
            return

        # Read the root data file
        try:
            self.data_root = pd.read_csv("data/Danh_gia.csv")
        except Exception as e:
            print("Error reading data/Danh_gia.csv:", e)
            return

        data_frames = []
        for file in file_list:
            try:
                # Use on_bad_lines to skip problematic lines
                df = pd.read_csv(file, on_bad_lines='skip')
                data_frames.append(df)
            except Exception as e:
                print(f"Error reading {file}:", e)

        if data_frames:
            self.data_processed = pd.concat(data_frames, ignore_index=True)
        else:
            print("No valid data was processed.")


    def visualize_sentiment(self):
        if 'sentiment' not in self.data_processed.columns:
            print("Error: Column 'sentiment' is missing in the dataset.")
            return None

        grouped_data = self.data_processed['sentiment'].value_counts()

        fig, ax = plt.subplots(figsize=(10, 6))
        grouped_data.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title("Số lượng theo Sentiment", fontsize=14)
        ax.set_xlabel("Sentiment", fontsize=12)
        ax.set_ylabel("Số lượng", fontsize=12)
        ax.set_xticklabels(grouped_data.index, rotation=0)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        return fig


    def visualize_time_group(self):
        if 'gio_binh_luan' not in self.data_processed.columns or 'sentiment_label' not in self.data_processed.columns:
            print("Error: Required columns are missing.")
            return None
        self.data_processed['gio_binh_luan'] = self.data_processed['gio_binh_luan'].str.split(':').str[0].astype(int)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=self.data_processed, x='gio_binh_luan', hue='sentiment_label', stat='probability', ax=ax)
        ax.set_title('Distribution of Sentiments Over Time')
        ax.set_xlabel('Time of Day')
        ax.set_ylabel('Probability')
        plt.tight_layout()

        return fig

    def visualize_day_group(self):
        if 'day_of_week' not in self.data_processed.columns or 'sentiment_label' not in self.data_processed.columns:
            print("Error: Required columns are missing.")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=self.data_processed, x='day_of_week', hue='sentiment_label', stat='probability', ax=ax)
        ax.set_title('Distribution of Sentiments by Day of the Week')
        ax.set_xlabel('Day of the Week')
        ax.set_ylabel('Probability')
        plt.tight_layout()

        return fig

    def visualize_length(self):
        if 'sentiment_label' not in self.data_processed.columns or 'noi_dung_binh_luan' not in self.data_processed.columns:
            print("Error: Required columns are missing.")
            return None

        figure, (pos_ax, neg_ax) = plt.subplots(1, 2, figsize=(15, 8))

        datachart = self.data_processed
        # Positive Sentiment
        pos_reviews = datachart[datachart['sentiment_label'] == 'positive']
        pos_word = pos_reviews['noi_dung_binh_luan'].str.split().apply(lambda x: [len(i) for i in x])
        sns.histplot(pos_word.map(lambda x: sum(x) / len(x)), ax=pos_ax, color='green')
        pos_ax.set_title('Average Word Length in Positive Reviews')
        pos_ax.set_xlabel('Average Word Length')
        pos_ax.set_ylabel('Density')

        # Negative Sentiment
        neg_reviews = datachart[datachart['sentiment_label'] == 'negative']
        neg_word = neg_reviews['noi_dung_binh_luan'].str.split().apply(lambda x: [len(i) for i in x])
        sns.histplot(neg_word.map(lambda x: sum(x) / len(x)), ax=neg_ax, color='red')
        neg_ax.set_title('Average Word Length in Negative Reviews')
        neg_ax.set_xlabel('Average Word Length')
        neg_ax.set_ylabel('Density')

        figure.suptitle('Average Word Length in Reviews')
        plt.tight_layout()

        return figure
