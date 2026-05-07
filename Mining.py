import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

nltk.download('stopwords')
nltk.download('punkt_tab')

# read data
print("Reading data...")
data_2026 = pd.read_csv('2026-data.csv', index_col=0, low_memory=False)
data_2025 = pd.read_csv('2025-data.csv', index_col=0, low_memory=False)
data_2024 = pd.read_csv('2024-data.csv', index_col=0, low_memory=False)
data_2023 = pd.read_csv('2023-data.csv', index_col=0, low_memory=False)
data_2022 = pd.read_csv('2022-data.csv', index_col=0, low_memory=False)
data_2021 = pd.read_csv('2021-data.csv', index_col=0, low_memory=False)

# concat all data
print("Concatenating data...")
data = pd.concat([data_2026, data_2025, data_2024, data_2023, data_2022, data_2021], ignore_index=True)

# delete unnecessary columns
 #   Column                                Non-Null Count   Dtype 
# ---  ------                                --------------   ----- 
#  0   Jurisdiction                          233727 non-null  str   
#  1   Kind                                  233727 non-null  str   
#  2   Display Key                           233727 non-null  str   
#  3   Lens ID                               233727 non-null  str   
#  4   Publication Date                      233727 non-null  str   
#  5   Publication Year                      233727 non-null  int64 
#  6   Application Number                    233727 non-null  str   
#  7   Application Date                      233727 non-null  str   
#  8   Priority Numbers                      233727 non-null  str   
#  9   Earliest Priority Date                233727 non-null  str   
#  10  Title                                 233727 non-null  str   
#  11  Abstract                              215279 non-null  str   
#  12  Applicants                            233620 non-null  str   
#  13  Inventors                             232441 non-null  str   
#  14  Owners                                34702 non-null   str   
#  15  URL                                   233727 non-null  str   
#  16  Document Type                         233727 non-null  str   
#  17  Has Full Text                         233727 non-null  str   
#  18  Cites Patent Count                    233727 non-null  int64 
#  19  Cited by Patent Count                 233727 non-null  int64 
#  20  Simple Family Size                    233727 non-null  int64 
#  21  Simple Family Members                 233727 non-null  str   
#  22  Simple Family Member Jurisdictions    233727 non-null  str   
#  23  Extended Family Size                  233727 non-null  int64 
#  24  Extended Family Members               233727 non-null  str   
#  25  Extended Family Member Jurisdictions  233727 non-null  str   
#  26  Sequence Count                        233727 non-null  int64 
#  27  CPC Classifications                   223646 non-null  str   
#  28  IPCR Classifications                  230760 non-null  str   
#  29  US Classifications                    1 non-null       object
#  30  NPL Citation Count                    233727 non-null  int64 
#  31  NPL Resolved Citation Count           233727 non-null  int64 
#  32  NPL Resolved Lens ID(s)               62881 non-null   str   
#  33  NPL Resolved External ID(s)           62881 non-null   str   
#  34  NPL Citations                         89044 non-null   str   
#  35  Legal Status                          233727 non-null  str  

print("Dropping unnecessary columns...")
data.drop(columns=[
	'Jurisdiction', 'Kind', 'Display Key', 'Lens ID', 'Application Number', 'Priority Numbers',
	'Inventors', 'Owners', 'URL', 'Document Type', 'Has Full Text', 'Cites Patent Count',
	'Simple Family Size', 'Simple Family Members', 'Simple Family Member Jurisdictions',
	'Extended Family Size', 'Extended Family Members', 'Extended Family Member Jurisdictions',
	'Sequence Count', 'IPCR Classifications', 'US Classifications', 'NPL Citation Count',
	'NPL Resolved Citation Count', 'NPL Resolved Lens ID(s)', 'NPL Resolved External ID(s)',
	'NPL Citations'
], inplace=True)

# show data info
# print(data.info())
# print(data.head())
# print(data.tail())
# print(data.describe())

print("Combining Title and Abstract into a single text column...")
data['text'] = data['Title'] + ' ' + data['Abstract'].fillna('')

# Remove non-englihsh characters and Remove countries Name
print("Cleaning text data...")
def clean_text(text):
    # Remove non-English characters (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove country names (example: USA, Korea, etc.)
    country_names = ['USA', 'Korea', 'China', 'Japan', 'Germany', 'France', 'UK', 'India']  # Add more as needed
    for country in country_names:
        text = re.sub(r'\b' + re.escape(country) + r'\b', '', text, flags=re.IGNORECASE)
    
    return text

data['text'] = data['text'].apply(clean_text)

# Remove stop words from the text data
print("Removing stop words...")
def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = ' '.join([word for word in word_tokens if word.lower() not in stop_words])
    return filtered_text

data['text'] = data['text'].apply(remove_stop_words)
print("Sample cleaned text data:")
print(data['text'].head())

print("Initializing BERTopic...")
# 1. GPU 사용을 위한 임베딩 모델 명시적 생성
# 'all-MiniLM-L6-v2'는 BERTopic의 기본 모델입니다.
# device='cuda'를 통해 GPU를 강제 지정합니다.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

# 2. BERTopic 모델 생성 시 embedding_model 전달
# verbose=True를 설정하면 진행 상황(GPU 연산 중인지 등)을 확인할 수 있어 좋습니다.
model = BERTopic(
    embedding_model=embedding_model,
    verbose=True
)

# 3. 학습 및 변환 (이제 GPU를 사용하여 임베딩을 추출합니다)
topics, probabilities = model.fit_transform(data['text'])

# 결과 저장 및 확인
print("Saving results to 'bertopic_result.txt'...")
topic_info = model.get_topic_info()
result_df = pd.DataFrame({
    'text': data['text'],
    'topic': topics,
    'probability': probabilities if probabilities is not None else [None] * len(topics)
})

with open('bertopic_result.txt', 'w', encoding='utf-8') as file:
    file.write('=== Topic Info ===\n')
    file.write(topic_info.to_string(index=False))
    file.write('\n\n=== Document Topic Assignments ===\n')
    file.write(result_df.to_string(index=False))

print(topic_info)