import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import re
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# nltk 리로스 다운로드
# nltk.download('stopwords')
# nltk.download('punkt_tab')

# csv 데이터를 불러와 pandas DataFrame으로 변환
print("Reading data files...")
files = ['2026-data.csv', '2025-data.csv', '2024-data.csv', '2023-data.csv', '2022-data.csv', '2021-data.csv', '2020-data.csv', '2019-data.csv', '2018-data.csv', '2017-data.csv']
dfs = []
for f in files:
    dfs.append(pd.read_csv(f, index_col=0, engine='pyarrow'))

# DataFrame을 하나로 합침
print("Concatenating data...")
data = pd.concat(dfs, ignore_index=True)

# 필요한 속성만 선택하도록
print("Selecting required columns...")
required_columns = [
    'Publication Date', 'Publication Year', 'Application Date',
    'Earliest Priority Date', 'Title', 'Abstract', 'Applicants',
    'Cited by Patent Count', 'CPC Classifications', 'Legal Status'
]

existing_cols = [c for c in required_columns if c in data.columns]
missing_cols = [c for c in required_columns if c not in data.columns]
if missing_cols:
    print(f"Warning: these required columns are missing and will be skipped: {missing_cols}")
data = data[existing_cols].copy()

# DataFrame 정보 출력
print("DataFrame info:")
print(data.info())
print("DataFrame Describe:")
print(data.describe())

# Title과 Abstract을 합쳐서 텍스트 데이터로 사용
print("Combining Title and Abstract into a single text column...")
data['text'] = data['Title'] + ' ' + data['Abstract'].fillna('')

# 영어가 아닌 텍스트, 국가 이름 등을 제거
print("Cleaning text data...")
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    country_names = ['usa', 'korea', 'china', 'japan', 'germany', 'france', 'uk', 'india']
    other_words = ['tif', 'tiff', 'jpg', 'jpeg', 'png', 'pdf', 'doc', 'docx', 'xls', 'xlsx']
    for country in country_names:
        text = re.sub(r'\b' + re.escape(country) + r'\b', '', text, flags=re.IGNORECASE)
    for word in other_words:
        text = re.sub(r'\b' + re.escape(word) + r'\b', '', text, flags=re.IGNORECASE)

    return text

data['text'] = data['text'].apply(clean_text)

# 불용어제거
# print("Removing stop words...")
# def remove_stop_words(text):
#     stop_words = set(stopwords.words('english'))
#     word_tokens = word_tokenize(text)
#     filtered_text = ' '.join([word for word in word_tokens if word.lower() not in stop_words])
#     return filtered_text

# data['text'] = data['text'].apply(remove_stop_words)

# BeaTopic 실행
print("Initializing BERTopic...")
device = 'cuda' if torch.cuda.is_available() else exit("No GPU available, exiting...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
texts = data['text'].tolist()
embeddings = embedding_model.encode(texts, show_progress_bar=True, batch_size=256, convert_to_numpy=True)

pca_model = PCA(n_components=5, svd_solver="randomized", random_state=42)
vectorizer_model = CountVectorizer(stop_words="english", max_features=5000)

model = BERTopic(
    embedding_model=None,
    umap_model=pca_model,
    vectorizer_model=vectorizer_model,
    verbose=True
)

topics, probabilities = model.fit_transform(data['text'], embeddings)

# 결과 저장 및 확인 (토픽 정보와 문서별 토픽을 CSV로 저장)
print("Saving BERTopic results to CSV files...")

# 토픽 요약 정보 DataFrame으로 가져오기
try:
    topic_info = model.get_topic_info()
except Exception:
    topic_info = None

if topic_info is not None:
    topic_info.to_csv('bertopic_topics.csv', index=False)

# 문서별 토픽 및 확률 정보 저장
def _extract_max_prob(p):
    try:
        if p is None:
            return float('nan')
        arr = np.array(p)
        if arr.size == 0:
            return float('nan')
        return float(np.max(arr))
    except Exception:
        return float('nan')

doc_probs = [_extract_max_prob(p) for p in probabilities]
doc_df = pd.DataFrame({
    'text': data['text'],
    'topic': topics,
    'probability': doc_probs
})
doc_df.to_csv('bertopic_documents.csv', index=False)