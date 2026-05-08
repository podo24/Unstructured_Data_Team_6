import pandas as pd
import pyarrow.json as paj
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
from multiprocessing import Pool

# ==========================================
# 1. 전역 변수 및 함수 정의 (워커 프로세스가 접근 가능하도록 최상단에 배치)
# ==========================================

bad_words = ['usa', 'korea', 'china', 'japan', 'germany', 'france', 'uk', 'india',
             'tif', 'tiff', 'jpg', 'jpeg', 'png', 'pdf', 'doc', 'docx', 'xls', 'xlsx']
bad_pattern_str = r'\b(' + '|'.join(bad_words) + r')\b'

def extract_text(val):
    try:
        if pd.isna(val):
            return ''
    except ValueError:
        # 배열이나 리스트 등의 경우 pd.isna 에서 에러가 발생할 수 있음
        pass
        
    if isinstance(val, dict):
        # 딕셔너리인 경우 값들만 추출하여 문자열로 연결
        return ' '.join(str(v) for v in val.values() if v is not None)
    elif isinstance(val, (list, np.ndarray)):
        # 리스트/배열인 경우 문자열로 연결
        return ' '.join(str(v) for v in val if v is not None)
        
    return str(val) if val is not None else ''

def clean_text_worker(text_series):
    bad_pattern = re.compile(bad_pattern_str, flags=re.IGNORECASE)
    
    result = text_series.str.replace(r'[^a-zA-Z\s]', ' ', regex=True).str.lower()
    result = result.str.replace(bad_pattern, '', regex=True)
    result = result.str.replace(r'\s+', ' ', regex=True).str.strip()
    return result

def parallelize_dataframe(df_col, func, n_cores=6):
    print(f"Parallelizing with {n_cores} cores...")
    # 경고 방지를 위해 numpy array_split 대신 리스트 슬라이싱을 권장하기도 하지만, 일단 유지
    df_split = np.array_split(df_col, n_cores)
    
    with Pool(n_cores) as pool:
        results = pool.map(func, df_split)
    
    return pd.concat(results)

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


# ==========================================
# 2. 메인 실행 블록 (멀티프로세싱 안정성을 위해 필수)
# ==========================================
if __name__ == '__main__':
    # nltk 리소스 다운로드 (최초 1회만 실행)
    # nltk.download('stopwords')
    # nltk.download('punkt_tab')

    # json lines 데이터를 불러와 pandas DataFrame으로 변환
    print("Reading data files...")
    opts = paj.ReadOptions(block_size=52428800) 
    table = paj.read_json('intrahepatic-cholangiocarcinoma-data.jsonl', read_options=opts)
    data = table.to_pandas()
    data = data.set_index(data.columns[0])

    # Abstract과 description과 claims를 합쳐서 텍스트 데이터로 사용
    print("Combining Title and Abstract into a single text column...")
    data['text'] = (
        data['abstract'].apply(extract_text) + ' ' +
        data['claims'].apply(extract_text) + ' ' +
        data['description'].apply(extract_text)
    ).str.strip()

    # 영어가 아닌 텍스트, 국가 이름 등을 제거 (멀티프로세싱 적용)
    print("Cleaning text data...")
    data['text'] = parallelize_dataframe(data['text'], clean_text_worker, n_cores=6)

    # BeaTopic 실행
    print("Initializing BERTopic...")
    device = 'cuda' if torch.cuda.is_available() else exit("No GPU available, exiting...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    texts = data['text'].tolist()
    embeddings = embedding_model.encode(texts, show_progress_bar=True, batch_size=128, convert_to_numpy=True)

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
        topic_info.to_csv('bertopic_intrahepatic-cholangiocarcinoma_topics.csv', index=False)

    # 문서별 토픽 및 확률 정보 저장
    doc_probs = [_extract_max_prob(p) for p in probabilities]
    doc_df = pd.DataFrame({
        'text': data['text'],
        'topic': topics,
        'probability': doc_probs
    })
    doc_df.to_csv('bertopic_intrahepatic-cholangiocarcinoma_documents.csv', index=False)
    
    print("All processes completed successfully!")