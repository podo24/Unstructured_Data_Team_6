import pandas as pd
import pyarrow.json as paj
import numpy as np
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
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
             'tif', 'tiff', 'jpg', 'jpeg', 'png', 'pdf', 'doc', 'docx', 'xls', 'xlsx',
             'embodiments', 'seq', 'id', 'mg', 'ml']
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
    
    result = text_series.str.replace(r'[^a-zA-Z0-9\-\s]', ' ', regex=True).str.lower()
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

    print("Reading data files...")
    files = ['2026-data.csv', '2025-data.csv', '2024-data.csv', '2023-data.csv', '2022-data.csv', '2021-data.csv', '2020-data.csv', '2019-data.csv', '2018-data.csv', '2017-data.csv']
    dfs = []
    for f in files:
        dfs.append(pd.read_csv(f, index_col=0, engine='pyarrow'))

    # DataFrame을 하나로 합침
    print("Concatenating data...")
    data = pd.concat(dfs, ignore_index=True)

    # Family 특허 제거 (동일 패밀리는 1건만 남김)
    print("Removing family patents...")
    family_members_col = None
    for col in ['Simple Family Members', 'Extended Family Members']:
        if col in data.columns:
            family_members_col = col
            break

    if family_members_col:
        before = len(data)
        if 'Earliest Priority Date' in data.columns:
            data = data.sort_values('Earliest Priority Date')
        lens_col = 'Lens ID' if 'Lens ID' in data.columns else None
        if lens_col:
            family_key = data[family_members_col].fillna(data[lens_col])
        else:
            family_key = data[family_members_col].fillna(data.index.astype(str))
        data = data.loc[~family_key.duplicated()].copy()
        print(f"Removed {before - len(data)} family patents using {family_members_col}.")
    else:
        family_size_col = None
        for col in ['Simple Family Size', 'Extended Family Size']:
            if col in data.columns:
                family_size_col = col
                break
        if family_size_col:
            before = len(data)
            family_size = pd.to_numeric(data[family_size_col], errors='coerce').fillna(1)
            data = data.loc[family_size <= 1].copy()
            print(f"Removed {before - len(data)} family patents using {family_size_col}.")
        else:
            print("Warning: no family columns found; skipping family patent removal.")

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


    # Title과 Abstract을 합쳐서 텍스트 데이터로 사용
    print("Combining Title and Abstract into a single text column...")
    data['text'] = data['Title'] + ' ' + data['Abstract'].fillna('')

    # 영어가 아닌 텍스트, 국가 이름 등을 제거 (멀티프로세싱 적용)
    print("Cleaning text data...")
    data['text'] = parallelize_dataframe(data['text'], clean_text_worker, n_cores=6)

    # BeaTopic 실행
    print("Initializing BERTopic...")
    device = 'cuda' if torch.cuda.is_available() else exit("No GPU available, exiting...")
    # embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    embedding_model = SentenceTransformer("pritamdeka/S-PubMedBERT-MS-MARCO", device=device)
    texts = data['text'].tolist()
    embeddings = embedding_model.encode(texts, show_progress_bar=True, batch_size=128, convert_to_numpy=True)

    umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    vectorizer_model = CountVectorizer(stop_words="english", max_features=5000)

    model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
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
        topic_info.to_csv('bertopic_ForALL_topics.csv', index=False)

    # 문서별 토픽 및 확률 정보 저장
    doc_probs = [_extract_max_prob(p) for p in probabilities]
    doc_df = pd.DataFrame({
        'text': data['text'],
        'topic': topics,
        'probability': doc_probs
    })
    doc_df.to_csv('bertopic_ForALL_documents.csv', index=False)
    
    print("All processes completed successfully!")