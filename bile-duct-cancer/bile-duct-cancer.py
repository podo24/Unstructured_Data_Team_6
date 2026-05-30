import pandas as pd
import pyarrow.json as paj
import numpy as np
from pathlib import Path
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import re
import torch
from multiprocessing import Pool

# ==========================================
# 1. 전역 변수 및 함수 정의 (워커 프로세스가 접근 가능하도록 최상단에 배치)
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'bile-duct-cancer-data.jsonl'
TOPIC_CSV_PATH = BASE_DIR / 'bertopic_bile-duct-cancer_topics.csv'
DOCS_CSV_PATH = BASE_DIR / 'bertopic_bile-duct-cancer_documents.csv'
READ_BLOCK_SIZE = 52428800
DEFAULT_NUM_CORES = 6
EMBEDDING_MODEL_NAME = "AI-Growth-Lab/PatentSBERTa"
EMBEDDING_BATCH_SIZE = 128

CUSTOM_STOP_WORDS = [
    'usa', 'korea', 'china', 'japan', 'germany', 'france', 'uk', 'india',
    'tif', 'tiff', 'jpg', 'jpeg', 'png', 'pdf', 'doc', 'docx', 'xls', 'xlsx',
    'embodiments', 'seq', 'id', 'mg', 'ml'
]
STOP_WORDS_PATTERN_STR = r'\b(' + '|'.join(CUSTOM_STOP_WORDS) + r')\b'

def extract_text(value):
    """
    다양한 데이터 타입(dict, list 등)에서 문자열만 추출하는 함수입니다.
    
    Args:
        value (any): 추출할 원본 데이터
        
    Returns:
        str: 추출된 문자열
    """
    try:
        if pd.isna(value):
            return ''
    except ValueError:
        # 배열이나 리스트 등의 경우 pd.isna 에서 에러가 발생할 수 있으므로 패스
        pass
        
    if isinstance(value, dict):
        # 딕셔너리인 경우 값들만 추출하여 공백으로 연결
        return ' '.join(str(v) for v in value.values() if v is not None)
    elif isinstance(value, (list, np.ndarray)):
        # 리스트 또는 Numpy 배열인 경우 문자열로 연결
        return ' '.join(str(v) for v in value if v is not None)
        
    return str(value) if value is not None else ''

def clean_text_worker(text_series):
    """
    텍스트 정제를 수행하는 워커(Worker) 함수입니다. (멀티프로세싱 용도)
    - 특수문자 제거, 소문자 변환, 사용자 정의 불용어 제거 및 다중 공백 단일화 처리
    
    Args:
        text_series (pd.Series): 정제 작업을 수행할 텍스트 열(Series)
        
    Returns:
        pd.Series: 정제 처리가 완료된 텍스트 열
    """
    bad_pattern = re.compile(STOP_WORDS_PATTERN_STR, flags=re.IGNORECASE)
    
    result = text_series.str.replace(r'[^a-zA-Z0-9\-\s]', ' ', regex=True).str.lower()
    result = result.str.replace(bad_pattern, '', regex=True)
    result = result.str.replace(r'\s+', ' ', regex=True).str.strip()
    return result

def parallelize_dataframe(series, func, num_cores=6):
    """
    Pandas Series에 함수를 병렬 프로세싱으로 적용하는 함수입니다.
    
    Args:
        series (pd.Series): 처리할 데이터가 담긴 Pandas Series
        func (function): 각 프로세스에서 수행할 함수 (ex. clean_text_worker)
        num_cores (int): 병렬 처리에 사용될 코어 수 (기본값: 6)
        
    Returns:
        pd.Series: 병렬 처리가 완료되어 병합된 Pandas Series
    """
    if series.empty:
        return series

    if num_cores <= 1:
        return func(series)

    num_cores = min(num_cores, len(series))
    print(f"Parallelizing with {num_cores} cores...")
    # 데이터를 코어 수에 맞게 분할
    series_split = np.array_split(series, num_cores)
    
    with Pool(num_cores) as pool:
        results = pool.map(func, series_split)
    
    return pd.concat(results)

def _extract_max_probability(prob_array):
    """
    다차원 확률 배열에서 최대 확률 값을 추출하는 내부(Helper) 함수입니다.
    
    Args:
        prob_array (list or array): 토픽별 확률 값 배열
        
    Returns:
        float: 배열 중 제일 높은(max) 확률 값. 배열이 비었거나 불량할 경우 NaN 반환.
    """
    try:
        if prob_array is None:
            return float('nan')
        arr = np.array(prob_array)
        if arr.size == 0:
            return float('nan')
        return float(np.max(arr))
    except Exception:
        return float('nan')

def _load_jsonl(path):
    read_options = paj.ReadOptions(block_size=READ_BLOCK_SIZE)
    table = paj.read_json(path, read_options=read_options)
    patent_data = table.to_pandas()
    return patent_data.set_index(patent_data.columns[0])

def _remove_family_patents(patent_data):
    family_members_col = None
    for col in ['Simple Family Members', 'Extended Family Members']:
        if col in patent_data.columns:
            family_members_col = col
            break

    if family_members_col:
        before_count = len(patent_data)
        if 'Earliest Priority Date' in patent_data.columns:
            # 먼저 출원된(우선권) 순으로 정렬하여 대표 특허를 남김
            patent_data = patent_data.sort_values('Earliest Priority Date')

        lens_col = 'Lens ID' if 'Lens ID' in patent_data.columns else None

        # Family 정보가 비어있으면 대체 Key 값을 사용
        if lens_col:
            family_key = patent_data[family_members_col].fillna(patent_data[lens_col])
        else:
            family_key = patent_data[family_members_col].fillna(patent_data.index.astype(str))

        patent_data = patent_data.loc[~family_key.duplicated()].copy()
        print(f"Removed {before_count - len(patent_data)} family patents using {family_members_col}.")
        return patent_data

    # 멤버 컬럼이 없으면 Size 컬럼을 이용해 중복 제거 시도
    family_size_col = None
    for col in ['Simple Family Size', 'Extended Family Size']:
        if col in patent_data.columns:
            family_size_col = col
            break

    if family_size_col:
        before_count = len(patent_data)
        family_size = pd.to_numeric(patent_data[family_size_col], errors='coerce').fillna(1)
        patent_data = patent_data.loc[family_size <= 1].copy()
        print(f"Removed {before_count - len(patent_data)} family patents using {family_size_col}.")
        return patent_data

    print("Warning: no family columns found; skipping family patent removal.")
    return patent_data

def _build_text_column(patent_data):
    patent_data['text'] = (
        patent_data['abstract'].apply(extract_text) + ' ' +
        patent_data['claims'].apply(extract_text) + ' ' +
        patent_data['description'].apply(extract_text)
    ).str.strip()
    return patent_data

def _get_device():
    if torch.cuda.is_available():
        return 'cuda'
    raise SystemExit("No GPU available, exiting...")

def _build_embeddings(texts_list, device):
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    return embedding_model.encode(
        texts_list,
        show_progress_bar=True,
        batch_size=EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True
    )

def _build_bertopic_model():
    umap_model = UMAP(n_components=5, n_neighbors=50, min_dist=0, metric="cosine", random_state=42)
    vectorizer_model = CountVectorizer(stop_words="english", max_features=5000)
    hdbscan_model = HDBSCAN(min_cluster_size=30, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

    return BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=30,
        # nr_topics="auto",
        verbose=True
    )

def _save_results(bertopic_model, patent_data, topics, probabilities):
    print("Saving BERTopic results to CSV files...")

    # 총괄적인 토픽 주제 및 요약 정보 가져오기 및 저장
    try:
        topic_info = bertopic_model.get_topic_info()
    except Exception:
        topic_info = None

    if topic_info is not None:
        topic_info.to_csv(TOPIC_CSV_PATH, index=False)

    # 개별 문서가 어떤 토픽에 가장 가깝고, 그 확률(Probability)이 얼마인지 정리 후 저장
    doc_probabilities = [_extract_max_probability(prob) for prob in probabilities]
    document_result_df = pd.DataFrame({
        'text': patent_data['text'],
        'topic': topics,
        'probability': doc_probabilities
    })
    document_result_df.to_csv(DOCS_CSV_PATH, index=False)

def main():
    # [Step 1] JSON Lines 데이터를 불러와 Pandas DataFrame으로 변환
    print("Reading data files...")
    patent_data = _load_jsonl(DATA_PATH)

    # [Step 2] Family 특허 중복 제거 (동일 구조를 갖는 패밀리 특허는 1건만 남김)
    print("Removing family patents...")
    patent_data = _remove_family_patents(patent_data)

    # [Step 3] Abstract, Claims, Description을 결합하여 하나의 분석용 텍스트 열 생성
    print("Combining Abstract, Claims, and Description into a single text column...")
    patent_data = _build_text_column(patent_data)

    # [Step 4] 텍스트 데이터 정제 (병렬 처리를 통한 속도 최적화)
    print("Cleaning text data...")
    patent_data['text'] = parallelize_dataframe(
        patent_data['text'],
        clean_text_worker,
        num_cores=DEFAULT_NUM_CORES
    )

    # [Step 5] BERTopic 모델 환경 구성 및 실행
    print("Initializing BERTopic...")
    device = _get_device()

    # Sentence Transformer 임베딩 모델 로드 (CUDA 가속 활용)
    texts_list = patent_data['text'].tolist()
    embeddings = _build_embeddings(texts_list, device)

    bertopic_model = _build_bertopic_model()

    # BERTopic 훈련 및 토픽 예측 수행
    topics, probabilities = bertopic_model.fit_transform(patent_data['text'], embeddings)

    # [Step 6] 최종 결과물 CSV 파일로 저장
    _save_results(bertopic_model, patent_data, topics, probabilities)

    print("All processes completed successfully!")

# ==========================================
# 2. 메인 실행 블록 (Windows 등에서의 멀티프로세싱 안정성을 위해 필수)
# ==========================================
if __name__ == '__main__':
    main()