import pandas as pd
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
DATA_FILES = [
    '2026-data.csv', '2025-data.csv', '2024-data.csv', '2023-data.csv',
    '2022-data.csv', '2021-data.csv', '2020-data.csv', '2019-data.csv',
    '2018-data.csv', '2017-data.csv'
]
BASE_DIR = Path(__file__).resolve().parent
TOPIC_CSV_PATH = BASE_DIR / 'bertopic_ForALL_topics.csv'
# DOCS_CSV_PATH = 'bertopic_ForALL_documents.csv'
DEFAULT_NUM_CORES = 6
EMBEDDING_MODEL_NAME = "AI-Growth-Lab/PatentSBERTa"
EMBEDDING_BATCH_SIZE = 128
REQUIRED_COLUMNS = [
    'Publication Date', 'Publication Year', 'Application Date',
    'Earliest Priority Date', 'Title', 'Abstract', 'Applicants',
    'Cited by Patent Count', 'CPC Classifications', 'Legal Status'
]

# 사용자 정의 불용어 (제거 대상 단어 목록) - 상수이므로 대문자 뱀표기법(Upper Snake Case) 사용
CUSTOM_STOP_WORDS = [
    'usa', 'korea', 'china', 'japan', 'germany', 'france', 'uk', 'india',
    'tif', 'tiff', 'jpg', 'jpeg', 'png', 'pdf', 'doc', 'docx', 'xls', 'xlsx',
    'embodiments', 'seq', 'id', 'mg', 'ml'
]
STOP_WORDS_PATTERN_STR = r'\b(' + '|'.join(CUSTOM_STOP_WORDS) + r')\b'

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

# def _extract_max_probability(prob_array):
#     """
#     다차원 확률 배열에서 최대 확률 값을 추출하는 내부(Helper) 함수입니다.
    
#     Args:
#         prob_array (list or array): 토픽별 확률 값 배열
        
#     Returns:
#         float: 배열 중 제일 높은(max) 확률 값. 배열이 비었거나 불량할 경우 NaN 반환.
#     """
#     try:
#         if prob_array is None:
#             return float('nan')
#         arr = np.array(prob_array)
#         if arr.size == 0:
#             return float('nan')
#         return float(np.max(arr))
#     except Exception:
#         return float('nan')

def _load_csv_files(file_list):
    dataframes = []
    for file_path in file_list:
        csv_path = BASE_DIR / file_path
        dataframes.append(pd.read_csv(csv_path, index_col=0, engine='pyarrow'))
    return dataframes

def _remove_family_patents(merged_data):
    family_members_col = None
    for col in ['Simple Family Members', 'Extended Family Members']:
        if col in merged_data.columns:
            family_members_col = col
            break

    if family_members_col:
        before_count = len(merged_data)
        if 'Earliest Priority Date' in merged_data.columns:
            # 먼저 출원된(우선권) 순으로 정렬하여 대표 특허를 남김
            merged_data = merged_data.sort_values('Earliest Priority Date')

        lens_col = 'Lens ID' if 'Lens ID' in merged_data.columns else None

        # Family 정보가 비어있으면 대체 Key 값을 사용
        if lens_col:
            family_key = merged_data[family_members_col].fillna(merged_data[lens_col])
        else:
            family_key = merged_data[family_members_col].fillna(merged_data.index.astype(str))

        merged_data = merged_data.loc[~family_key.duplicated()].copy()
        print(f"Removed {before_count - len(merged_data)} family patents using {family_members_col}.")
        return merged_data

    # 멤버 컬럼이 없으면 Size 컬럼을 이용해 중복 제거 시도
    family_size_col = None
    for col in ['Simple Family Size', 'Extended Family Size']:
        if col in merged_data.columns:
            family_size_col = col
            break

    if family_size_col:
        before_count = len(merged_data)
        family_size = pd.to_numeric(merged_data[family_size_col], errors='coerce').fillna(1)
        merged_data = merged_data.loc[family_size <= 1].copy()
        print(f"Removed {before_count - len(merged_data)} family patents using {family_size_col}.")
        return merged_data

    print("Warning: no family columns found; skipping family patent removal.")
    return merged_data

def _select_required_columns(merged_data):
    existing_cols = [c for c in REQUIRED_COLUMNS if c in merged_data.columns]
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in merged_data.columns]

    if missing_cols:
        print(f"Warning: these required columns are missing and will be skipped: {missing_cols}")

    return merged_data[existing_cols].copy()

def _build_text_column(merged_data):
    merged_data['text'] = merged_data['Title'] + ' ' + merged_data['Abstract'].fillna('')
    return merged_data

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

def _save_results(bertopic_model, merged_data, topics, probabilities):
    print("Saving BERTopic results to CSV files...")

    # 총괄적인 토픽 주제 및 요약 정보 가져오기 및 저장
    try:
        topic_info = bertopic_model.get_topic_info()
    except Exception:
        topic_info = None

    if topic_info is not None:
        topic_info.to_csv(TOPIC_CSV_PATH, index=False)

    # # 개별 문서가 어떤 토픽에 가장 가깝고, 그 확률(Probability)이 얼마인지 정리 후 저장
    # doc_probabilities = [_extract_max_probability(prob) for prob in probabilities]
    # document_result_df = pd.DataFrame({
    #     'text': merged_data['text'],
    #     'topic': topics,
    #     'probability': doc_probabilities
    # })
    # document_result_df.to_csv(DOCS_CSV_PATH, index=False)

def main():
    print("Reading data files...")
    # [Step 1] 연도별 데이터 CSV 파일 로드
    dataframes = _load_csv_files(DATA_FILES)

    # [Step 2] 로드된 연도별 데이터를 하나의 통합 DataFrame으로 합침
    print("Concatenating data...")
    merged_data = pd.concat(dataframes, ignore_index=True)

    # [Step 3] Family 특허 중복 제거 (동일 구조를 갖는 패밀리 특허는 1건만 남김)
    print("Removing family patents...")
    merged_data = _remove_family_patents(merged_data)

    print("Data Count")
    print(merged_data.shape[0])

    # [Step 4] 분석에 필요한 필수 속성(Column)만 추출
    print("Selecting required columns...")
    merged_data = _select_required_columns(merged_data)

    # [Step 5] Title(제목)과 Abstract(요약)를 결합하여 하나의 분석용 텍스트 열 생성
    print("Combining Title and Abstract into a single text column...")
    merged_data = _build_text_column(merged_data)

    # [Step 6] 텍스트 데이터 정제 (병렬 처리를 통한 속도 최적화)
    # 영문 외 문자 및 특수문자, 불용어 등을 클렌징 처리
    print("Cleaning text data...")
    merged_data['text'] = parallelize_dataframe(
        merged_data['text'],
        clean_text_worker,
        num_cores=DEFAULT_NUM_CORES
    )

    # [Step 7] BERTopic 모델 환경 구성 및 실행
    print("Initializing BERTopic...")
    device = _get_device()

    # Sentence Transformer 임베딩 모델 로드 (CUDA 가속 활용)
    texts_list = merged_data['text'].tolist()
    embeddings = _build_embeddings(texts_list, device)

    bertopic_model = _build_bertopic_model()

    # BERTopic 훈련 및 토픽 예측 수행
    topics, probabilities = bertopic_model.fit_transform(merged_data['text'], embeddings)

    # [Step 8] 최종 결과물 CSV 파일로 저장
    _save_results(bertopic_model, merged_data, topics, probabilities)

    print("All processes completed successfully!")


# ==========================================
# 2. 메인 실행 블록 (Windows 등에서의 멀티프로세싱 안정성을 위해 필수)
# ==========================================
if __name__ == '__main__':
    main()