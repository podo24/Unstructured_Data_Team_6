import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Csv 데이터를 DataFrame으로 읽음.
# 2. Title과 Abstract 열에서 암 이름별로 개수를 카운트함 (단, 유의어도 고려함)
# 3. 암 이름별로 개수를 시각화함

# 1. Csv 데이터를 DataFrame으로 읽음.
BASE_DIR = Path(__file__).resolve().parent
DATA_FILES = [
    BASE_DIR / '2026-data.csv', BASE_DIR / '2025-data.csv',
    BASE_DIR / '2024-data.csv', BASE_DIR / '2023-data.csv',
    BASE_DIR / '2022-data.csv', BASE_DIR / '2021-data.csv',
    BASE_DIR / '2020-data.csv', BASE_DIR / '2019-data.csv',
    BASE_DIR / '2018-data.csv', BASE_DIR / '2017-data.csv'
]
SYNONYMS_FILE = BASE_DIR / 'CANCER_SYNONYMS.json'

def _load_csv_files(file_list):
    missing = [str(path) for path in file_list if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing CSV files: {', '.join(missing)}")

    dataframes = []
    for file_path in file_list:
        dataframes.append(pd.read_csv(file_path, index_col=0, engine='pyarrow'))
    return dataframes

Data = _load_csv_files(DATA_FILES)

# 2. Title과 Abstract 열에서 암 이름별로 개수를 카운트함 (단, 유의어도 고려함)
def _load_cancer_synonyms(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")

    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)

CANCER_SYNONYMS = _load_cancer_synonyms(SYNONYMS_FILE)

# '-'를 공백으로 대체하는 함수
def normalize_text(text):
    return str(text).replace('-', ' ').lower()

# 암 이름별로 개수를 카운트하는 함수)
def count_cancer_occurrences(dataframes, cancer_synonyms):
    cancer_counts = {cancer: 0 for cancer in cancer_synonyms.keys()}
    
    for df in dataframes:
        for index, row in df.iterrows():
            title = normalize_text(row['Title'])
            abstract = normalize_text(row['Abstract'])
            text = title + ' ' + abstract
            matched_cancers = set()

            for cancer, synonyms in cancer_synonyms.items():
                if any(str(synonym).lower() in text for synonym in synonyms):
                    matched_cancers.add(cancer)

            for cancer in matched_cancers:
                cancer_counts[cancer] += 1
    
    return cancer_counts

cancer_counts = count_cancer_occurrences(Data, CANCER_SYNONYMS)

# 내림차 순으로 정렬
cancer_counts = dict(sorted(cancer_counts.items(), key=lambda item: item[1], reverse=True))
print(cancer_counts)

# 3. 암 이름별로 개수를 시각화 후 저장
def visualize_cancer_counts(cancer_counts):
    cancers = list(cancer_counts.keys())
    counts = list(cancer_counts.values())
    
    plt.figure(figsize=(12, 6))
    plt.bar(cancers, counts, color='skyblue')
    plt.xlabel('Cancer Type')
    plt.ylabel('Count')
    plt.title('Count of Cancer Occurrences in Title and Abstract')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1000)
    plt.tight_layout()
    plt.savefig(BASE_DIR / 'cancer_counts.png')

visualize_cancer_counts(cancer_counts)