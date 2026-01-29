import os
import glob
import pandas as pd
import hashlib
from tqdm.auto import tqdm
import shutil
from pathlib import Path

def analyze_and_merge_datasets_fixed(paths):
    all_data = []
    print("--- 1단계: 각 데이터셋별 정밀 분석 ---")
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    
    for i, p in enumerate(paths, 1):
        if not os.path.exists(p):
            print(f"경로 없음: {p}")
            continue
            
        current_dataset_files = []
        for ext in extensions:
            current_dataset_files.extend(glob.glob(os.path.join(p, '**', ext), recursive=True))
        
        for file_path in current_dataset_files:
            class_name = os.path.basename(os.path.dirname(file_path))
            # 레이블 통합 로직 적용 (사용자 요청)
            clean_label = 'TLight' if class_name in ['Traffic Light', 'TLight', 'Traffic Lights'] else class_name
            
            all_data.append({
                'path': file_path,
                'original_label': class_name,
                'clean_label': clean_label,
                'source': f'dataset_{i}'
            })
        print(f"[Dataset {i}] 발견된 총 이미지: {len(current_dataset_files)}개 (Path: {p})")

    df = pd.DataFrame(all_data)
    if df.empty:
        print("\n❌ 에러: 이미지를 하나도 찾지 못했습니다. 경로를 확인하세요.")
        return df

    print("\n--- 2단계: 중복 제거 (Path 기준) ---")
    initial_count = len(df)
    df = df.drop_duplicates(subset=['path'])
    print(f"- 중복 제거 전: {initial_count}")
    print(f"- 중복 제거 후 (Path 기준): {len(df)}")
    
    return df

def advanced_clean(df):
    print("\n--- 3단계: 고급 클리닝 (의미 없는 레이블 제거 및 해싱) ---")
    # 1. 의미 없는 레이블 및 데이터 부족 클래스 삭제
    exclude_labels = [
        'Other', 'data', 'Recaptcha Dataset', 'Google_Recaptcha_V2_Images_Dataset', 
        'images', 'Mountain'  # 데이터가 너무 적은 클래스 drop
    ]
    df = df[~df['clean_label'].isin(exclude_labels)]
    print(f"- 레이블 필터링 후: {len(df)}")
    
    # 2. 이미지 해시(MD5)로 진짜 중복 제거
    def get_hash(path):
        try:
            with open(path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return None
    
    print("Hashing images for deep duplicate removal...")
    tqdm.pandas()
    df['hash'] = df['path'].progress_apply(get_hash)
    df = df.dropna(subset=['hash'])
    df = df.drop_duplicates(subset=['hash'])
    print(f"- 최종 유효 이미지 (해시 기준): {len(df)}")
    
    print("\n--- 4단계: 최종 통합 클래스 분포 (Top 15) ---")
    print(df['clean_label'].value_counts().head(15))
    
    return df

def reorganize_dataset(df, target_base_dir):
    print(f"\n--- 5단계: 데이터셋 재구성 (Target: {target_base_dir}) ---")
    target_base_dir = Path(target_base_dir)
    if target_base_dir.exists():
        shutil.rmtree(target_base_dir)
    target_base_dir.mkdir(parents=True)
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Copying files"):
        class_dir = target_base_dir / row['clean_label']
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일명 중복 방지를 위해 해시값 사용하거나 원본 유지
        src_path = Path(row['path'])
        dest_path = class_dir / f"{row['hash']}{src_path.suffix}"
        
        # 실제 복사
        shutil.copy2(src_path, dest_path)
    
    print(f"✅ 데이터셋 재구성 완료: {target_base_dir}")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    raw_paths = [
        str(project_root / "data/raw/google-recaptcha"),
        str(project_root / "data/raw/google-recaptcha-v2"),
        str(project_root / "data/raw/test-dataset")
    ]
    
    final_df = analyze_and_merge_datasets_fixed(raw_paths)
    if not final_df.empty:
        cleaned_df = advanced_clean(final_df)
        reorganize_dataset(cleaned_df, project_root / "data/processed/v3")
