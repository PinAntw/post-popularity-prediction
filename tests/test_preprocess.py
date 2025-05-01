import os
import pandas as pd
import pickle
import subprocess

def test_preprocess_pipeline():
    # 1. 執行 preprocess.py
    print("📦 執行 preprocess.py...")
    result = subprocess.run(['python', 'data/preprocess.py'], capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ 執行失敗：", result.stderr)
        return

    # 2. 檢查輸出檔案是否存在
    assert os.path.exists("./data/train.csv"), "train.csv 不存在"
    assert os.path.exists("./data/test.csv"), "test.csv 不存在"
    assert os.path.exists("./experiments/checkpoints/encoder.pkl"), "encoder.pkl 不存在"

    # 3. 檢查 train/test csv 是否有 one-hot 欄位
    train_df = pd.read_csv("./data/train.csv")
    test_df = pd.read_csv("./data/test.csv")
    onehot_cols = [col for col in train_df.columns if col.startswith(('Category_', 'Concept_', 'Subcategory_'))]

    assert len(onehot_cols) > 0, "⚠️ one-hot 欄位沒產生"
    assert all(col in test_df.columns for col in onehot_cols), "⚠️ test.csv 缺少 one-hot 欄位"

    # 4. 測試 encoder 是否能載入
    with open("./experiments/checkpoints/encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    assert hasattr(encoder, "transform"), "encoder 格式不正確"

    print("✅ 測試成功：preprocess.py 執行正常，輸出檔案結構合理")

if __name__ == "__main__":
    test_preprocess_pipeline()
