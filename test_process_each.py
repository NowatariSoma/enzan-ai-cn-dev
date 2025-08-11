#!/usr/bin/env python3
"""
process-eachエンドポイントのテストスクリプト
make-datasetからデータを取得し、process-eachで処理を実行
"""

import requests
import json
import logging
from typing import Dict, Any

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# APIベースURL
BASE_URL = "http://localhost:8000/api/v1"

def call_make_dataset(folder_name: str = "01-hokkaido-akan", max_distance: float = 100.0) -> Dict[str, Any]:
    """
    make-datasetエンドポイントを呼び出してデータセットを生成
    """
    url = f"{BASE_URL}/measurements/make-dataset"
    payload = {
        "folder_name": folder_name,
        "max_distance_from_face": max_distance
    }
    
    logger.info(f"Calling make-dataset with: {payload}")
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # データの概要をログ出力
        logger.info(f"Settlement data records: {len(data.get('settlement_data', []))}")
        logger.info(f"Convergence data records: {len(data.get('convergence_data', []))}")
        
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling make-dataset: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response content: {e.response.text}")
        raise

def call_process_each(
    model_name: str,
    data_type: str,
    folder_name: str = "01-hokkaido-akan",
    max_distance: float = 100.0,
    td: float = None,
    predict_final: bool = True
) -> Dict[str, Any]:
    """
    process-eachエンドポイントを呼び出してモデル処理を実行
    """
    url = f"{BASE_URL}/models/process-each"
    payload = {
        "model_name": model_name,
        "folder_name": folder_name,
        "max_distance_from_face": max_distance,
        "data_type": data_type,
        "predict_final": predict_final
    }
    
    if td is not None:
        payload["td"] = td
    
    logger.info(f"Calling process-each with: {payload}")
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # 結果の概要をログ出力
        logger.info(f"Model: {data.get('model_name')}")
        logger.info(f"Data type: {data.get('data_type')}")
        logger.info(f"Train count: {data.get('train_count')}")
        logger.info(f"Validate count: {data.get('validate_count')}")
        
        # メトリクスを表示
        metrics = data.get('metrics', {})
        logger.info("Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
        
        # 特徴量重要度のトップ5を表示
        feature_importance = data.get('feature_importance', {})
        if feature_importance:
            logger.info("Top 5 important features:")
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for feature, importance in sorted_features:
                logger.info(f"  {feature}: {importance:.4f}")
        
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling process-each: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response content: {e.response.text}")
        raise

def get_available_models() -> list:
    """
    利用可能なモデル一覧を取得
    """
    url = f"{BASE_URL}/models/"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        models = data.get('models', [])
        return [model['name'] for model in models]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting models: {e}")
        return []

def main():
    """
    メイン処理
    """
    logger.info("=" * 50)
    logger.info("Starting process-each endpoint test")
    logger.info("=" * 50)
    
    # パラメータ設定
    folder_name = "01-hokkaido-akan"
    max_distance = 100.0
    
    # Step 1: 利用可能なモデルを取得
    logger.info("\n--- Step 1: Getting available models ---")
    available_models = get_available_models()
    if available_models:
        logger.info(f"Available models: {available_models}")
    else:
        logger.error("Failed to get available models")
        return
    
    # Step 2: make-datasetでデータを生成
    logger.info("\n--- Step 2: Calling make-dataset ---")
    try:
        dataset_result = call_make_dataset(folder_name, max_distance)
        
        # データが存在するか確認
        has_settlement = len(dataset_result.get('settlement_data', [])) > 0
        has_convergence = len(dataset_result.get('convergence_data', [])) > 0
        
        if not has_settlement and not has_convergence:
            logger.error("No data returned from make-dataset")
            return
    except Exception as e:
        logger.error(f"Failed to call make-dataset: {e}")
        return
    
    # Step 3: process-eachで各モデル・データタイプを処理
    logger.info("\n--- Step 3: Calling process-each ---")
    
    # テストするモデルとデータタイプの組み合わせ
    test_cases = []
    
    # 利用可能なデータタイプに応じてテストケースを追加
    if has_settlement:
        for model in available_models[:2]:  # 最初の2つのモデルでテスト
            test_cases.append({
                "model_name": model,
                "data_type": "settlement",
                "predict_final": True
            })
    
    if has_convergence:
        for model in available_models[:2]:  # 最初の2つのモデルでテスト
            test_cases.append({
                "model_name": model,
                "data_type": "convergence",
                "predict_final": True
            })
    
    # 各テストケースを実行
    results = []
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n--- Test case {i}/{len(test_cases)} ---")
        logger.info(f"Model: {test_case['model_name']}, Data type: {test_case['data_type']}")
        
        try:
            result = call_process_each(
                model_name=test_case["model_name"],
                data_type=test_case["data_type"],
                folder_name=folder_name,
                max_distance=max_distance,
                predict_final=test_case["predict_final"]
            )
            results.append({
                "test_case": test_case,
                "success": True,
                "result": result
            })
            logger.info(f"✓ Test case {i} completed successfully")
        except Exception as e:
            logger.error(f"✗ Test case {i} failed: {e}")
            results.append({
                "test_case": test_case,
                "success": False,
                "error": str(e)
            })
    
    # 結果サマリー
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    successful_tests = sum(1 for r in results if r["success"])
    logger.info(f"Total tests: {len(results)}")
    logger.info(f"Successful: {successful_tests}")
    logger.info(f"Failed: {len(results) - successful_tests}")
    
    # 詳細結果を保存
    output_file = "process_each_test_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\nDetailed results saved to: {output_file}")
    
    logger.info("\nTest completed!")

if __name__ == "__main__":
    main()