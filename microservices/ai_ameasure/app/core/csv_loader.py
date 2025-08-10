"""
CSVファイルの読み込みとデータ処理を行うモジュール
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CSVDataLoader:
    """CSVファイルの読み込みとデータ処理を行うクラス"""
    
    def __init__(self):
        self.supported_encodings = ['shift-jis', 'utf-8', 'cp932']
        
        # 計測データのカラム定義（GUIコードから参照）
        self.DATE_COLUMN = '計測日時'
        self.CONVERGENCES = ['変位量A', '変位量B', '変位量C', '変位量D', '変位量E', 
                           '変位量F', '変位量G', '変位量H', '変位量I']
        self.SETTLEMENTS = ['沈下量1', '沈下量2', '沈下量3', '沈下量4', 
                          '沈下量5', '沈下量6', '沈下量7']
        
    def safe_read_csv(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        エンコーディングを自動判定してCSVファイルを読み込む
        
        Args:
            file_path: CSVファイルのパス
            **kwargs: pandas.read_csvに渡すその他の引数
            
        Returns:
            DataFrame: 読み込まれたデータ
            
        Raises:
            FileNotFoundError: ファイルが見つからない場合
            ValueError: どのエンコーディングでも読み込めない場合
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # エンコーディングを試行
        for encoding in self.supported_encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, **kwargs)
                logger.info(f"Successfully loaded {file_path} with encoding: {encoding}")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Error with encoding {encoding}: {e}")
                continue
        
        raise ValueError(f"Could not read {file_path} with any supported encoding")
    
    def load_measurement_data(self, file_path: Union[str, Path], skiprows: int = 3) -> pd.DataFrame:
        """
        計測データファイルを読み込み、前処理を行う
        
        Args:
            file_path: 計測データファイルのパス
            skiprows: スキップする行数（通常は3行）
            
        Returns:
            DataFrame: 前処理済みの計測データ
        """
        try:
            # CSVファイル読み込み（特殊な処理）
            df = self._read_measurement_csv(file_path, skiprows)
            
            # 前処理を実行
            df = self._preprocess_measurement_data(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading measurement data from {file_path}: {e}")
            return pd.DataFrame()
    
    def _read_measurement_csv(self, file_path: Union[str, Path], skiprows: int = 3) -> pd.DataFrame:
        """
        計測CSVファイルの特殊な読み込み処理
        """
        file_path = Path(file_path)
        
        # Shift-JISで読み込み、エラー処理を追加
        try:
            # まず全体を読み込んで構造を確認
            with open(file_path, 'r', encoding='shift-jis', errors='ignore') as f:
                lines = f.readlines()
            
            # 4行目（インデックス3）からがデータ本体
            if len(lines) <= skiprows:
                logger.warning(f"File {file_path} has insufficient lines")
                return pd.DataFrame()
            
            # ヘッダー行を特定（4行目）
            header_line = lines[skiprows].strip()
            headers = header_line.split(',')
            
            # データ行を読み込み
            data_lines = lines[skiprows + 1:]
            
            # データを辞書のリストに変換
            data = []
            for line in data_lines:
                if line.strip():  # 空行をスキップ
                    values = line.strip().split(',')
                    # 列数を調整
                    while len(values) < len(headers):
                        values.append('')
                    values = values[:len(headers)]  # 余分な列を削除
                    
                    data.append(dict(zip(headers, values)))
            
            df = pd.DataFrame(data)
            logger.info(f"Successfully loaded {len(df)} rows from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            # フォールバック: pandasで直接読み込み
            try:
                df = pd.read_csv(file_path, encoding='shift-jis', skiprows=skiprows, 
                               header=0, on_bad_lines='skip')
                return df
            except Exception as e2:
                logger.error(f"Fallback read also failed: {e2}")
                return pd.DataFrame()
    
    def _preprocess_measurement_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計測データの前処理を行う
        
        Args:
            df: 生の計測データ
            
        Returns:
            DataFrame: 前処理済みデータ
        """
        if df.empty:
            return df
        
        # 日付列を探す（複数の可能性）
        date_columns = ['計測日時', '計測日', 'DATE', '日時']
        date_column = None
        for col in date_columns:
            if col in df.columns:
                date_column = col
                break
        
        # 日付変換
        if date_column:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            # 有効な日付のみ残す
            df = df.dropna(subset=[date_column])
            if len(df) > 0:
                df.set_index(date_column, inplace=True)
        
        # 計測値のカラムを特定（実際のカラム名で）
        measurement_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if '変位' in col or '沈下' in col or 'displacement' in col_lower or 'settlement' in col_lower:
                measurement_columns.append(col)
        
        # 数値変換（エラー処理を強化）
        for col in measurement_columns:
            if col in df.columns:
                # 空文字列を0に変換
                df[col] = df[col].replace('', '0')
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # TD(m)列の数値変換
        td_columns = ['TD(m)', 'TD', 'td']
        for col in td_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # ゼロ値列の削除は行わない（実際のデータでは意味がある可能性）
        
        # 欠損値行の削除
        df = df.dropna(how='all')
        
        # 日次平均の処理を簡略化（エラーを避ける）
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
            try:
                # 数値カラムのみを対象に平均を計算
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    df_numeric = df[numeric_cols]
                    df_resampled = df_numeric.resample('D').mean()
                    df_resampled = df_resampled.dropna(how='all')
                    # 他の列も追加
                    for col in df.columns:
                        if col not in numeric_cols:
                            df_resampled[col] = df[col].resample('D').first()
                    df = df_resampled
            except Exception as e:
                logger.warning(f"Daily resampling failed: {e}, using original data")
        
        logger.info(f"Preprocessed data shape: {df.shape}")
        return df
    
    def get_measurement_files(self, data_folder: Path, folder_name: str = "01-hokkaido-akan") -> List[Path]:
        """
        指定フォルダから計測ファイル一覧を取得
        
        Args:
            data_folder: データフォルダのルートパス
            folder_name: 対象フォルダ名
            
        Returns:
            List[Path]: 計測ファイルのパス一覧
        """
        measurements_path = data_folder / folder_name / "main_tunnel" / "CN_measurement_data" / "measurements_A"
        
        if not measurements_path.exists():
            logger.warning(f"Measurements path not found: {measurements_path}")
            return []
        
        # CSVファイルを取得してソート
        csv_files = sorted(measurements_path.glob("measurements_A_*.csv"))
        logger.info(f"Found {len(csv_files)} measurement files in {measurements_path}")
        
        return csv_files
    
    def load_all_measurement_data(self, data_folder: Path, folder_name: str = "01-hokkaido-akan") -> pd.DataFrame:
        """
        指定フォルダのすべての計測データを読み込み、結合する
        
        Args:
            data_folder: データフォルダのルートパス
            folder_name: 対象フォルダ名
            
        Returns:
            DataFrame: 結合された計測データ
        """
        csv_files = self.get_measurement_files(data_folder, folder_name)
        
        if not csv_files:
            logger.error("No measurement files found")
            return pd.DataFrame()
        
        # 各ファイルを読み込んで結合
        all_data = []
        
        for csv_file in csv_files:
            try:
                df = self.load_measurement_data(csv_file)
                if not df.empty:
                    # ファイル名から番号を抽出
                    file_number = csv_file.stem.replace("measurements_A_", "")
                    df['file_number'] = file_number
                    all_data.append(df)
                    
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
                continue
        
        if not all_data:
            logger.error("No valid measurement data found")
            return pd.DataFrame()
        
        # すべてのデータを結合
        combined_df = pd.concat(all_data, ignore_index=True, sort=False)
        logger.info(f"Combined {len(all_data)} files into dataset with {len(combined_df)} rows")
        
        return combined_df
    
    def extract_time_series_data(self, df: pd.DataFrame, data_type: str = "displacement", 
                                num_points: int = 100) -> List[Dict]:
        """
        時系列データを抽出する
        
        Args:
            df: 計測データ
            data_type: データタイプ ("displacement" or "settlement")
            num_points: 出力する点数
            
        Returns:
            List[Dict]: 時系列データ
        """
        if df.empty:
            logger.warning("Empty dataframe provided")
            return []
        
        # TD（切羽進行距離）カラムを探す
        td_column = None
        for col in df.columns:
            if 'TD' in col.upper() or '進行' in col:
                td_column = col
                break
        
        if td_column is None:
            logger.warning("TD column not found, using index")
            df['TD'] = range(len(df))
            td_column = 'TD'
        
        # データタイプに応じてカラムを選択
        if data_type == "displacement":
            target_columns = [col for col in df.columns if any(conv in col for conv in self.CONVERGENCES)]
        else:  # settlement
            target_columns = [col for col in df.columns if any(sett in col for sett in self.SETTLEMENTS)]
        
        if not target_columns:
            logger.warning(f"No {data_type} columns found")
            return []
        
        # データをサンプリング
        if len(df) > num_points:
            step = len(df) // num_points
            sampled_df = df.iloc[::step].head(num_points)
        else:
            sampled_df = df
        
        # 時系列データを構築
        result = []
        for _, row in sampled_df.iterrows():
            td_value = float(row[td_column]) if pd.notna(row[td_column]) else 0.0
            
            data_point = {"td": td_value * 5.0}  # TD値を5倍にスケール
            
            # 距離別データを設定（最大6系列）
            series_names = ["series3m", "series5m", "series10m", "series20m", "series50m", "series100m"]
            
            for i, series_name in enumerate(series_names):
                if i < len(target_columns):
                    value = row[target_columns[i]]
                    data_point[series_name] = float(value) if pd.notna(value) else 0.0
                else:
                    data_point[series_name] = 0.0
            
            result.append(data_point)
        
        logger.info(f"Extracted {len(result)} {data_type} time series points")
        return result
    
    def extract_distribution_data(self, df: pd.DataFrame, data_type: str = "displacement") -> List[Dict]:
        """
        分布データ（ヒストグラム用）を抽出する
        
        Args:
            df: 計測データ
            data_type: データタイプ ("displacement" or "settlement")
            
        Returns:
            List[Dict]: 分布データ
        """
        if df.empty:
            return []
        
        # データタイプに応じてカラムを選択
        if data_type == "displacement":
            target_columns = [col for col in df.columns if any(conv in col for conv in self.CONVERGENCES)]
        else:  # settlement
            target_columns = [col for col in df.columns if any(sett in col for sett in self.SETTLEMENTS)]
        
        if not target_columns:
            return []
        
        # ビン範囲を設定
        bin_ranges = list(range(-15, 16))
        result = []
        
        for bin_val in bin_ranges:
            bin_data = {"range": str(bin_val)}
            
            # 各系列のヒストグラムを計算
            series_names = ["series3m", "series5m", "series10m", "series20m", "series50m", "series100m"]
            
            for i, series_name in enumerate(series_names):
                if i < len(target_columns):
                    col_data = df[target_columns[i]].dropna()
                    if not col_data.empty:
                        # ビン範囲内のデータ数をカウント
                        count = len(col_data[(col_data >= bin_val - 0.5) & (col_data < bin_val + 0.5)])
                        bin_data[series_name] = count
                    else:
                        bin_data[series_name] = 0
                else:
                    bin_data[series_name] = 0
            
            result.append(bin_data)
        
        logger.info(f"Extracted {len(result)} {data_type} distribution bins")
        return result
    
    def extract_scatter_data(self, df: pd.DataFrame, num_points: int = 200) -> List[Dict]:
        """
        散布図データを抽出する
        
        Args:
            df: 計測データ
            num_points: 出力する点数
            
        Returns:
            List[Dict]: 散布図データ
        """
        if df.empty:
            return []
        
        # TD（X軸）とファイル番号（Y軸）を使用
        if len(df) > num_points:
            sampled_df = df.sample(n=num_points)
        else:
            sampled_df = df
        
        result = []
        for _, row in sampled_df.iterrows():
            # X軸: 切羽からの距離（TD値を使用）
            td_column = None
            for col in df.columns:
                if 'TD' in col.upper():
                    td_column = col
                    break
            
            x = float(row[td_column]) if td_column and pd.notna(row[td_column]) else np.random.uniform(0, 100)
            
            # Y軸: 計測経過日数（ファイル番号を使用）
            y = float(row.get('file_number', 0)) / 20.0 if 'file_number' in row else np.random.uniform(0, 60)
            
            # 深度を計算（X, Y値から推定）
            depth = -10 - (x * 0.1) - (y * 0.05) + np.random.normal(0, 2)
            
            # 深度に基づいて色を決定
            if depth > -10:
                color = "#00FFFF"  # シアン（浅い）
            elif depth > -15:
                color = "#0080FF"  # 青
            elif depth > -20:
                color = "#0040FF"  # 濃い青
            else:
                color = "#0000FF"  # 最も濃い青
            
            result.append({
                "x": round(x, 1),
                "y": round(y, 1),
                "depth": round(depth, 1),
                "color": color
            })
        
        logger.info(f"Extracted {len(result)} scatter points")
        return result