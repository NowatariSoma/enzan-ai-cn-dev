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
        self.Y_COLUMNS = ['沈下量1', '沈下量2', '沈下量3', '変位量A', '変位量B', '変位量C']
        self.DISTANCES_FROM_FACE = [3, 5, 10, 20, 50, 100]
        self.DURATION_DAYS = 90
        self.CYCLE_NO = 'ｻｲｸﾙNo'
        self.TD_NO = 'TD(m)'
        self.STA = 'STA'
        self.DATE = '計測日時'
        self.SECTION_TD = '実TD'
        self.FACE_TD = '切羽TD'
        self.CONVERGENCES = ['変位量A', '変位量B', '変位量C', '変位量D', '変位量E', '変位量F', '変位量G', '変位量H', '変位量I']
        self.SETTLEMENTS = ['沈下量1', '沈下量2', '沈下量3', '沈下量4', '沈下量5', '沈下量6', '沈下量7']
        self.CONVERGENCE_OFFSETS = ['変位量ｵﾌｾｯﾄA', '変位量ｵﾌｾｯﾄB', '変位量ｵﾌｾｯﾄC', '変位量ｵﾌｾｯﾄD', '変位量ｵﾌｾｯﾄE', '変位量ｵﾌｾｯﾄF', '変位量ｵﾌｾｯﾄG', '変位量ｵﾌｾｯﾄH', '変位量ｵﾌｾｯﾄI']
        self.SETTLEMENT_OFFSETS = ['沈下量ｵﾌｾｯﾄ1', '沈下量ｵﾌｾｯﾄ2', '沈下量ｵﾌｾｯﾄ3', '沈下量ｵﾌｾｯﾄ4', '沈下量ｵﾌｾｯﾄ5', '沈下量ｵﾌｾｯﾄ6', '沈下量ｵﾌｾｯﾄ7']
        self.DISTANCE_FROM_FACE = '切羽からの距離'
        self.DAYS_FROM_START = '計測経過日数'
        self.DIFFERENCE_FROM_FINAL_CONVERGENCES = ['最終変位量との差分A', '最終変位量との差分B', '最終変位量との差分C', '最終変位量との差分D', '最終変位量との差分E', '最終変位量との差分F', '最終変位量との差分G', '最終変位量との差分H', '最終変位量との差分I']
        self.DIFFERENCE_FROM_FINAL_SETTLEMENTS = ['最終沈下量との差分1', '最終沈下量との差分2', '最終沈下量との差分3', '最終沈下量との差分4', '最終沈下量との差分5', '最終沈下量との差分6', '最終沈下量との差分7']

    
    def generate_dataframes(self, measurement_a_csvs, max_distance_from_face):

        df_all = []
        for csv_file in sorted(measurement_a_csvs):
            try:
                df = self.proccess_a_measure_file(csv_file, max_distance_from_face)
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue
            df_all.append(df)

        df_all = pd.concat(df_all)
        # Filter out rows where DISTANCE_FROM_FACE is less than or equal to -1
        df_all = df_all[df_all[self.DISTANCE_FROM_FACE]>=-1]
        # Filter out rows where DISTANCE_FROM_FACE is greater than 200
        df_all = df_all[df_all[self.DISTANCE_FROM_FACE]<=max_distance_from_face]
        settlements = [settle for settle in self.SETTLEMENTS if settle in df.columns]
        convergences = [conv for conv in self.CONVERGENCES if conv in df.columns]
        dct_df_settlement = {}
        dct_df_convergence = {}
        dct_df_td ={}
        for distance_from_face in self.DISTANCES_FROM_FACE:
            if max_distance_from_face < distance_from_face:
                continue
            dct_df_settlement[f"{distance_from_face}m"] = []
            dct_df_convergence[f"{distance_from_face}m"] = []
            # Filter the DataFrame for the specific distance from face
            dfs = []
            for td, _df in df_all.groupby(self.TD_NO):
                rows = _df[_df[self.DISTANCE_FROM_FACE] <= distance_from_face]
                if rows.empty:
                    continue
                dfs.append(rows.iloc[-1][[self.TD_NO]+settlements+convergences])
                dct_df_settlement[f"{distance_from_face}m"] += rows.iloc[-1][settlements].values.tolist()
                dct_df_convergence[f"{distance_from_face}m"] += rows.iloc[-1][convergences].values.tolist()
            dct_df_td[f"{distance_from_face}m"] = pd.DataFrame(dfs).reset_index()

        return df_all, dct_df_settlement, dct_df_convergence, dct_df_td, settlements, convergences
    
    def proccess_a_measure_file(self, input_path, max_distance_from_face):
        # Read the CSV file, skipping the first 3 lines and using the 4th line as the header
        df = pd.read_csv(input_path, skiprows=3, encoding='shift-jis', header=0)
        df = self.preprocess(df, max_distance_from_face)
        
        return df
    
    def preprocess(self, df, max_distance_from_face):
        # Convert DATE column to datetime
        df[self.DATE] = pd.to_datetime(df[self.DATE], errors='coerce')
        df.set_index(self.DATE, inplace=True)
        # Remove columns where the sum of CONVERGENCES is 0
        for convergence in self.SETTLEMENTS + self.CONVERGENCES:
            if convergence in df.columns and df[convergence].sum() == 0:
                df.drop(columns=[convergence], inplace=True)
        # Filter rows within DURATION_DAYS from the first row -> disable for now
        start_date = df.index[0]
        end_date = start_date + pd.Timedelta(days=self.DURATION_DAYS)
        df = df[(df.index >= start_date) & (df.index <= end_date)].copy()  # Use .copy() to avoid SettingWithCopyWarning
        # Drop the STA column if it exists
        sta = df[self.STA].mode().iloc[0]
        if self.STA in df.columns:
            df = df.drop(columns=[self.STA])  # Avoid inplace operation on filtered dataframe
        # Group by day and take the daily average
        df = df.resample('D').mean()
        #df = df.interpolate(limit_direction='both', method='index').reset_index()
        # Drop rows where all values are NaN
        df = df.dropna(how='all').reset_index()
        df[self.STA] = sta
        df = df[[self.DATE, self.CYCLE_NO, self.TD_NO, self.STA, self.SECTION_TD, self.FACE_TD] + self.SETTLEMENTS[:3] + self.CONVERGENCES[:3]]
        df[self.DISTANCE_FROM_FACE] = df[self.FACE_TD] - df[self.SECTION_TD].iloc[0]
        df = df[df[self.DISTANCE_FROM_FACE]<=max_distance_from_face]
        df[self.DAYS_FROM_START] = (df[self.DATE] - df[self.DATE].min()).dt.days
        for i, settlement in enumerate(self.SETTLEMENTS):
            if settlement in df.columns:
                df[self.DIFFERENCE_FROM_FINAL_SETTLEMENTS[i]] = df[settlement].iloc[-1] - df[settlement]
        for i, convergence in enumerate(self.CONVERGENCES):
            if convergence in df.columns:
                df[self.DIFFERENCE_FROM_FINAL_CONVERGENCES[i]] = df[convergence].iloc[-1] - df[convergence]

        return df