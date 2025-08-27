import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import os
from sklearn.metrics import mean_squared_error, r2_score
import shap

try:
    from app.displacement import proccess_a_measure_file, DATE, CYCLE_NO, SECTION_TD, FACE_TD, TD_NO, CONVERGENCES, SETTLEMENTS, STA, DISTANCE_FROM_FACE, DAYS_FROM_START, DIFFERENCE_FROM_FINAL_CONVERGENCES, DIFFERENCE_FROM_FINAL_SETTLEMENTS
except:
    from displacement import proccess_a_measure_file, DATE, CYCLE_NO, SECTION_TD, FACE_TD,TD_NO, CONVERGENCES, SETTLEMENTS, STA, DISTANCE_FROM_FACE, DAYS_FROM_START, DIFFERENCE_FROM_FINAL_CONVERGENCES, DIFFERENCE_FROM_FINAL_SETTLEMENTS
import joblib

Y_COLUMNS = ['沈下量1', '沈下量2', '沈下量3', '変位量A', '変位量B', '変位量C']
DISTANCES_FROM_FACE = [3, 5, 10, 20, 50, 100]

def draw_charts_distance_displace(output_path, dict_df, column):
    plt.figure(figsize=(10, 6))
    axis_name = column[0].replace("1", "")
    for name, df in dict_df.items():
        try:
            df[axis_name] = df[column].mean(axis=1)
            plt.plot(df[TD_NO], df[axis_name], label=name, marker='o', linestyle='-', markersize=4)
        except Exception as e:
            print(f"Error: {e}")

    plt.title(f"{axis_name} over TD")
    plt.xlabel('TD (m)')
    plt.ylabel(f"{axis_name} (mm)")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    plt.close()

def draw_charts_histram_displace(output_path, dct_df, column):

    # Draw distribution chart for 'settlement'
    plt.figure(figsize=(10, 6))
    axis_name = column[0].replace("1", "")
    for name, values in dct_df.items():
        data = np.array(values)
        try:
            #plt.hist(df[column], bins=range(int(df[column].min()), int(df[column].max()) + 2), alpha=0.5, label=name)
            sns.histplot(data, bins=range(int(data.min()), int(data.max()) + 2), alpha=0.5, label=name, kde=True) #kdeで曲線を追加
        except Exception as e:
            print(f"Error: {e}")

    plt.title(f"{axis_name} Distribution")
    plt.xlabel(f"{axis_name} (mm)")
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    plt.close()

def draw_heatmap(output_path, df):
    plt.figure(figsize=(12, 12))
    sns.heatmap(df.corr(), annot=False, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def draw_prediction_chart(output_path, df, column):
    plt.figure(figsize=(10, 6))
    plt.plot(df[SECTION_TD], df[column], label=column, marker='x', linestyle='--', markersize=4)
    plt.plot(df[SECTION_TD], df[column + '_pred'], label=column + '_pred', marker='o', linestyle='-', markersize=4)
    plt.title(f"Comparison of '{column}' and '{column}_pred'")
    plt.xlabel(SECTION_TD)
    plt.ylabel(f"{column} (mm)")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    plt.close()

def draw_shap(output_path, model, df, y_columns, additinal=""):
    df_x, _ = create_dataset(df, y_columns)
    explainer = shap.Explainer(model, df_x)
    shap_values = explainer(df_x, check_additivity=False)
    for i, y_column in enumerate(y_columns):
        plt.figure(figsize=(12, 8))
        shap_value = shap_values.values[:, :, i]
        shap.summary_plot(shap_value, df_x, sort=False, show=False, max_display=len(df_x.columns))
        plt.title(f"SHAP of {y_column}")
        plt.savefig(os.path.join(output_path, f"shap_{y_column}_{additinal}.png"))
        plt.close()

def generate_additional_info_df(cycle_support_csv, observation_of_face_csv):
    try:
        df_cycle_support = pd.read_csv(cycle_support_csv).iloc[1:]
    except:
        df_cycle_support = pd.read_csv(cycle_support_csv, encoding='cp932').iloc[1:]
    try:
        df_observation_of_face = pd.read_csv(observation_of_face_csv)
    except:
        df_observation_of_face = pd.read_csv(observation_of_face_csv, encoding='cp932')
    # Concatenate df_cycle_support and df_observation_of_face by their first columns
    df_additional_info = pd.merge(
        df_cycle_support, 
        df_observation_of_face, 
        left_on=df_cycle_support.columns[0], 
        right_on=df_observation_of_face.columns[0], 
        how='inner'
    )
    return df_additional_info

def drop_unnecessary_columns(df_distance, index, df_additional_info, x_columns):
    df_final = df_distance[0].copy().dropna(subset=[CYCLE_NO])
    df_final[x_columns] = df_distance[index].copy().dropna(subset=[CYCLE_NO])[x_columns]
    cycles = df_additional_info[df_additional_info.columns[0]]
    for i, row in df_final.iterrows():  
        matching_index = df_additional_info.index[cycles <= row[CYCLE_NO]].max()
        df_final.loc[i, df_additional_info.columns] = df_additional_info.iloc[matching_index]
    
    df_data_only = df_final.drop(columns=[DATE, TD_NO])
    df_data_only = df_data_only.select_dtypes(exclude=['object'])
    df_data_only = df_data_only.interpolate(method='linear', axis=0, limit_direction='both')
    df_data_only = df_data_only.dropna(axis=1, how='any')
    df_data_only = df_data_only.dropna(axis=0, how='any')
    # Remove columns containing 'STA' and '断面番号'
    df_data_only = df_data_only.loc[:, ~df_data_only.columns.str.contains('ＳＴＡ*|断面番号*|St№*|年|月|日|測点*')]
    df_data_only = df_data_only.sort_values(by=SECTION_TD)
    df_data_only = df_data_only[df_data_only[SECTION_TD] > 0]
    df_data_only = df_data_only.drop(columns=[SECTION_TD, '支保№', '予備', '坑口からの距離'], errors='ignore')
    df_data_only.reset_index(inplace=True)

    return df_data_only

def convert_support_pattern_to_numeric(value):
    if isinstance(value, str):
        value = value.lower().translate(str.maketrans({
            'ａ': 'a', 'ｂ': 'b', 'ｃ': 'c', 'ｄ': 'd', 'ｅ': 'e', 'ｆ': 'f', 'ｇ': 'g', 'ｈ': 'h', 'ｉ': 'i', 'ｊ': 'j',
            'ｋ': 'k', 'ｌ': 'l', 'ｍ': 'm', 'ｎ': 'n', 'ｏ': 'o', 'ｐ': 'p', 'ｑ': 'q', 'ｒ': 'r', 'ｓ': 's', 'ｔ': 't',
            'ｕ': 'u', 'ｖ': 'v', 'ｗ': 'w', 'ｘ': 'x', 'ｙ': 'y', 'ｚ': 'z'
        }))
        if 'a' in value:
            return 1
        elif 'b' in value:
            return 2
        elif 'c' in value:
            return 3
        elif 'd' in value:
            return 4
        else:
            return -1  # その他特殊な断面
    else:
        return 0

def create_dataset(df, df_additional_info):
    common_columns = [DATE, CYCLE_NO, TD_NO, STA, SECTION_TD, FACE_TD, DISTANCE_FROM_FACE, DAYS_FROM_START]
    #additional_info_common_columns = ['支保工種', '断面名称', '支保間隔', '支保寸法', '吹付厚', 'ﾛｯｸﾎﾞﾙﾄ数', 'ﾛｯｸﾎﾞﾙﾄ長', '覆工厚', '支保№', '断面番号', '支保パターン', '土被り高さ', '岩石名', '岩石グループ', '岩石名コード']
    # #additional_info_common_columns = ['支保間隔', '支保寸法', '吹付厚', 'ﾛｯｸﾎﾞﾙﾄ数', 'ﾛｯｸﾎﾞﾙﾄ長', '覆工厚', '支保№', '断面番号', '土被り高さ', '岩石グループ', '岩石名コード']
    # additional_info_left_columns = ['左肩・評価点', '圧縮強度左肩', '風化変質左肩', '割れ目の間隔左肩', '割れ目の状態左肩', '湧水調整点左肩']
    # additional_info_crown_columns = ['天端・評価点', '圧縮強度中央', '風化変質中央', '割れ目の間隔中央', '割れ目の状態中央', '湧水調整点中央']
    # additional_info_right_columns = ['右肩・評価点', '圧縮強度右肩', '風化変質右肩', '割れ目の間隔右肩', '割れ目の状態右肩','湧水調整点右肩']
    # additional_info_columns = ['評価点', '圧縮強度', '風化変質', '割れ目の間隔', '割れ目の状態','湧水調整点']
    # 林さん案
    # onehot必要: ['支保工種', '断面名称', '支保パターン2']
    # onehot必要: ['補助工法の緒元', '増し支保工の緒元', '計測条件・状態等', 'インバートの早期障害']
    # onehot必要: '岩石グループ', '岩石名コード'
    additional_info_common_columns = ['支保寸法', '吹付厚', 'ﾛｯｸﾎﾞﾙﾄ数', 'ﾛｯｸﾎﾞﾙﾄ長', '覆工厚', '土被り高さ', '岩石グループ', '岩石名コード', '加重平均評価点']
    additional_info_common_support_columns = ['支保工種', '支保パターン2']
    additional_info_common_bit_columns = ['補助工法の緒元', '増し支保工の緒元', '計測条件・状態等', 'インバートの早期障害']
    additional_info_left_columns = ['左肩・圧縮強度', '左肩・風化変質', '左肩・割目間隔', '左肩・割目状態', '左肩割目の方向・平行', '左肩・湧水量', '左肩・劣化', '左肩・評価点']
    additional_info_crown_columns = ['天端・圧縮強度', '天端・風化変質', '天端・割目間隔', '天端・割目状態', '天端割目の方向・平行', '天端・湧水量', '天端・劣化', '天端・評価点']
    additional_info_right_columns = ['右肩・圧縮強度', '右肩・風化変質', '右肩・割目間隔', '右肩・割目状態', '右肩割目の方向・平行', '右肩・湧水量', '右肩・劣化', '右肩・評価点']
    additional_info_columns = ['圧縮強度', '風化変質', '割目間隔', '割目状態', '割目の方向・平行', '湧水量', '劣化', '評価点']
    
    # df_data_only = df_final.drop(columns=[DATE, TD_NO])
    # df_data_only = df_data_only.select_dtypes(exclude=['object'])
    # df_data_only = df_data_only.interpolate(method='linear', axis=0, limit_direction='both')
    # df_data_only = df_data_only.dropna(axis=1, how='any')
    # df_data_only = df_data_only.dropna(axis=0, how='any')
    # # Remove columns containing 'STA' and '断面番号'
    # df_data_only = df_data_only.loc[:, ~df_data_only.columns.str.contains('ＳＴＡ*|断面番号*|St№*|年|月|日|測点*')]
    # df_data_only = df_data_only.sort_values(by=SECTION_TD)
    # df_data_only = df_data_only[df_data_only[SECTION_TD] > 0]
    # df_data_only = df_data_only.drop(columns=[SECTION_TD, '支保№', '予備', '坑口からの距離'], errors='ignore')

    def decompose_columns(df, df_additional_info, targets, diff_targets):
        df_decomposed = []
        # Merge df and df_additional_info_filtered by CYCLE_NO and 'ｻｲｸﾙ'
        for i, row in df.iterrows():
            matching_index = df_additional_info.index[df_additional_info['ｻｲｸﾙ'] <= row[CYCLE_NO]].max()
            df_additional_info_filtered = df_additional_info[df_additional_info.index==matching_index]
            filtered_columns = additional_info_common_columns + \
                additional_info_left_columns + \
                additional_info_crown_columns + \
                additional_info_right_columns + \
                additional_info_common_bit_columns + \
                additional_info_common_support_columns
            df_additional_info_filtered = df_additional_info_filtered[filtered_columns]
            df.loc[i, df_additional_info_filtered.columns] = df_additional_info_filtered.values.squeeze()

        df_decomposed = pd.DataFrame()
        for i, (a, b) in enumerate(zip(targets[:3],  # 3 at the moment
                                    [additional_info_left_columns, additional_info_crown_columns, additional_info_right_columns])):
            filtered_columns = common_columns + [a] + additional_info_common_columns + additional_info_common_bit_columns + additional_info_common_support_columns + b + [diff_targets[i]]
            _df = df[filtered_columns]
            _df.rename(columns={a: a[:-1], diff_targets[i]: diff_targets[0][:-1]}, inplace=True)
            _df.rename(columns={src: tgt for src, tgt in zip(b, additional_info_columns)}, inplace=True)
            _df.loc[:, 'position_id'] = i
            df_decomposed = pd.concat([df_decomposed, _df], axis=0, ignore_index=True)

        df_decomposed[[f"{c}_bit" for c in additional_info_common_bit_columns]] = (~df_decomposed[additional_info_common_bit_columns].isna()).astype(int)
        for c in additional_info_common_support_columns:
            df_decomposed[f"{c}_numeric"] = df_decomposed[c].apply(convert_support_pattern_to_numeric)

        x_columns = df_decomposed.drop(columns=[DATE, CYCLE_NO, TD_NO, STA, SECTION_TD, FACE_TD, diff_targets[0][:-1]]).columns.to_list()
        x_columns.remove('position_id')
        x_columns = [col for col in x_columns if col not in additional_info_common_bit_columns + additional_info_common_support_columns]
        y_column = diff_targets[0][:-1]
        df_decomposed.dropna(subset=x_columns, inplace=True)
        df_decomposed.dropna(subset=y_column, inplace=True)

        return df_decomposed, x_columns, y_column

    settlement = decompose_columns(df, df_additional_info, SETTLEMENTS, DIFFERENCE_FROM_FINAL_SETTLEMENTS)
    convergence = decompose_columns(df, df_additional_info, CONVERGENCES, DIFFERENCE_FROM_FINAL_CONVERGENCES)

    return settlement, convergence

def analyize_ml(model, df_train, df_validate, x_columns, y_column):

    try:
        model.fit(df_train[x_columns], df_train[y_column])
        y_pred_train = model.predict(df_train[x_columns])
        y_pred_validate = model.predict(df_validate[x_columns])
    except ValueError as e:
        print(f"Error fitting model: {e}")
    # Evaluate the model
    mse_train = mean_squared_error(df_train[y_column].values, y_pred_train)
    r2_train = r2_score(df_train[y_column].values, y_pred_train)
    mse_validate = mean_squared_error(df_validate[y_column].values, y_pred_validate)
    r2_validate = r2_score(df_validate[y_column].values, y_pred_validate)

    df_train['pred'] = y_pred_train
    df_validate['pred'] = y_pred_validate
    
    print(f"Mean Squared Error for train: {mse_train}")
    print(f"R2 Score for train: {r2_train}")
    print(f"Mean Squared Error for validate: {mse_validate}")
    print(f"R2 Score for validate: {r2_validate}")

    metrics = {
        'mse_train': mse_train,
        'r2_train': r2_train,
        'mse_validate': mse_validate,
        'r2_validate': r2_validate
    }
    return df_train, df_validate, model, metrics

def draw_scatter_plot_distance_days_convergences(output_path, df_all, values, label):
    plt.figure(figsize=(10, 6))
    for value in values:
        scatter = plt.scatter(df_all[DISTANCE_FROM_FACE], df_all[DAYS_FROM_START], c=df_all[value], cmap='jet', alpha=0.5, s=5)
    plt.colorbar(scatter, label=label)
    plt.title(f"Scatter Plot of {DISTANCE_FROM_FACE} vs {DAYS_FROM_START}")
    plt.xlabel(f"{DISTANCE_FROM_FACE} (m)")
    plt.ylabel(DAYS_FROM_START)
    plt.grid()
    plt.savefig(os.path.join(output_path, f"scatter_distance_days_{label}.png"))
    plt.close()

def generate_dataframes(measurement_a_csvs, max_distance_from_face):
    df_all = []
    for csv_file in sorted(measurement_a_csvs):
        try:
            df = proccess_a_measure_file(csv_file, max_distance_from_face)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
        df_all.append(df)

    df_all = pd.concat(df_all)
    # Filter out rows where DISTANCE_FROM_FACE is less than or equal to -1
    df_all = df_all[df_all[DISTANCE_FROM_FACE]>=-1]
    # Filter out rows where DISTANCE_FROM_FACE is greater than 200
    df_all = df_all[df_all[DISTANCE_FROM_FACE]<=max_distance_from_face]
    settlements = [settle for settle in SETTLEMENTS if settle in df.columns]
    convergences = [conv for conv in CONVERGENCES if conv in df.columns]
    dct_df_settlement = {}
    dct_df_convergence = {}
    dct_df_td ={}
    for distance_from_face in DISTANCES_FROM_FACE:
        if max_distance_from_face < distance_from_face:
            continue
        dct_df_settlement[f"{distance_from_face}m"] = []
        dct_df_convergence[f"{distance_from_face}m"] = []
        # Filter the DataFrame for the specific distance from face
        dfs = []
        for td, _df in df_all.groupby(TD_NO):
            rows = _df[_df[DISTANCE_FROM_FACE] <= distance_from_face]
            if rows.empty:
                continue
            dfs.append(rows.iloc[-1][[TD_NO]+settlements+convergences])
            dct_df_settlement[f"{distance_from_face}m"] += rows.iloc[-1][settlements].values.tolist()
            dct_df_convergence[f"{distance_from_face}m"] += rows.iloc[-1][convergences].values.tolist()
        dct_df_td[f"{distance_from_face}m"] = pd.DataFrame(dfs).reset_index()

    return df_all, dct_df_settlement, dct_df_convergence, dct_df_td, settlements, convergences

def analyze_displacement(input_folder, output_path, model_paths, model, max_distance_from_face, td=None):
    
    os.makedirs(output_path, exist_ok=True)
    # Get all CSV files in the input folder
    measurement_a_csvs = [os.path.join(input_folder, 'measurements_A', f) for f in os.listdir(os.path.join(input_folder, 'measurements_A')) if f.endswith('.csv')]
    cycle_support_csv = os.path.join(input_folder, 'cycle_support/cycle_support.csv')
    observation_of_face_csv = os.path.join(input_folder, 'observation_of_face/observation_of_face.csv')

    df_additional_info = generate_additional_info_df(cycle_support_csv, observation_of_face_csv)
    df_additional_info.drop(columns=[STA], inplace=True)
    # Process each CSV file using the preprocess function
    df_all, dct_df_settlement, dct_df_convergence, dct_df_td, settlements, convergences = generate_dataframes(measurement_a_csvs, max_distance_from_face)
    settlement_data, convergence_data = create_dataset(df_all, df_additional_info)
    # Call the method
    draw_scatter_plot_distance_days_convergences(output_path, df_all, convergences, "変位量")
    draw_scatter_plot_distance_days_convergences(output_path, df_all, settlements, "沈下量")
    draw_charts_distance_displace(os.path.join(output_path, 'settle.png'), dct_df_td, settlements)
    draw_charts_distance_displace(os.path.join(output_path, 'conv.png'), dct_df_td, convergences)
    draw_charts_histram_displace(os.path.join(output_path, 'settle_hist2.png'), dct_df_settlement, settlements)
    draw_charts_histram_displace(os.path.join(output_path, 'conv_hist2.png'), dct_df_convergence, convergences)
    
    # analyize the final displacement data
    def process_each(model, df, x_columns, y_column, td):
        try:
            draw_heatmap(os.path.join(output_path, f"heatmap_{y_column}.png"), df[x_columns + [y_column]])
            if td is None:
                td = df[SECTION_TD].max()
            train_date = df[df[SECTION_TD] < td][DATE].max()
            df_train = df[df[DATE] <= train_date]
            df_validate = df[df[SECTION_TD] >= td]
            df_train, df_validate, model, metrics = analyize_ml(model, df_train, df_validate, x_columns, y_column)
            # Scatter plot of actual vs predicted values for training data
            def draw_scatter_plot(output_path, gt, pred, label, text):
                plt.figure(figsize=(8, 8))
                plt.scatter(gt, pred, alpha=0.5, label=label)
                plt.plot([gt.min(), gt.max()],
                        [gt.min(), gt.max()],
                        color='red', linestyle='--', label='Ideal Fit')
                plt.title(f"Actual vs Predicted for {y_column} ({label})")
                plt.xlabel(f"Actual {y_column}")
                plt.ylabel(f"Predicted {y_column}")
                plt.text(0.05, 0.95, text, 
                         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
                plt.legend()
                plt.grid()
                plt.savefig(output_path)
                plt.close()

            draw_scatter_plot(os.path.join(output_path, f"scatter_{y_column}_train.png"), df_train[y_column], df_train['pred'], label='Train Data', text=f"MSE: {metrics['mse_train']:.2f}\nR2: {metrics['r2_train']:.2f}")
            draw_scatter_plot(os.path.join(output_path, f"scatter_{y_column}_validate.png"), df_validate[y_column], df_validate['pred'], label='Validate Data', text=f"MSE: {metrics['mse_validate']:.2f}\nR2: {metrics['r2_validate']:.2f}")
            df_train['mode'] = 'train'
            df_validate['mode'] = 'validate'
            df_result = pd.concat([df_train, df_validate])

            def draw_feature_importance(output_path, model, x_columns):
                plt.figure(figsize=(10, 6))
                feature_importances = model.feature_importances_
                indices = np.argsort(feature_importances)[::-1]
                plt.bar(range(len(x_columns)), feature_importances[indices], align='center')
                plt.xticks(range(len(x_columns)), [x_columns[i] for i in indices], rotation=90)
                plt.title(f"Feature Importances for {y_column}")
                plt.xlabel("Features")
                plt.ylabel("Importance")
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()
            draw_feature_importance(os.path.join(output_path, f"feature_importance_{y_column}.png"), model, x_columns)

            #draw_shap(output_path, model, df_data_only, Y_COLUMNS, f"{day}days")
            # Draw charts comparing '沈下量1' and '沈下量1_pred'
            #draw_prediction_chart(os.path.join(output_path, f"comparison_{y_column}.png"), df_result, y_column)
            df_result.to_csv(os.path.join(output_path, f"result{y_column}.csv"))

            return model
        except Exception as e:
            print(f"Error processing each {y_column}m: {e}")

    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(process_each, distance, i) for i, distance in enumerate(DISTANCE_FROM_FACE)]
    #     concurrent.futures.wait(futures)
    for i, (df, x_columns, y_column) in enumerate([settlement_data, convergence_data]):
        # 最終変位量、沈下量モデル
        model = process_each(model, df, x_columns, y_column, td)
        joblib.dump(model, model_paths["final_value_prediction_model"][i])
        # 変位量、沈下量モデル
        y_column = x_columns[2]
        x_columns = [x for x in x_columns if x != y_column]
        model = process_each(model, df, x_columns, y_column, td)
        joblib.dump(model, model_paths["prediction_model"][i])
    return df_all

def main():
    parser = argparse.ArgumentParser(description="Process input and output paths.")
    parser.add_argument('input_folder', type=str, help="Path to the input file or directory.")
    parser.add_argument('output_path', type=str, help="Path to the output file or directory.")
    parser.add_argument('--max-distance-from-face', type=float, default=100, help="max distance from face to consider.")
    parser.add_argument('--td', type=float, default=None, help="TD value to filter the data.")
    
    args = parser.parse_args()
    
    analyze_displacement(args.input_folder, args.output_path, args.max_distance_from_face, td=args.td)

if __name__ == "__main__":
    main()