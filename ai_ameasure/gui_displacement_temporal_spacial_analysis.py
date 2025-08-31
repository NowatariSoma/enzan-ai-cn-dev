import streamlit as st
import math
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import joblib
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from app.displacement import DURATION_DAYS
from app.displacement_temporal_spacial_analysis import analyze_displacement, Y_COLUMNS, STA, DISTANCE_FROM_FACE, TD_NO, DATE
from app.displacement_temporal_spacial_analysis import generate_additional_info_df, generate_dataframes, create_dataset


models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Linear Regression": LinearRegression(),
    "SVR": SVR(kernel='linear', C=1.0, epsilon=0.2),
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=42),
    "MLP":  MLPRegressor(hidden_layer_sizes=(100, 100, 50), max_iter=1000, random_state=42),
}
# config.jsonのパスを動的に決定
config_paths = [
    'config.json',  # 現在のディレクトリ
    '/app/config.json',  # Dockerコンテナ内
    os.path.join(os.path.dirname(__file__), 'config.json'),  # このスクリプトと同じディレクトリ
]

config = None
for config_path in config_paths:
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        break
    except FileNotFoundError:
        continue

if config is None:
    # デフォルト設定を使用
    config = {
        "selected_index": 0,
        "selected_folder": "",
        "input_folder": "/app/data"
    }

# Define the input folder
INPUT_FOLDER = config['input_folder']
OUTPUT_FOLDER = "./output"
MAX_DISTANCE_M = 200

def list_folders():
    """List all folder names in the INPUT_FOLDER."""
    if not os.path.exists(INPUT_FOLDER):
        return []
    return sorted([f for f in os.listdir(INPUT_FOLDER) if os.path.isdir(os.path.join(INPUT_FOLDER, f))])

def display_selected_folders(selected_folders):
    """Display the selected folder names."""
    return f"Selected Folders: {', '.join(selected_folders)}"

def draw_local_prediction_chart(output_path, x_measure, df_measure_y, x_predict, df_predict_y, title):
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')
    for i, c in enumerate(df_measure_y.columns):
        plt.plot(x_measure, df_measure_y[c], label=c, marker='x', linestyle='--', markersize=4, alpha=0.5, color=cmap(i))
    
    #plt.plot(x, df_measure_y, label=df_measure_y.columns, marker='x', linestyle='--', markersize=4, alpha=0.5, color=cmap.colors)
    for i, c in enumerate(df_predict_y.columns):
        plt.plot(x_predict, df_predict_y[c], label=['予測最終' + c], marker='o', linestyle='-', markersize=4, color=cmap(i))
    #plt.plot(x, df_predict_y, label=['予測' + c for c in df_measure_y.columns], marker='o', linestyle='-', markersize=4, color=cmap.colors)
    plt.title(title)
    plt.xlabel(DISTANCE_FROM_FACE)
    plt.ylabel(f"(mm)")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    plt.close()

def simulate_displacement(input_folder, a_measure_path, max_distance_from_face, daily_advance=None, distance_from_face=None, recursive=False):

    cycle_support_csv = os.path.join(input_folder, 'cycle_support/cycle_support.csv')
    observation_of_face_csv = os.path.join(input_folder, 'observation_of_face/observation_of_face.csv')

    df_additional_info = generate_additional_info_df(cycle_support_csv, observation_of_face_csv)
    df_additional_info.drop(columns=[STA], inplace=True)
    # Process each CSV file using the preprocess function
    df_all, _, _, _, settlements, convergences = generate_dataframes([a_measure_path], max_distance_from_face)
    if daily_advance and distance_from_face:
        # Create a new dataframe with the specified length and interval
        remaining_distance = max_distance_from_face - distance_from_face
        max_record = math.ceil(min(remaining_distance / daily_advance, DURATION_DAYS))
        max_record = max(max_record, 1)  # Ensure at least 1 record
        
        print(f"🔍 SIMULATION DEBUG:")
        print(f"  - distance_from_face: {distance_from_face}")
        print(f"  - max_distance_from_face: {max_distance_from_face}")
        print(f"  - daily_advance: {daily_advance}")
        print(f"  - remaining_distance: {remaining_distance}")
        print(f"  - max_record: {max_record}")
        df_all_actual = df_all[df_all[DISTANCE_FROM_FACE] < distance_from_face]
        if df_all_actual.empty:
            df_all_new = pd.DataFrame([df_all.iloc[0]] * max_record).reset_index()
        else:
            df_all_new = pd.DataFrame([df_all_actual.iloc[-1]] * max_record).reset_index()
        df_all_new[DATE] = pd.to_datetime(df_all.iloc[0][DATE]) + pd.to_timedelta(range(max_record), unit='D')

        df_all_new[DISTANCE_FROM_FACE] = distance_from_face + daily_advance * pd.Series(range(max_record))
        df_all = pd.concat([df_all_actual, df_all_new[df_all_new[DISTANCE_FROM_FACE] >= distance_from_face]], ignore_index=True).reset_index()

    settlement_data, convergence_data = create_dataset(df_all, df_additional_info)

    for i, ((df, x_columns, y_column), target) in enumerate(zip([settlement_data, convergence_data], [settlements, convergences])):
        final_model = joblib.load(model_paths["final_value_prediction_model"][i])
        model =  joblib.load(model_paths["prediction_model"][i])
        if recursive:
            # まずは沈下量・変位量を予測し、それに基づき最終沈下量・変位量を予測する
            _y_column = x_columns[2]
            _x_columns = [x for x in x_columns if x != _y_column]
            _x_columns = [x for x in _x_columns if x != y_column]
            _y_hat = model.predict(pd.DataFrame(df[_x_columns]))
            df.loc[df[DISTANCE_FROM_FACE] > distance_from_face, _y_column] = _y_hat[df[DISTANCE_FROM_FACE] > distance_from_face]

        y_hat = final_model.predict(df[x_columns])
        for position_id in df['position_id'].unique():
            df_all[f"{target[position_id]}_prediction"] = y_hat[df['position_id'] == position_id] + df_all[target[position_id]]
            
    return df_all, settlements, convergences

# Streamlit関連のコードは__main__実行時のみ
if __name__ == "__main__":
    # Streamlit app
    st.header("Select Folders from Input Directory")
    # Dropdown for folder selection
    folders = list_folders()

    selected_folder = st.selectbox("Select Folder", folders, index=0)
    csv_files = [
        f for f in os.listdir(os.path.join(INPUT_FOLDER, selected_folder, 'main_tunnel', 'CN_measurement_data', 'measurements_A'))
        if f.endswith('.csv')
    ]
    model_name = st.selectbox("Select Model", list(models.keys()))
    td = st.number_input("prediction TD(m)", value=500, step=1, min_value=0, max_value=2000, format="%d", key="td")
    max_distance_from_face = st.number_input("Max distance from cutter face", value=100, step=1, min_value=10, max_value=MAX_DISTANCE_M, format="%d", key="max_distance")
    tab1, tab2 = st.tabs(["Whole analysis", "Local analysis"])
    model_paths = {
        # 最終沈下量、変位量予測モデルのパス
        "final_value_prediction_model": [
        os.path.join(OUTPUT_FOLDER, f"model_final_settlement.pkl"),
        os.path.join(OUTPUT_FOLDER, f"model_final_convergence.pkl")
        ],
        # 沈下量、変位量予測モデルのパス
        "prediction_model": [
        os.path.join(OUTPUT_FOLDER, f"model_settlement.pkl"),
        os.path.join(OUTPUT_FOLDER, f"model_convergence.pkl")
        ]
    }
    
    try:
        with tab1:
            if st.button("Analyze Displacement"):
                st.write(f"Selected Folder: {selected_folder}")
    
                st.subheader("Displacement Analysis")
                analyze_displacement(os.path.join(INPUT_FOLDER, selected_folder, 'main_tunnel', 'CN_measurement_data'), OUTPUT_FOLDER, model_paths, models[model_name], max_distance_from_face, td=td)
                # Add tabs for better organization
    
                st.success("Analysis Complete!")
                # Display results in two columns
                col1, col2 = st.columns(2)
    
                with col1:
                    st.subheader("Displacement")
                    st.image(os.path.join(OUTPUT_FOLDER, 'conv.png'), caption="Convergence")
                    st.image(os.path.join(OUTPUT_FOLDER, 'conv_hist2.png'), caption="Distribution")
                    st.image(os.path.join(OUTPUT_FOLDER, 'scatter_distance_days_変位量.png'), caption="Temporal Spacial space")
                    st.image(os.path.join(OUTPUT_FOLDER, 'scatter_最終変位量との差分_train.png'), caption="Train data")
                    st.image(os.path.join(OUTPUT_FOLDER, 'scatter_最終変位量との差分_validate.png'), caption="Validation data")
                    st.image(os.path.join(OUTPUT_FOLDER, 'heatmap_最終変位量との差分.png'), caption="Heatmap")
                    st.image(os.path.join(OUTPUT_FOLDER, 'feature_importance_最終変位量との差分.png'), caption="Feature importance")
                with col2:
                    st.subheader("Settlement")
                    st.image(os.path.join(OUTPUT_FOLDER, 'settle.png'), caption="Settlement")
                    st.image(os.path.join(OUTPUT_FOLDER, 'settle_hist2.png'), caption="Distribution")
                    st.image(os.path.join(OUTPUT_FOLDER, 'scatter_distance_days_沈下量.png'), caption="Temporal Spacial space")
                    st.image(os.path.join(OUTPUT_FOLDER, 'scatter_最終沈下量との差分_train.png'), caption="Train data")
                    st.image(os.path.join(OUTPUT_FOLDER, 'scatter_最終沈下量との差分_validate.png'), caption="Validation data")
                    st.image(os.path.join(OUTPUT_FOLDER, 'heatmap_最終沈下量との差分.png'), caption="Heatmap")
                    st.image(os.path.join(OUTPUT_FOLDER, 'feature_importance_最終沈下量との差分.png'), caption="Feature importance")
    
        with tab2:
            ameasure_file = st.selectbox("Select Cycle Number",sorted(csv_files))
            distance_from_face = st.number_input("distance_from_face (m)", value=1.0, step=0.1, min_value=0.1, max_value=float(MAX_DISTANCE_M), format="%.1f")
            daily_advance = st.number_input("Input daily excavation advance (m/day)", value=5.0, step=0.1, min_value=0.1, max_value=50.0, format="%.1f")
    
            if st.button("Analyze Local Displacement"):
                col1, col2 = st.columns(2)
                input_folder = os.path.join(INPUT_FOLDER, selected_folder, 'main_tunnel', 'CN_measurement_data')
                a_measure_path = os.path.join(INPUT_FOLDER, selected_folder, 'main_tunnel', 'CN_measurement_data', 'measurements_A', ameasure_file)
                # prediction
                df_all, settlements, convergences = simulate_displacement(input_folder, a_measure_path, max_distance_from_face)
                td = float(df_all[TD_NO].values[0])
                cycle_no = float(os.path.basename(a_measure_path).split('_')[2].split('.')[0])
                
                settlement_prediction_path = os.path.join(OUTPUT_FOLDER, f"settlement_prediction_{cycle_no}.png")
                convergence_prediction_path = os.path.join(OUTPUT_FOLDER, f"convergence_prediction_{cycle_no}.png")
                draw_local_prediction_chart(settlement_prediction_path, df_all[DISTANCE_FROM_FACE], df_all[settlements], df_all[DISTANCE_FROM_FACE], df_all[[l + "_prediction" for l in settlements]], f"最終沈下量予測 for Cycle {cycle_no} (TD: {td})")
                draw_local_prediction_chart(convergence_prediction_path, df_all[DISTANCE_FROM_FACE], df_all[convergences], df_all[DISTANCE_FROM_FACE], df_all[[l + "_prediction" for l in convergences]], f"最終変位量予測 for Cycle {cycle_no} (TD: {td})")
    
                # simulation
                df_all_simulated, _, _ = simulate_displacement(input_folder, a_measure_path, max_distance_from_face, daily_advance, distance_from_face, recursive=True)
                df_all_simulated.to_csv(os.path.join(OUTPUT_FOLDER, f"{os.path.basename(selected_folder)}_{os.path.basename(a_measure_path)}.csv"), index=False)
                
                settlement_simulation_path = os.path.join(OUTPUT_FOLDER, f"settlement_simulation_{cycle_no}.png")
                convergence_simulation_path = os.path.join(OUTPUT_FOLDER, f"convergence_simulation_{cycle_no}.png")
                draw_local_prediction_chart(settlement_simulation_path, df_all[DISTANCE_FROM_FACE], df_all[settlements], df_all_simulated[DISTANCE_FROM_FACE], df_all_simulated[[l + "_prediction" for l in settlements]], f"最終沈下量予測 for Cycle {cycle_no} (TD: {td})")
                draw_local_prediction_chart(convergence_simulation_path, df_all[DISTANCE_FROM_FACE], df_all[convergences], df_all_simulated[DISTANCE_FROM_FACE], df_all_simulated[[l + "_prediction" for l in convergences]], f"最終変位量予測 for Cycle {cycle_no} (TD: {td})")
    
                st.subheader(f"Prediction (Actual excavation)")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(convergence_prediction_path, caption="Convergence")
                with col2:
                    st.image(settlement_prediction_path, caption="Settlement")
    
                st.subheader(f"Simulation ({daily_advance} m/day from TD:{distance_from_face}m)")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(convergence_simulation_path, caption="Convergence")
                with col2:
                    st.image(settlement_simulation_path, caption="Settlement")
                
                st.dataframe(df_all_simulated[[DISTANCE_FROM_FACE] + [l + "_prediction" for l in convergences + settlements]])
    
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
        st.rerun()