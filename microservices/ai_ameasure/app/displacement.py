import argparse

import japanize_matplotlib
import matplotlib.pyplot as plt
import pandas as pd

DURATION_DAYS = 90
CYCLE_NO = "ｻｲｸﾙNo"
TD_NO = "TD(m)"
STA = "STA"
DATE = "計測日時"
SECTION_TD = "実TD"
FACE_TD = "切羽TD"
CONVERGENCES = [
    "変位量A",
    "変位量B",
    "変位量C",
    "変位量D",
    "変位量E",
    "変位量F",
    "変位量G",
    "変位量H",
    "変位量I",
]
SETTLEMENTS = ["沈下量1", "沈下量2", "沈下量3", "沈下量4", "沈下量5", "沈下量6", "沈下量7"]
CONVERGENCE_OFFSETS = [
    "変位量ｵﾌｾｯﾄA",
    "変位量ｵﾌｾｯﾄB",
    "変位量ｵﾌｾｯﾄC",
    "変位量ｵﾌｾｯﾄD",
    "変位量ｵﾌｾｯﾄE",
    "変位量ｵﾌｾｯﾄF",
    "変位量ｵﾌｾｯﾄG",
    "変位量ｵﾌｾｯﾄH",
    "変位量ｵﾌｾｯﾄI",
]
SETTLEMENT_OFFSETS = [
    "沈下量ｵﾌｾｯﾄ1",
    "沈下量ｵﾌｾｯﾄ2",
    "沈下量ｵﾌｾｯﾄ3",
    "沈下量ｵﾌｾｯﾄ4",
    "沈下量ｵﾌｾｯﾄ5",
    "沈下量ｵﾌｾｯﾄ6",
    "沈下量ｵﾌｾｯﾄ7",
]
DISTANCE_FROM_FACE = "切羽からの距離"
DAYS_FROM_START = "計測経過日数"
DIFFERENCE_FROM_FINAL_CONVERGENCES = [
    "最終変位量との差分A",
    "最終変位量との差分B",
    "最終変位量との差分C",
    "最終変位量との差分D",
    "最終変位量との差分E",
    "最終変位量との差分F",
    "最終変位量との差分G",
    "最終変位量との差分H",
    "最終変位量との差分I",
]
DIFFERENCE_FROM_FINAL_SETTLEMENTS = [
    "最終沈下量との差分1",
    "最終沈下量との差分2",
    "最終沈下量との差分3",
    "最終沈下量との差分4",
    "最終沈下量との差分5",
    "最終沈下量との差分6",
    "最終沈下量との差分7",
]


def draw_charts(df):
    # Plot CONVERGENCES
    plt.figure(figsize=(10, 6))
    for convergence in CONVERGENCES:
        if convergence in df.columns:
            plt.plot(df[DATE], df[convergence], label=convergence)
    plt.title("Convergences Over Time")
    plt.xlabel("Date")
    plt.ylabel("Convergence Values")
    plt.legend()
    plt.grid()
    plt.savefig("output/convergences.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot SETTLEMENTS
    plt.figure(figsize=(10, 6))
    for settlement in SETTLEMENTS:
        if settlement in df.columns:
            plt.plot(df[DATE], df[settlement], label=settlement)
    plt.title("Settlements Over Time")
    plt.xlabel("Date")
    plt.ylabel("Settlement Values")
    plt.legend()
    plt.grid()
    plt.savefig("output/settlements.png", dpi=300, bbox_inches="tight")
    plt.close()


def preprocess(df, max_distance_from_face):
    # Convert DATE column to datetime
    df[DATE] = pd.to_datetime(df[DATE], errors="coerce")
    df.set_index(DATE, inplace=True)
    # Remove columns where the sum of CONVERGENCES is 0
    for convergence in SETTLEMENTS + CONVERGENCES:
        if convergence in df.columns and df[convergence].sum() == 0:
            df.drop(columns=[convergence], inplace=True)
    # Filter rows within DURATION_DAYS from the first row -> disable for now
    start_date = df.index[0]
    end_date = start_date + pd.Timedelta(days=DURATION_DAYS)
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    # Drop the STA column if it exists
    sta = df[STA].mode().iloc[0]
    if STA in df.columns:
        df.drop(columns=[STA], inplace=True)
    # Group by day and take the daily average
    df = df.resample("D").mean()
    # df = df.interpolate(limit_direction='both', method='index').reset_index()
    # Drop rows where all values are NaN
    df = df.dropna(how="all").reset_index()
    df[STA] = sta
    df = df[[DATE, CYCLE_NO, TD_NO, STA, SECTION_TD, FACE_TD] + SETTLEMENTS[:3] + CONVERGENCES[:3]]
    df[DISTANCE_FROM_FACE] = df[FACE_TD] - df[SECTION_TD].iloc[0]
    df = df[df[DISTANCE_FROM_FACE] <= max_distance_from_face]
    df[DAYS_FROM_START] = (df[DATE] - df[DATE].min()).dt.days
    for i, settlement in enumerate(SETTLEMENTS):
        if settlement in df.columns:
            df[DIFFERENCE_FROM_FINAL_SETTLEMENTS[i]] = df[settlement].iloc[-1] - df[settlement]
    for i, convergence in enumerate(CONVERGENCES):
        if convergence in df.columns:
            df[DIFFERENCE_FROM_FINAL_CONVERGENCES[i]] = df[convergence].iloc[-1] - df[convergence]

    return df


def proccess_a_measure_file(input_path, max_distance_from_face):
    # Read the CSV file, skipping the first 3 lines and using the 4th line as the header
    df = pd.read_csv(input_path, skiprows=3, encoding="shift-jis", header=0)
    df = preprocess(df, max_distance_from_face)

    draw_charts(df)

    return df


def main():
    parser = argparse.ArgumentParser(description="Process input and output paths.")
    parser.add_argument("input_path", type=str, help="Path to the input file or directory.")
    parser.add_argument("output_path", type=str, help="Path to the output file or directory.")
    parser.add_argument(
        "--max_distance",
        type=float,
        default=100.0,
        help="Maximum distance from face (default: 100.0)",
    )

    args = parser.parse_args()

    df = proccess_a_measure_file(args.input_path, args.max_distance)
    # Save the processed DataFrame to the output path
    df.to_csv(args.output_path, index=False)

    print(f"Input Path: {args.input_path}")
    print(f"Output Path: {args.output_path}")
    print(f"Max Distance: {args.max_distance}")


if __name__ == "__main__":
    main()
