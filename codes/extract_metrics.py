import pandas as pd
from pathlib import Path
from io import StringIO

METRICS_FILE = Path(r"C:\Users\Student\Desktop\IMT2023608\UAV\UAV_1_Node\MetricsPrint11.csv")
DATASET_FILE = Path(r"C:\Users\Student\Desktop\IMT2023608\UAV\UAV_1_Node\uav_runs_dataset.csv")


def read_application_metrics(csv_path: Path):
    """Reads Application_Metrics by manually indexing columns."""
    with csv_path.open("r", errors="ignore") as f:
        lines = f.readlines()

    # find Application_Metrics line
    start = None
    for i, line in enumerate(lines):
        if "Application_Metrics" in line:
            start = i
            break

    if start is None:
        raise ValueError("Application_Metrics not found!")

    # SKIP 2 ROWS:
    #   1 = blank commas
    #   2 = column header
    data_lines = lines[start + 3 :]

    # Load CSV with NO HEADER
    raw_df = pd.read_csv(StringIO("".join(data_lines)), header=None)

    # First TRUE data row
    row = raw_df.iloc[0]

    # manually map columns
    df = pd.DataFrame([{
        "App_ID": row[0],
        "App_Name": row[1],
        "Src_ID": row[2],
        "Dest_ID": row[3],
        "GenRate_Mbps": float(row[4]),
        "Pkts_Gen": int(row[5]),
        "Pkts_Recv": int(row[6]),
        "Payload_Gen_B": int(row[7]),
        "Payload_Recv_B": int(row[8]),
        "Throughput_Mbps": float(row[9]),
        "Delay_us": float(row[10]),
        "Jitter_us": float(row[11]),
    }])

    return df


def append_run_to_dataset(df, run_name):
    row = df.iloc[0].copy()

    # calculate packet size safely
    pkt_size = row["Payload_Gen_B"] / max(row["Pkts_Gen"], 1)

    # calculate IAT (us) from Gen Rate
    IAT_us = (8 * pkt_size / row["GenRate_Mbps"]) * 1e6

    # add metadata
    row["Packet_Size_Bytes"] = pkt_size
    row["IAT_us"] = IAT_us
    row["run_name"] = run_name

    out_df = pd.DataFrame([row])

    if DATASET_FILE.exists():
        out_df.to_csv(DATASET_FILE, mode="a", index=False, header=False)
    else:
        out_df.to_csv(DATASET_FILE, index=False, header=True)


def main():
    print(f"\nReading metrics from: {METRICS_FILE}")
    df = read_application_metrics(METRICS_FILE)

    print("\n=== Extracted Metrics ===")
    print(df[["Throughput_Mbps", "Delay_us", "Jitter_us"]])

    append_run_to_dataset(df, "run_1_default")

    print(f"\nSaved → {DATASET_FILE}")


if __name__ == "__main__":
    main()
