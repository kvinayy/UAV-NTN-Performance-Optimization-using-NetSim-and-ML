import os
import time
import glob
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ========= FIXED PATHS =========

WORKSPACE_DIR = r"C:\Users\Student\Documents\NetSim\Workspaces\UAVTest_24_11\UAV_1User_Scenario"
LOG_DIR       = os.path.join(WORKSPACE_DIR, "log")
CONFIG_PATH   = os.path.join(WORKSPACE_DIR, "Configuration.netsim")

DATA_PATH_OL  = r"C:\Users\Student\Desktop\IMT2023608\UAV\UAV_1_Node\uav_runs_dataset_ol.csv"
RESULT_PATH   = r"C:\Users\Student\Desktop\IMT2023608\UAV\UAV_1_Node\open_loop_results.csv"

NETSIM_EXE    = r"C:\Program Files\NetSim\Standard_v14_3\bin\bin_x64\NetSimCore.exe"
APP_PATH      = r"C:\Program Files\NetSim\Standard_v14_3\bin\bin_x64"
LICENSE       = "5053@172.16.124.40"

SIM_TIME_S    = 100.0


# ========= HELPER =========

def find_col(cols, *keywords):
    for c in cols:
        name = str(c).lower()
        if all(k.lower() in name for k in keywords):
            return c
    return None


# ========= LOAD SMALL DATASET =========

def load_ol_dataset():
    df = pd.read_csv(DATA_PATH_OL)
    print(f"[INFO] Loaded OPEN LOOP dataset with {len(df)} rows")

    X = df[["Packet_Size_Bytes", "IAT_us"]].values
    y = df["Throughput_Mbps"].values

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    return df, model


# ========= ML PREDICTION (ONCE) =========

def predict_once(model):
    packet_sizes = [256, 512, 1024]
    iats = [10000, 15000, 20000, 30000]

    best_thr = -1
    best_pkt, best_iat = None, None

    print("\n[ML] Candidate predictions:")
    for pkt in packet_sizes:
        for iat in iats:
            pred = model.predict([[pkt, iat]])[0]
            print(f" ({pkt}, {iat}) => {pred:.4f} Mbps")
            if pred > best_thr:
                best_thr = pred
                best_pkt, best_iat = pkt, iat

    print(f"\n[ML] Selected for OPEN LOOP: PACKET={best_pkt}, IAT={best_iat}")
    return best_pkt, best_iat


# ========= WRITE CONFIG =========

def update_config(pkt, iat):
    tree = ET.parse(CONFIG_PATH)
    root = tree.getroot()

    for node in root.findall(".//APPLICATION"):
        if node.get("NAME") == "App1_CBR":
            node.find("PACKET_SIZE").set("VALUE", str(pkt))
            node.find("INTER_ARRIVAL_TIME").set("VALUE", str(iat))
            break

    tree.write(CONFIG_PATH)
    print("[CONFIG] Updated.")


# ========= RUN NETSIM ONCE =========

def run_netsim():
    cmd = [
        NETSIM_EXE,
        "-apppath", APP_PATH,
        "-iopath", WORKSPACE_DIR,
        "-license", LICENSE
    ]
    print("[SIM] Running NetSim...")
    subprocess.run(cmd, cwd=WORKSPACE_DIR)
    print("[SIM] NetSim finished.")


# ========= READ METRICS =========

def read_metrics():

    app_logs = glob.glob(os.path.join(LOG_DIR, "Application_Packet_Log.*"))
    if not app_logs:
        raise RuntimeError("Application_Packet_Log not found!")

    file = app_logs[0]
    df = pd.read_excel(file) if file.endswith("xlsx") else pd.read_csv(file)
    if df.empty:
        raise RuntimeError("Empty Application_Packet_Log!")

    cols = df.columns

    size_col = find_col(cols, "size")
    delay_col = find_col(cols, "latency")
    jitter_col = find_col(cols, "jitter")
    rx_col = find_col(cols, "rx", "time")

    total_bytes = df[size_col].astype(float).sum()

    if rx_col:
        tmin = df[rx_col].astype(float).min()
        tmax = df[rx_col].astype(float).max()
        duration = max((tmax - tmin) / 1000, 1e-6)
    else:
        duration = SIM_TIME_S

    thr = (total_bytes * 8) / (duration * 1e6)
    delay = df[delay_col].mean()
    jitter = df[jitter_col].mean()

    print(f"[METRICS] Thr={thr:.4f} Mbps | Delay={delay:.2f} | Jitter={jitter:.2f}")

    return thr, delay, jitter


# ========= SAVE RESULT =========

def save_result(pkt, iat, thr, delay, jitter):
    df = pd.DataFrame([{
        "Packet_Size": pkt,
        "IAT_us": iat,
        "Throughput_Mbps": thr,
        "Delay_us": delay,
        "Jitter_us": jitter
    }])
    df.to_csv(RESULT_PATH, index=False)
    print(f"[RESULT] Saved to {RESULT_PATH}")


# ========= MAIN (OPEN LOOP ML) =========

def main():
    print("\n========== OPEN LOOP ML START ==========")

    df, model = load_ol_dataset()

    pkt, iat = predict_once(model)
    update_config(pkt, iat)

    run_netsim()
    time.sleep(2)

    thr, delay, jitter = read_metrics()
    save_result(pkt, iat, thr, delay, jitter)

    print("========== OPEN LOOP ML COMPLETE ==========\n")


if __name__ == "__main__":
    main()
