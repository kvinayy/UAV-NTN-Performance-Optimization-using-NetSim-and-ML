import os
import time
import glob
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# =====================================================
# FIXED PATHS
# =====================================================

WORKSPACE_DIR = r"C:\Users\Student\Documents\NetSim\Workspaces\UAVTest_24_11\UAV_1User_Scenario"
LOG_DIR       = os.path.join(WORKSPACE_DIR, "log")
CONFIG_PATH   = os.path.join(WORKSPACE_DIR, "Configuration.netsim")

DATA_PATH     = r"C:\Users\Student\Desktop\IMT2023608\UAV\UAV_1_Node\uav_runs_dataset.csv"

NETSIM_EXE    = r"C:\Program Files\NetSim\Standard_v14_3\bin\bin_x64\NetSimCore.exe"
APP_PATH      = r"C:\Program Files\NetSim\Standard_v14_3\bin\bin_x64"
LICENSE       = "5053@172.16.124.40"

SIM_TIME_S    = 100.0


# =====================================================
# HELPER FUNCTION
# =====================================================

def find_col(cols, *keywords):
    for c in cols:
        name = str(c).lower()
        if all(k.lower() in name for k in keywords):
            return c
    return None


# =====================================================
# LOAD DATASET AND TRAIN MODEL
# =====================================================

def load_dataset_and_model():
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Dataset size: {len(df)} rows")

    X = df[["Packet_Size_Bytes", "IAT_us"]].values
    y = df["Throughput_Mbps"].values

    if len(df) > 3:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
    else:
        X_train, y_train = X, y
        X_val, y_val = X, y

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)

    if len(df) > 3:
        print(f"[MODEL] Validation R² = {model.score(X_val, y_val):.4f}")

    return df, model


# =====================================================
# SELECT NEXT PARAMETERS (PURE ML)
# =====================================================

def choose_best_parameters(model):
    packet_sizes = [256, 512, 1024]
    iats = [10000, 15000, 20000, 30000]

    best_thr = -np.inf
    best_pkt, best_iat = None, None

    print("[MODEL] Evaluating candidate parameters:")
    for pkt in packet_sizes:
        for iat in iats:
            pred = model.predict([[pkt, iat]])[0]
            print(f"  Packet={pkt}, IAT={iat} → Predicted Thr={pred:.4f} Mbps")

            if pred > best_thr:
                best_thr = pred
                best_pkt = pkt
                best_iat = iat

    print(f"[MODEL] Selected → Packet={best_pkt}, IAT={best_iat}")
    return best_pkt, best_iat


# =====================================================
# UPDATE NETSIM CONFIGURATION
# =====================================================

def update_config(pkt, iat):
    tree = ET.parse(CONFIG_PATH)
    root = tree.getroot()

    for app in root.findall(".//APPLICATION"):
        if app.get("NAME") == "App1_CBR":
            app.find("PACKET_SIZE").set("VALUE", str(pkt))
            app.find("INTER_ARRIVAL_TIME").set("VALUE", str(iat))
            break

    tree.write(CONFIG_PATH)
    print("[CONFIG] NetSim configuration updated.")


# =====================================================
# RUN NETSIM
# =====================================================

def run_netsim():
    cmd = [
        NETSIM_EXE,
        "-apppath", APP_PATH,
        "-iopath", WORKSPACE_DIR,
        "-license", LICENSE
    ]

    print("[SIM] Running NetSim...")
    subprocess.run(cmd, cwd=WORKSPACE_DIR)
    print("[SIM] NetSim execution completed.")


# =====================================================
# READ PERFORMANCE METRICS
# =====================================================

def read_metrics():
    logs = glob.glob(os.path.join(LOG_DIR, "Application_Packet_Log.*"))
    if not logs:
        raise RuntimeError("Application_Packet_Log not found!")

    file = logs[0]
    df = pd.read_csv(file) if file.endswith(".csv") else pd.read_excel(file)

    if df.empty:
        raise RuntimeError("Application_Packet_Log is empty!")

    size_col   = find_col(df.columns, "size")
    delay_col  = find_col(df.columns, "delay")
    jitter_col = find_col(df.columns, "jitter")
    rx_col     = find_col(df.columns, "rx", "time")

    total_bytes = df[size_col].astype(float).sum()

    if rx_col:
        t_start = df[rx_col].min()
        t_end = df[rx_col].max()
        duration = max((t_end - t_start) / 1000.0, 1e-6)
    else:
        duration = SIM_TIME_S

    throughput = (total_bytes * 8) / (duration * 1e6)
    delay = df[delay_col].mean()
    jitter = df[jitter_col].mean()

    print(f"[METRICS] Thr={throughput:.4f} Mbps | Delay={delay:.2f} µs | Jitter={jitter:.2f} µs")

    return throughput, delay, jitter


# =====================================================
# APPEND RESULTS TO DATASET
# =====================================================

def append_to_dataset(df, pkt, iat, thr, delay, jitter):
    new_row = {
        "Packet_Size_Bytes": pkt,
        "IAT_us": iat,
        "Throughput_Mbps": thr,
        "Delay_us": delay,
        "Jitter_us": jitter
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)
    print("[DATA] New iteration appended to dataset.")
    return df


# =====================================================
# MAIN CLOSED-LOOP EXECUTION
# =====================================================

def main():
    print("===== CLOSED-LOOP ML (1-USER) START =====")

    ITERATIONS = 10
    thr_list, delay_list, jitter_list = [], [], []

    for it in range(1, ITERATIONS + 1):
        print(f"\n========== ITERATION {it} ==========")

        df, model = load_dataset_and_model()
        pkt, iat = choose_best_parameters(model)

        update_config(pkt, iat)
        run_netsim()
        time.sleep(2)

        thr, delay, jitter = read_metrics()

        thr_list.append(thr)
        delay_list.append(delay)
        jitter_list.append(jitter)

        df = append_to_dataset(df, pkt, iat, thr, delay, jitter)

        # Live plot
        plt.clf()
        plt.subplot(3, 1, 1)
        plt.plot(thr_list, marker='o')
        plt.title("Throughput (Mbps)")

        plt.subplot(3, 1, 2)
        plt.plot(delay_list, marker='o')
        plt.title("Delay (µs)")

        plt.subplot(3, 1, 3)
        plt.plot(jitter_list, marker='o')
        plt.title("Jitter (µs)")

        plt.tight_layout()
        plt.pause(0.5)

    print("\n===== CLOSED-LOOP ML FINISHED =====")
    plt.show()


if __name__ == "__main__":
    main()
