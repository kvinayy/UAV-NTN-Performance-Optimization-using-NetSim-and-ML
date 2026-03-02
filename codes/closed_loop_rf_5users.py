import os
import time
import glob
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ================================
# FILE PATHS
# ================================
WORKSPACE_DIR = r"C:\Users\Student\Documents\NetSim\Workspaces\UAVTest_24_11\UAV_1User_Scenario"
LOG_DIR       = os.path.join(WORKSPACE_DIR, "log")
CONFIG_PATH   = os.path.join(WORKSPACE_DIR, "Configuration.netsim")

DATA_PATH = r"C:\Users\Student\Desktop\IMT2023608\UAV\UAV_1_Node\uav_runs_dataset_5user.xlsx.csv"
OUTPUT_DATA = r"C:\Users\Student\Desktop\IMT2023608\UAV\UAV_1_Node\uav_runs_5users_closedloop.csv"

NETSIM_EXE = r"C:\Program Files\NetSim\Standard_v14_3\bin\bin_x64\NetSimCore.exe"
APP_PATH   = r"C:\Program Files\NetSim\Standard_v14_3\bin\bin_x64"
LICENSE    = "5053@172.16.124.40"

SIM_TIME_S = 100.0


# ================================
# HELPERS
# ================================
def find_col(cols, *keys):
    for c in cols:
        name = str(c).lower()
        if all(k.lower() in name for k in keys):
            return c
    return None


# ================================
# LOAD DATASET
# ================================
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded 5-user dataset with {len(df)} rows")

    # ML inputs and outputs
    X = df[["Packet_Size_Bytes", "IAT_us"]].values
    y = df["Throughput_Mbps"].values      # <=== FIXED

    if len(df) >= 4:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
    else:
        X_train, y_train = X, y
        X_val, y_val = X, y

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    if len(df) >= 4:
        print(f"[MODEL] Validation R² = {model.score(X_val, y_val):.4f}")
    else:
        print("[MODEL] Not enough rows for validation split.")

    return df, model


# ================================
# CHOOSE NEXT PARAMETERS
# ================================
def choose_next_params(model, iteration):
    pkt_sizes = [256, 512, 1024]
    iats = [10000, 15000, 20000, 30000]

    best_thr = -999
    best_pkt = None
    best_iat = None

    print("\n[MODEL] Candidate predictions:")
    for p in pkt_sizes:
        for i in iats:
            pred = model.predict([[p, i]])[0]
            print(f" ({p}, {i}) => {pred:.4f} Mbps")

            if pred > best_thr:
                best_thr = pred
                best_pkt = p
                best_iat = i

    print(f"[MODEL] Selected PACKET={best_pkt}, IAT={best_iat}")
    return best_pkt, best_iat


# ================================
# UPDATE NETSIM CONFIG
# ================================
def update_config(pkt, iat):
    tree = ET.parse(CONFIG_PATH)
    root = tree.getroot()

    apps = root.findall(".//APPLICATION")

    for app in apps:
        app.find("PACKET_SIZE").set("VALUE", str(pkt))
        app.find("INTER_ARRIVAL_TIME").set("VALUE", str(iat))

    tree.write(CONFIG_PATH)
    print("[CONFIG] Updated all 5 users.")


# ================================
# RUN NETSIM
# ================================
def run_netsim():
    cmd = [
        NETSIM_EXE,
        "-apppath", APP_PATH,
        "-iopath", WORKSPACE_DIR,
        "-license", LICENSE
    ]

    print("[SIM] Running NetSim...")
    subprocess.run(cmd, cwd=WORKSPACE_DIR)
    print("[SIM] Finished.")


# ================================
# READ METRICS
# ================================
def read_metrics():
    print(f"[METRICS] Reading logs from: {LOG_DIR}")

    # Load file
    logs = glob.glob(os.path.join(LOG_DIR, "Application_Packet_Log.*"))
    if not logs:
        raise RuntimeError("Application_Packet_Log not found!")

    file = logs[0]
    df = pd.read_csv(file) if file.endswith(".csv") else pd.read_excel(file)

    if df.empty:
        raise RuntimeError("Application_Packet_Log is empty!")

    # ---- FIXED COLUMN NAMES BASED ON YOUR FILE ----
    size_col   = "Packet or Segment size(Bytes)"
    start_col  = "Packet or Segment Start Time(ms)"
    end_col    = "Packet or Segment End Time(ms)"
    delay_col  = "Latency(Microseconds)"
    jitter_col = "Jitter(Microseconds)"

    # ---- Compute throughput using end-start ----
    total_bytes = df[size_col].astype(float).sum()

    t_start = df[start_col].astype(float).min()
    t_end   = df[end_col].astype(float).max()

    # Convert ms → seconds
    duration_sec = max((t_end - t_start) / 1000.0, 1e-6)

    throughput = (total_bytes * 8) / (duration_sec * 1e6)  # Mbps

    # Delay & Jitter
    delay = df[delay_col].mean()
    jitter = df[jitter_col].mean()

    print(f"[METRICS] Thr={throughput:.4f} Mbps | Delay={delay:.2f} | Jitter={jitter:.2f}")

    return throughput, delay, jitter



# ================================
# APPEND TO DATASET
# ================================
def append_row(df, pkt, iat, thr, delay, jitter):
    new_row = {
        "Packet_Size_Bytes": pkt,
        "IAT_us": iat,
        "Throughput_Mbps": thr,
        "Delay_us": delay,
        "Jitter_us": jitter
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)
    df.to_csv(OUTPUT_DATA, index=False)

    print("[DATA] Row added.")
    return df


# ================================
# MAIN LOOP
# ================================
def main():
    print("===== 5-USER CLOSED LOOP START =====")
    ITER = 10

    thr_list, d_list, j_list = [], [], []

    for it in range(1, ITER + 1):
        print(f"\n====== ITERATION {it} ======")

        df, model = load_dataset()

        pkt, iat = choose_next_params(model, it)
        update_config(pkt, iat)

        run_netsim()
        time.sleep(3)

        thr, delay, jitter = read_metrics()

        df = append_row(df, pkt, iat, thr, delay, jitter)

        thr_list.append(thr)
        d_list.append(delay)
        j_list.append(jitter)

    # plot
    iters = list(range(1, ITER + 1))
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(iters, thr_list, marker='o')
    plt.title("5-User Closed-Loop Throughput")
    plt.ylabel("Mbps")

    plt.subplot(3, 1, 2)
    plt.plot(iters, d_list, marker='o')
    plt.title("Delay")
    plt.ylabel("us")

    plt.subplot(3, 1, 3)
    plt.plot(iters, j_list, marker='o')
    plt.title("Jitter")
    plt.ylabel("us")
    plt.xlabel("Iteration")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
