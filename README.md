# UAV-NTN Performance Optimization using NetSim and Machine Learning

This repository presents an implementation of **machine learning driven performance optimization for UAV-based Non-Terrestrial Networks (NTNs)** using **NetSim simulation and Python-based closed-loop automation**.

The project demonstrates how adaptive Machine Learning can dynamically optimize communication parameters to improve **throughput, delay, and jitter** in next-generation **6G NTN communication systems**.

---

## Motivation

Non-Terrestrial Networks (NTNs) are a key component of future 6G communication systems, enabling connectivity beyond traditional terrestrial infrastructure.

NTNs provide communication support for:

- Remote and rural connectivity
- Disaster recovery networks
- Maritime and aviation communication
- Defense and tactical systems
- Massive IoT deployments

Unmanned Aerial Vehicles (UAVs) acting as aerial base stations offer flexible and rapidly deployable coverage.  
However, network performance strongly depends on traffic configuration parameters.

This work introduces a **Machine Learning based adaptive optimization framework** integrated with NetSim to automatically tune these parameters.

---

## Project Overview

The project integrates:

- UAV-based NTN simulation (NetSim)
- Dataset generation from simulation logs
- Machine Learning modelling
- Python–NetSim automation
- Iterative closed-loop optimization

Two optimization approaches are evaluated.

---

## Open-Loop Machine Learning

Open-loop optimization follows a traditional workflow:

1. Perform multiple NetSim simulations manually.
2. Extract performance metrics.
3. Train a Random Forest regression model.
4. Predict optimal traffic parameters.
5. Apply prediction once.

### Limitation
- No feedback mechanism
- Performance depends on initial dataset quality

---

## Closed-Loop Machine Learning (Proposed Framework)

A closed-loop system was developed where Python automatically interacts with NetSim.

### Automation Workflow

Initial Dataset
↓
Train Random Forest Model
↓
Predict Packet Size & IAT
↓
Update NetSim XML Configuration
↓
Run NetSim Simulation
↓
Extract Throughput / Delay / Jitter
↓
Append Dataset
↓
Retrain Model

The system iteratively learns optimal parameters and converges toward improved network performance.

---

## ⚙️ NTN Simulation Setup

According to the NetSim topology (Report – Fig.1):

- UAV acts as an aerial Base Station
- Ground users communicate through UAV
- UDP-based CBR traffic model
- UAV mobility defined using trace files

Two scenarios were evaluated:

Single-user NTN link  
Five-user shared UAV network  

---

## Performance Metrics

Network performance is evaluated using:

| Metric        | Description                        |
|---------------|------------------------------------|
| Throughput    | Successfully delivered data rate   |
| End-to-End Delay | Packet transmission latency     |
| Jitter        | Packet arrival time variation      |

These metrics provide a complete evaluation of link efficiency and stability.

---

## Results

### 1-User Scenario
Closed-loop optimization achieved:

- Throughput: **0.65 → 0.78 Mbps**
- Delay: **1500 → 1350 µs**
- Jitter: **360 → 290 µs**

The ML model converged rapidly due to strong UAV Line-of-Sight conditions.

---

### 5-User Scenario
Under higher traffic load:

- Throughput: **0.42 → 0.55 Mbps**
- Delay: **1650 → 1500 µs**
- Jitter: **410 → 330 µs**

Performance improvement confirms realistic load-dependent NTN behaviour.

---

## Closed-Loop Optimization Insight

The learning algorithm consistently converged to:
Packet Size = 1024 Bytes
Inter-Arrival Time = 10000 µs

demonstrating stable adaptive optimization across different user densities.

---

## Technologies Used

- Python
- Random Forest Regression
- NetSim Network Simulator
- UAV Communication Modeling
- Machine Learning Optimization
- 6G NTN Concepts

---

## Repository Structure

codes/ → ML models & NetSim automation scripts
datasets/ → Generated simulation datasets
results/ → Performance plots and logs
REPORT.pdf → Detailed technical report
REFERENCES.md → Literature references
README.md

---

## Future Work

Potential extensions include:

- Multi-UAV cooperative NTN systems
- Reinforcement learning based optimization
- UAV trajectory optimization
- Interference-aware NTN–TN coexistence
- Digital Twin enabled NTN optimization
- 3GPP Release-17/18 compliant NTN modeling

---

## References

Relevant literature and standards are listed in [REFERENCES.md](REFERENCES.md)

---

## Author

**Vinay Kusumanchi**  
IMTech - Electronics & Communication Engineering  

---

## Key Contribution

This work demonstrates an end-to-end workflow combining:

Simulation → Dataset Creation → Machine Learning → Automated Feedback → Network Optimization

and validates the feasibility of **ML-driven adaptive optimization for UAV-based NTN systems** aligned with emerging 6G research directions.
