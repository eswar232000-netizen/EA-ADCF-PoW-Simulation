import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

# Fix backend
matplotlib.use('TkAgg')

# -------------------------------
# CONFIGURATION
# -------------------------------
INITIAL_NODES = 50
MAX_NODES = 100
MIN_NODES = 10
SIMULATION_STEPS = 200
RUNS = 5

TARGET_BLOCK_TIME = 10
ALPHA = 0.3
ENERGY_PER_HASH = 1e-9

np.random.seed(42)

# -------------------------------
# METRICS FUNCTION
# -------------------------------
def compute_metrics(block_time, energy, throughput):
    return {
        "Mean Block Time": np.mean(block_time),
        "Std Block Time": np.std(block_time),
        "Block Time Error": np.mean(np.abs(np.array(block_time) - TARGET_BLOCK_TIME)),
        "Mean Energy": np.mean(energy),
        "Std Energy": np.std(energy),
        "Mean Throughput": np.mean(throughput),
        "Std Throughput": np.std(throughput)
    }

# -------------------------------
# SIMULATION FUNCTION
# -------------------------------
def run_simulation():

    hash_rates = np.random.uniform(5, 15, INITIAL_NODES)

    difficulty_adaptive = 1000
    predicted_hash_rate = np.sum(hash_rates)
    difficulty_static = 1000

    energy_adaptive = []
    difficulty_adaptive_list = []
    block_time_adaptive = []
    throughput_adaptive = []

    energy_static = []
    block_time_static = []
    throughput_static = []

    time_axis = []

    for t in range(SIMULATION_STEPS):

        # Node Join / Leave
        if np.random.rand() < 0.1:
            change = np.random.randint(-5, 6)
            new_size = np.clip(len(hash_rates) + change, MIN_NODES, MAX_NODES)

            if new_size > len(hash_rates):
                new_nodes = np.random.uniform(5, 15, new_size - len(hash_rates))
                hash_rates = np.append(hash_rates, new_nodes)
            else:
                hash_rates = hash_rates[:new_size]

        # Hash rate fluctuation
        hash_rates *= np.random.uniform(0.95, 1.05, len(hash_rates))

        # Sudden spikes
        if t % 50 == 0 and t != 0:
            spike_factor = np.random.uniform(1.5, 2.5)
            hash_rates *= spike_factor

        total_hash_rate = np.sum(hash_rates)

        # -------- Adaptive Model --------
        predicted_hash_rate = ALPHA * total_hash_rate + (1 - ALPHA) * predicted_hash_rate

        block_time_a = difficulty_adaptive / total_hash_rate
        predicted_block_time = difficulty_adaptive / predicted_hash_rate

        difficulty_adaptive = difficulty_adaptive * (TARGET_BLOCK_TIME / predicted_block_time)

        energy_a = total_hash_rate * block_time_a * ENERGY_PER_HASH
        throughput_a = 1 / block_time_a

        # -------- Static Model --------
        block_time_s = difficulty_static / total_hash_rate
        energy_s = total_hash_rate * block_time_s * ENERGY_PER_HASH
        throughput_s = 1 / block_time_s

        # Store
        energy_adaptive.append(energy_a)
        difficulty_adaptive_list.append(difficulty_adaptive)
        block_time_adaptive.append(block_time_a)
        throughput_adaptive.append(throughput_a)

        energy_static.append(energy_s)
        block_time_static.append(block_time_s)
        throughput_static.append(throughput_s)

        time_axis.append(t)

    return {
        "time": time_axis,
        "adaptive": {
            "block_time": block_time_adaptive,
            "energy": energy_adaptive,
            "throughput": throughput_adaptive,
            "difficulty": difficulty_adaptive_list
        },
        "static": {
            "block_time": block_time_static,
            "energy": energy_static,
            "throughput": throughput_static
        }
    }

# -------------------------------
# MULTIPLE RUNS
# -------------------------------
adaptive_results = []
static_results = []

print("\n===== INDIVIDUAL RUN RESULTS =====\n")

for i in range(RUNS):
    sim = run_simulation()

    adaptive_metrics = compute_metrics(
        sim["adaptive"]["block_time"],
        sim["adaptive"]["energy"],
        sim["adaptive"]["throughput"]
    )

    static_metrics = compute_metrics(
        sim["static"]["block_time"],
        sim["static"]["energy"],
        sim["static"]["throughput"]
    )

    print(f"\n--- Run {i+1} ---")
    run_df = pd.DataFrame([adaptive_metrics, static_metrics],
                          index=["Adaptive", "Static"])
    print(run_df)

    adaptive_results.append(adaptive_metrics)
    static_results.append(static_metrics)

# -------------------------------
# FINAL AVERAGE TABLE
# -------------------------------
adaptive_df = pd.DataFrame(adaptive_results)
static_df = pd.DataFrame(static_results)

final_df = pd.DataFrame({
    "Adaptive (EA-ADCF)": adaptive_df.mean(),
    "Static": static_df.mean()
})

print("\n===== FINAL AVERAGED RESULTS =====\n")
print(final_df)

# Save CSV
final_df.to_csv("final_results.csv")

# -------------------------------
# STEP-WISE TABLE (DISPLAY SAMPLE)
# -------------------------------
sim = run_simulation()

step_df = pd.DataFrame({
    "Time": sim["time"],
    "BlockTime_Adaptive": sim["adaptive"]["block_time"],
    "BlockTime_Static": sim["static"]["block_time"],
    "Energy_Adaptive": sim["adaptive"]["energy"],
    "Energy_Static": sim["static"]["energy"],
    "Throughput_Adaptive": sim["adaptive"]["throughput"],
    "Throughput_Static": sim["static"]["throughput"]
})

print("\n===== SAMPLE STEP-WISE DATA (FIRST 10 ROWS) =====\n")
print(step_df.head(10))

step_df.to_csv("detailed_simulation.csv")

# -------------------------------
# PLOTS
# -------------------------------

time_axis = sim["time"]

# Energy
plt.figure()
plt.plot(time_axis, sim["adaptive"]["energy"], label="Adaptive")
plt.plot(time_axis, sim["static"]["energy"], linestyle='--', label="Static")
plt.title("Energy per Block vs Time")
plt.xlabel("Time")
plt.ylabel("Energy")
plt.legend()
plt.grid()
plt.show()

# Difficulty
plt.figure()
plt.plot(time_axis, sim["adaptive"]["difficulty"])
plt.title("Difficulty vs Block Height")
plt.xlabel("Block Height")
plt.ylabel("Difficulty")
plt.grid()
plt.show()

# Block Stability
plt.figure()
plt.plot(time_axis, sim["adaptive"]["block_time"], label="Adaptive")
plt.plot(time_axis, sim["static"]["block_time"], linestyle='--', label="Static")
plt.axhline(y=TARGET_BLOCK_TIME, label="Target")
plt.title("Block Interval Stability")
plt.xlabel("Time")
plt.ylabel("Block Time")
plt.legend()
plt.grid()
plt.show()

# Throughput
plt.figure()
plt.plot(time_axis, sim["adaptive"]["throughput"], label="Adaptive")
plt.plot(time_axis, sim["static"]["throughput"], linestyle='--', label="Static")
plt.title("Throughput vs Time")
plt.xlabel("Time")
plt.ylabel("Throughput")
plt.legend()
plt.grid()
plt.show()