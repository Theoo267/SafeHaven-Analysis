import runpy
import time

scripts_to_run = [
    "src/importing_and_preparing_data.py",
    "src/global_analysis.py",
    "src/linear_regression.py",
    "src/xgboost_randomforest_kmeans.py"
]

print("\n============== SAFE HAVEN FULL PIPELINE EXECUTION ==============\n")

for script in scripts_to_run:
    print(f"\n➡️  Running script: {script}")
    runpy.run_path(script)
    print(f"✔️  Finished: {script}")
    time.sleep(1)

print("\n======================= PIPELINE COMPLETE =======================\n")