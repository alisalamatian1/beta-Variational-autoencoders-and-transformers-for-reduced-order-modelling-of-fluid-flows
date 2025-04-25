# import re
# import pandas as pd

# # Load and split the log file
# with open("metrics.log", "r") as f:
#     content = f.read()

# reports = content.split("-" * 20)
# results = []

# for report in reports:
#     if not report.strip():
#         continue

#     # Extract metrics
#     rmse = re.search(r"RMSE:\s*([\d.]+)", report)
#     nrmse = re.search(r"nRMSE:\s*([\d.]+)", report)
#     csv_err = re.search(r"CSV Error:\s*([\d.]+)", report)
#     max_err = re.search(r"Max Error:\s*([\d.]+)", report)
#     bd_rmse = re.search(r"Boundary RMSE:\s*([\d.]+)", report)
#     fourier_err = re.search(r"Fourier Error:\s*tensor\(\[([^\]]+)\]", report)

#     # Extract hyperparameters
#     latent_dim = re.search(r"latent_dim:\s*(\d+)", report)
#     epochs = re.search(r"epochs:\s*(\d+)", report)
#     beta = re.search(r"beta:\s*([\deE.-]+)", report)
#     batch_size = re.search(r"batch_size:\s*(\d+)", report)
#     lr = re.search(r"\blr:\s*([\deE.-]+)", report)

#     results.append({
#         "latent_dim": int(latent_dim.group(1)) if latent_dim else None,
#         "epochs": int(epochs.group(1)) if epochs else None,
#         "beta": float(beta.group(1)) if beta else None,
#         "batch_size": int(batch_size.group(1)) if batch_size else None,
#         "lr": float(lr.group(1)) if lr else None,
#         "RMSE": float(rmse.group(1)) if rmse else None,
#         "nRMSE": float(nrmse.group(1)) if nrmse else None,
#         "CSV Error": float(csv_err.group(1)) if csv_err else None,
#         "Max Error": float(max_err.group(1)) if max_err else None,
#         "Boundary RMSE": float(bd_rmse.group(1)) if bd_rmse else None,
#         "Fourier Error": [float(v.strip()) for v in fourier_err.group(1).split(",")] if fourier_err else [None, None, None]
#     })

# # Expand Fourier components
# for r in results:
#     f = r.pop("Fourier Error")
#     r["Fourier1"], r["Fourier2"], r["Fourier3"] = f

# # Create dataframe
# df = pd.DataFrame(results)
# df.to_csv("parsed_metrics.csv", index=False)

# # Generate top-3 lowest error entries for each error type
# top_k = 3
# metrics = ["RMSE", "nRMSE", "CSV Error", "Max Error", "Boundary RMSE", "Fourier1", "Fourier2", "Fourier3"]
# top_results = []

# for metric in metrics:
#     top = df.nsmallest(top_k, metric)
#     top["Metric"] = metric
#     top["ErrorValue"] = top[metric]
#     top_results.append(top)

# top_df = pd.concat(top_results)
# top_df = top_df[["Metric", "ErrorValue", "latent_dim", "epochs", "beta", "batch_size", "lr"] + metrics]
# top_df.to_csv("top3_errors.csv", index=False)

# print("Saved: parsed_metrics_larger_run.csv and top3_errors_larger_run.csv")

import re
import pandas as pd

# Load and split the log file
with open("metrics_hyperparam_tuning_both_vae_easy.log", "r") as f:
    content = f.read()

reports = content.split("-" * 20)
results = []

for report in reports:
    if not report.strip():
        continue

    # Extract metrics
    rmse = re.search(r"RMSE:\s*([\d.]+)", report)
    nrmse = re.search(r"nRMSE:\s*([\d.]+)", report)
    csv_err = re.search(r"CSV Error:\s*([\d.]+)", report)
    max_err = re.search(r"Max Error:\s*([\d.]+)", report)
    bd_rmse = re.search(r"Boundary RMSE:\s*([\d.]+)", report)
    fourier_err = re.search(r"Fourier Error:\s*tensor\(\[([^\]]+)\]", report)

    # Extract hyperparameters
    latent_dim = re.search(r"latent_dim:\s*(\d+)", report)
    epoch = re.search(r"Epoch:\s*(\d+)", report)
    beta = re.search(r"beta:\s*([\deE.-]+)", report)
    in_dim = re.search(r"in_dim:\s*(\d+)", report)
    out_dim = re.search(r"out_dim:\s*(\d+)", report)
    d_model = re.search(r"d_model:\s*(\d+)", report)
    time_proj = re.search(r"time_proj:\s*(\d+)", report)
    nmode = re.search(r"nmode:\s*(\d+)", report)
    num_head = re.search(r"num_head:\s*(\d+)", report)

    results.append({
        "latent_dim": int(latent_dim.group(1)) if latent_dim else None,
        "Epoch": int(epoch.group(1)) if epoch else None,
        "beta": float(beta.group(1)) if beta else None,
        "in_dim": int(in_dim.group(1)) if in_dim else None,
        "out_dim": int(out_dim.group(1)) if out_dim else None,
        "d_model": int(d_model.group(1)) if d_model else None,
        "time_proj": int(time_proj.group(1)) if time_proj else None,
        "nmode": int(nmode.group(1)) if nmode else None,
        "num_head": int(num_head.group(1)) if num_head else None,
        "RMSE": float(rmse.group(1)) if rmse else None,
        "nRMSE": float(nrmse.group(1)) if nrmse else None,
        "CSV Error": float(csv_err.group(1)) if csv_err else None,
        "Max Error": float(max_err.group(1)) if max_err else None,
        "Boundary RMSE": float(bd_rmse.group(1)) if bd_rmse else None,
        "Fourier Error": [float(v.strip()) for v in fourier_err.group(1).split(",")] if fourier_err else [None, None, None]
    })

# Expand Fourier components
for r in results:
    f = r.pop("Fourier Error")
    r["Fourier1"], r["Fourier2"], r["Fourier3"] = f

# Create dataframe
df = pd.DataFrame(results)
df.to_csv("parsed_metrics.csv", index=False)

# Generate top-3 lowest error entries for each error type
top_k = 3
metrics = ["RMSE", "nRMSE", "CSV Error", "Max Error", "Boundary RMSE", "Fourier1", "Fourier2", "Fourier3"]
top_results = []

for metric in metrics:
    top = df.nsmallest(top_k, metric)
    top["Metric"] = metric
    top["ErrorValue"] = top[metric]
    top_results.append(top)

top_df = pd.concat(top_results)
top_df = top_df[["Metric", "ErrorValue", "latent_dim", "Epoch", "beta",
                 "in_dim", "out_dim", "d_model", "time_proj", "nmode", "num_head"] + metrics]
top_df.to_csv("top3_errors.csv", index=False)

print("Saved: parsed_metrics.csv and top3_errors.csv")
