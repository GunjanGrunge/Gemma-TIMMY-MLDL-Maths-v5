# CGI Portable Martha Client

This folder is a small portable client for the `V6` Martha hybrid runtime.

It is meant for teammates who want to:

- download the published runtime from GitHub or Hugging Face
- ask ML / DL / stats / forecasting / DTree style questions
- use the same deterministic-hybrid logic without setting up the whole training workspace

## Important

This client uses the **V6 hybrid path**, not the failed standalone raw adapter path.

That means:

- deterministic calculators do the exact arithmetic
- the hybrid router decides which calculator / solver to use
- this is the recommended production-style interface for Martha

## Install

```powershell
python -m pip install -r CGI/requirements.txt
```

## Download Runtime

From GitHub:

```powershell
python CGI/download_runtime.py --source github --dest CGI/runtime
```

From Hugging Face:

```powershell
python CGI/download_runtime.py --source hf --dest CGI/runtime
```

If you want a specific source:

```powershell
python CGI/download_runtime.py --source github --github-repo https://github.com/GunjanGrunge/Gemma-TIMMY-MLDL-Maths-v5 --ref main --dest CGI/runtime
python CGI/download_runtime.py --source hf --hf-repo Gunjan/Gemma-TIMMY-MLDL-Maths-v5 --dest CGI/runtime
```

## Run Martha

Decision-tree split:

```powershell
python CGI/ml_consultant.py --runtime-path CGI/runtime --question "Decision tree split case: parent class counts=[9, 5], child counts=[[6, 1], [3, 4]]. Compute information gain and gini."
```

Forecast diagnostics:

```powershell
python CGI/ml_consultant.py --runtime-path CGI/runtime --question "Forecast diagnostics: actuals=[100, 110, 105], forecasts=[98, 112, 108], train_history=[90, 95, 100, 104]. Compute sMAPE and MASE."
```

Hyperparameter-style consultant question:

```powershell
python CGI/ml_consultant.py --runtime-path CGI/runtime --question "For a binary classifier with unstable validation loss and overfitting after epoch 6, explain what to inspect first and what hyperparameter changes are safest."
```

Structured JSON output:

```powershell
python CGI/ml_consultant.py --runtime-path CGI/runtime --question "Portfolio variance is given as -0.02 with weights [0.5, 0.5]. Compute volatility." --json
```

## Notes

- The downloaded runtime must contain `einstein_v6_hybrid_assistant.py`.
- The current public V6 hybrid runtime is specialized. It is strong on ML / DL / stats / forecasting / guardrails, but it is not a general contest-math solver.
- If a question is outside the supported V6 routes, the runtime falls back to the older hybrid path shipped in the downloaded repo.
