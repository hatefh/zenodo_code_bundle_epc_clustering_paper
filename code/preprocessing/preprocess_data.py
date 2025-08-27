import pandas as pd
from pathlib import Path
import yaml
from code.utils.paths import pth

def load_cfg():
    with open(pth("config", "config.yaml"), "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_cfg()
    data_dir = pth(cfg["data_dir"])

    # EPC metadata (already CSV after export from Numbers)
    epc = pd.read_csv(data_dir / cfg["inputs"]["epc"])

    # District heating
    dh = pd.read_csv(data_dir / cfg["inputs"]["dh"], parse_dates=["timestamp"])

    # Outdoor temperature (raw has 'Timestamp' and 'Air temperature, deg C')
    temp_raw = pd.read_csv(data_dir / cfg["inputs"]["weather"])
    if "Timestamp" in temp_raw.columns and "Air temperature, deg C" in temp_raw.columns:
        temp = temp_raw.rename(columns={"Air temperature, deg C": "outdoor_temp_C"}).copy()
        temp["timestamp"] = pd.to_datetime(temp["Timestamp"], format="%d.%m.%Y %H:%M")
        temp = temp[["timestamp", "outdoor_temp_C"]]
    else:
        # already cleaned
        temp = temp_raw.copy()
        if "timestamp" in temp.columns:
            temp["timestamp"] = pd.to_datetime(temp["timestamp"])

    # Save standardized outputs
    out_dh = pth(cfg["outputs"]["dh_clean"]); Path(out_dh).parent.mkdir(parents=True, exist_ok=True)
    out_temp = pth(cfg["outputs"]["temp_clean"])
    out_epc = pth(cfg["outputs"]["epc_clean"])

    dh.to_parquet(out_dh, index=False)
    temp.to_parquet(out_temp, index=False)
    epc.to_parquet(out_epc, index=False)

if __name__ == "__main__":
    main()
