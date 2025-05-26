#  • For one calendar day (`run_date`) read every *raw* CSV that arrived
#    under …/Daily/YYYY-MM-DD/.
#  • Convert the four streams (steps / HR / HRV / sleep) for each device
#    into **tidy local-time hourly tables**.
#  • Outer-join the four streams within each device,
#    then simply *stack* the three device blocks together.
#  • Write the resulting slice to
#        …/Daily/YYYY-MM-DD/dfHourly_YYYY-MM-DD.csv
#
#  NOTE:  This script **does not** touch the long-term master `dfHourly.csv`.
#         A separate weekly / monthly compaction job can append or rebuild
#         that file from the per-day slices if desired.
# ---------------------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
from datetime import date as _date
import sys
import datetime as dt
import pandas as pd

from config         import RAW_DAILY
from HourlyBuilders import build_device_hourly      # ← public helper we wrote
# ---------------------------------------------------------------------------

# ── 1.  Which day are we processing?  ──────────────────────────────────────
if len(sys.argv) == 2:
    run_date = pd.to_datetime(sys.argv[1]).date()   # “2025-05-20” on CLI
else:
    run_date = dt.date.today()                      # default = today

day_str = run_date.isoformat()                      # "YYYY-MM-DD"
day_dir = RAW_DAILY / day_str                       # …/Daily/YYYY-MM-DD/
day_dir.mkdir(parents=True, exist_ok=True)

# ── 2.  Build per-device hourly tables  ───────────────────────────────────
apple_df  = build_device_hourly(run_date, "healthkit")
garmin_df = build_device_hourly(run_date, "garmin")
fitbit_df = build_device_hourly(run_date, "fitbit")

device_blocks = [df for df in (apple_df, garmin_df, fitbit_df) if not df.empty]

# ── 3.  Stack devices  +  save slice  ─────────────────────────────────────
df_day = (
    pd.concat(device_blocks, ignore_index=True)
      .sort_values(["PARTICIPANTIDENTIFIER", "time", "Device"])
      .reset_index(drop=True)
)

out_path = day_dir / f"dfHourly_{day_str}.csv"
df_day.to_csv(out_path, index=False)
print(f"[DailyHourlyBuilder]  wrote  {len(df_day):,} rows  →  {out_path}")