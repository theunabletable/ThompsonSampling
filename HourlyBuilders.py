# HourlyBuilders.py  ---------------------------------------------------
"""
All ‘raw CSV → tidy hourly’ transforms live here.

Public API
----------
load_raw_stream(run_date, device, stream)  -> hourly DataFrame
build_device_hourly(run_date, device)      -> hourly DataFrame for ONE device
"""
from __future__ import annotations

# ---- stdlib ----------------------------------------------------------
from pathlib import Path
from datetime import date as _date
import logging

# ---- third-party -----------------------------------------------------
import pandas as pd

# ---- project-local helpers ------------------------------------------
import config                             # RAW_DAILY / raw_path / DEVICE_STREAM_MAP
from DataframeHelpersNoTimezones import * # all low-level aggregation utilities
from utils import combine_hourly_data     # outer-join helper

# ---------------------------------------------------------------------
#  logging
# ---------------------------------------------------------------------
log = logging.getLogger("HourlyBuilders")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s  %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)

# =====================================================================
#  PER-STREAM BUILDERS
# =====================================================================
# ▶▶  APPLE / HealthKit  ◀◀ -------------------------------------------
def build_healthkit_hourly_steps(raw: pd.DataFrame) -> pd.DataFrame:
    df = buildMissingHoursHealthkit(raw)
    df = merge_on_record_date(df).rename(
        columns={"RECORD_DATE": "time", "VALUE": "steps"}
    )
    df["missing_steps"] = df["steps"].isna()
    return _finalise(df, "healthkit", "healthkit-steps")

def build_healthkit_hourly_heartrate(raw: pd.DataFrame) -> pd.DataFrame:
    df = build_healthkit_hourly_hr(raw)
    df = fill_healthkit_hr(df).rename(columns={"avg_heartrate": "hr_avg"})
    df["missing_hr"] = df["hr_avg"].isna()
    return _finalise(df, "healthkit", "healthkit-heartrate")

def build_healthkit_hourly_hrv(raw: pd.DataFrame) -> pd.DataFrame:
    df = aggregate_healthkit_hrv_hourly(raw)
    df = fill_healthkit_hrv(df).rename(columns={"avg_hrv": "hrv_avg"})
    df["missing_hrv"] = df["hrv_avg"].isna()
    return _finalise(df, "healthkit", "healthkit-hrv")

def build_healthkit_hourly_sleep(raw: pd.DataFrame) -> pd.DataFrame:
    merged = merge_healthkit_intervals(raw)
    df = build_healthkit_hourly_sleep_local(merged)   # LOCAL-time helper
    df = fill_healthkit_sleep(df).rename(columns={"minutes": "minutes_sleep"})
    df["missing_sleep"] = df["minutes_sleep"].isna()
    return _finalise(df, "healthkit", "healthkit-sleep")

# ▶▶  FITBIT  ◀◀ -------------------------------------------------------
def build_fitbit_hourly_steps(raw: pd.DataFrame) -> pd.DataFrame:
    df = aggregate_fitbit_steps(raw)
    df = fill_fitbit_hours(df).rename(columns={"STEPS": "steps"})
    df["missing_steps"] = df["steps"].isna()
    return _finalise(df, "fitbit", "fitbit-steps")

def build_fitbit_hourly_heartrate(raw: pd.DataFrame) -> pd.DataFrame:
    df = aggregate_fitbit_hr_to_hourly(raw)
    df = fill_fitbit_hr_hours(df).rename(columns={"AVG_HEARTRATE": "hr_avg"})
    df["missing_hr"] = df["hr_avg"].isna()
    return _finalise(df, "fitbit", "fitbit-heartrate")

def build_fitbit_hourly_hrv(raw: pd.DataFrame) -> pd.DataFrame:
    df = aggregate_fitbit_hrv_to_hourly(raw)
    df = fill_fitbit_hrv_hours(df).rename(columns={"HRV_avg": "hrv_avg"})
    df["missing_hrv"] = df["hrv_avg"].isna()
    return _finalise(df, "fitbit", "fitbit-hrv")

def build_fitbit_hourly_sleep(raw: pd.DataFrame) -> pd.DataFrame:
    df = build_fitbit_sleep_hourly(raw)
    df = add_adjusted_sleep(df)
    df = fill_fitbit_sleep(df).rename(columns={"minutes_asleep": "minutes_sleep"})
    df["missing_sleep"] = df["minutes_sleep"].isna()
    return _finalise(df, "fitbit", "fitbit-sleep")

# ▶▶  GARMIN  ◀◀ -------------------------------------------------------
def build_garmin_hourly_steps(raw: pd.DataFrame) -> pd.DataFrame:
    df = aggregate_garmin_steps_hourly(raw)
    df = fill_garmin_steps(df).rename(columns={"STEPS": "steps"})
    df["missing_steps"] = df["steps"].isna()
    return _finalise(df, "garmin", "garmin-steps")

def build_garmin_hourly_heartrate(raw: pd.DataFrame) -> pd.DataFrame:
    df = aggregate_garmin_hr_hourly(raw)
    df = fill_garmin_hr(df).rename(columns={"AVG_HEARTRATE": "hr_avg"})
    df["missing_hr"] = df["hr_avg"].isna()
    return _finalise(df, "garmin", "garmin-heartrate")

def build_garmin_hourly_hrv(raw: pd.DataFrame) -> pd.DataFrame:
    df = aggregate_garmin_hrv_hourly(raw)
    df = fill_garmin_hrv_hours(df).rename(columns={"HRV_avg": "hrv_avg"})
    df["missing_hrv"] = df["hrv_avg"].isna()
    return _finalise(df, "garmin", "garmin-hrv")

def build_garmin_hourly_sleep(raw: pd.DataFrame) -> pd.DataFrame:
    df = build_garmin_sleep_hourly(raw)
    df = fill_garmin_sleep(df).rename(columns={"minutes_asleep": "minutes_sleep"})
    df["missing_sleep"] = df["minutes_sleep"].isna()
    return _finalise(df, "garmin", "garmin-sleep")

# =====================================================================
#  INTERNAL UTILITIES  (after builders so they’re in scope)
# =====================================================================
def _finalise(df: pd.DataFrame, device: str, tag: str) -> pd.DataFrame:
    """Add Device column, sort, log.""" 
    df = df.copy()
    df["Device"] = device.capitalize()
    df = (
        df.sort_values(["PARTICIPANTIDENTIFIER", "time"])
          .reset_index(drop=True)
    )
    log.info(
        f"{tag:<18}: "
        f"{df['PARTICIPANTIDENTIFIER'].nunique():4d} users, "
        f"{len(df):7d} rows"
    )
    return df

# ---------------------------------------------------------------------
#  Builder dispatch table (now that functions exist)
# ---------------------------------------------------------------------
_BUILDER = {
    # HealthKit
    ("healthkit", "steps")     : build_healthkit_hourly_steps,
    ("healthkit", "heartrate") : build_healthkit_hourly_heartrate,
    ("healthkit", "hrv")       : build_healthkit_hourly_hrv,
    ("healthkit", "sleep")     : build_healthkit_hourly_sleep,
    # Fitbit
    ("fitbit",    "steps")     : build_fitbit_hourly_steps,
    ("fitbit",    "heartrate") : build_fitbit_hourly_heartrate,
    ("fitbit",    "hrv")       : build_fitbit_hourly_hrv,
    ("fitbit",    "sleep")     : build_fitbit_hourly_sleep,
    # Garmin
    ("garmin",    "steps")     : build_garmin_hourly_steps,
    ("garmin",    "heartrate") : build_garmin_hourly_heartrate,
    ("garmin",    "hrv")       : build_garmin_hourly_hrv,
    ("garmin",    "sleep")     : build_garmin_hourly_sleep,
}

# =====================================================================
#  PUBLIC HELPERS
# =====================================================================
def load_raw_stream(run_date: str | _date,
                    device:   str,
                    stream:   str) -> pd.DataFrame:
    """
    Read one raw CSV (Daily/YYYY-MM-DD/device_stream.csv) and return
    the hourly DataFrame produced by the matching builder.
    """
    device, stream = device.lower(), stream.lower()
    builder = _BUILDER[(device, stream)]     # KeyError ⇒ clear traceback
    csv_path = config.raw_path(run_date, device, stream)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    raw_df  = pd.read_csv(csv_path)
    return builder(raw_df)


def build_device_hourly(run_date: str | _date,
                        device:   str,
                        streams:  list[str] | None = None) -> pd.DataFrame:
    """
    • Loads *streams* (“steps/hr/hrv/sleep” by default) for one device
    • Merges them with combine_hourly_data → tidy hourly table
    • If nothing exists for that day/device → empty DataFrame
    """
    device  = device.lower()
    streams = streams or config.DEVICE_STREAM_MAP[device]

    parts = {}
    for s in streams:
        try:
            parts[s] = load_raw_stream(run_date, device, s)
        except FileNotFoundError:
            continue

    if not parts:
        log.info(f"{device:<9}: no streams found  (skipping)")
        return pd.DataFrame()

    return combine_hourly_data(
        sleep_df = parts.get("sleep",     pd.DataFrame()),
        steps_df = parts.get("steps",     pd.DataFrame()),
        hr_df    = parts.get("heartrate", pd.DataFrame()),
        hrv_df   = parts.get("hrv",       pd.DataFrame()),
        device   = device.capitalize(),
    )