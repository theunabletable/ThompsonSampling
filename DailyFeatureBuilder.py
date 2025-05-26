from pathlib import Path; import pandas as pd, datetime as dt, sys
from config import RAW_DAILY, HOURLY_CSV
from feature_blocks.steps  import add_step_features
from feature_blocks.sleep  import add_sleep_features   # same style
from feature_blocks.mood   import add_mood_features
from feature_blocks.hr     import add_hr_features
from feature_blocks.util   import load_daily_rhr, load_survey_master, ...
from HourlyBuilders        import combine_hourly_data   # already exists

# 1) decide run_date
run_date = dt.date.today()
#run_date = date(yyyy, mm, dd)

# 2) build current hourly  (same helpers we drafted earlier) ...
df_hourly = build_current_hourly(run_date)

# 3) skeleton
pids = df_hourly["PARTICIPANTIDENTIFIER"].unique()
df_ev = make_df_skeleton(run_date, pids)

# 4) feature blocks
df_ev = add_step_features (df_ev, df_hourly)
df_ev = add_sleep_features(df_ev, df_hourly)
df_ev = add_mood_features (df_ev, df_hourly)
df_ev = add_hr_features   (df_ev, df_hourly)
# plus any reward or survey merges
df_ev = compute_steps_reward(df_ev)
df_ev = compute_sleep_reward(df_ev)
df_ev = pd.merge(df_ev, load_survey_master(), on="PARTICIPANTIDENTIFIER", how="left")

# 5) save
day_dir = RAW_DAILY / run_date.isoformat()
day_dir.mkdir(parents=True, exist_ok=True)
out = day_dir / f"dfEvents_{run_date.isoformat()}.csv"
df_ev.to_csv(out, index=False)
print("✓ wrote", out)