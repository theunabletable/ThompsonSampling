


from pathlib import Path
# --- File paths ---
DF_EVENTS_PATH = "C:/Users/Drew/Desktop/JOB/2024 data/new dataframes/dfEventsThreeMonthsTimezones.csv"
DF_HOURLY_PATH       = "data/dfHourly.csv"           # master hourly table
DECISIONS_OUT_PATH   = "output/decisions_today.csv"  # daily send list

BASE_DIR            = Path(r"C:\Users\Drew\Desktop\JOB\ThompsonSampling\Data")
DECISIONS_SAVE_DIR  = BASE_DIR / "decisions"

MODELS_CURRENT_DIR  = BASE_DIR / "models" / "current"
MODELS_WEEKLY_DIR   = BASE_DIR / "models" / "weekly"

STEPS_MANAGER_PATH  = MODELS_CURRENT_DIR / "stepsManager.pkl"
SLEEP_MANAGER_PATH  = MODELS_CURRENT_DIR / "sleepManager.pkl"
MOOD_MANAGER_PATH   = MODELS_CURRENT_DIR / "moodManager.pkl"

STEPS_MANAGER_UPDATED_PATH = r"C:\Users\Drew\Desktop\JOB\ThompsonSampling\AgentSaves\StepsManagerUpdated.pkl"
SLEEP_MANAGER_UPDATED_PATH = r"C:\Users\Drew\Desktop\JOB\ThompsonSampling\AgentSaves\SleepManagerUpdated.pkl"
MOOD_MANAGER_UPDATED_PATH  = r"C:\Users\Drew\Desktop\JOB\ThompsonSampling\AgentSaves\MoodManagerUpdated.pkl"



DECISIONS_SAVE_DIR = r"C:\Users\Drew\Desktop\JOB\ThompsonSampling\DecisionSaves"
# --- General settings ---
ACTION_COL           = "sent"                       # name of the action column
INTERCEPT_COL        = "intercept"                  # name of the intercept column

# --- Steps agent feature specs ---
STEP_MAIN_COLS = [
    "StepsPastDay",
    "Steps1HourBefore",
    "is_weekend",
    "StepsPastWeek",
    "StepsPast3Days",
    "maxHRPast24Hours",
    "RHR",
    "HoursSinceLastMood",
]

STEP_INTERACTION_COLS = [
    "sent_is_weekend",
    "sent_HoursSinceLastMood",
    "sent_Steps1HourBefore",
    "sent_StepsPastDay",
    "sent_StepsPastWeek",
    "sent_RHR",
]

STEP_FEATURE_COLS = (
    STEP_MAIN_COLS
    + ["sent", "intercept"]
    + STEP_INTERACTION_COLS
)
STEPS_REWARD       = "LogStepsReward" 


# Sleep
SLEEP_MAIN_COLS = [
    "MinutesSleepLast24Hours",
    "underslept",
    "RHR",
    "maxHRPast24Hours",
    "stdDevSleepSoFar",
    "avgSleepSoFar",
    "HoursSinceLastMood",
    "is_weekend",
    "StepsPastDay",
    "exercisedPast24Hours",
    "Steps1HourBefore"
]

SLEEP_INTERACTION_COLS = [
    "sent_is_weekend",
    "sent_underslept",
    "sent_HoursSinceLastMood",
    "sent_Steps1HourBefore",
    "sent_exercisedPast24Hours",
    "sent_MinutesSleepLast24Hours",
]

SLEEP_FEATURE_COLS = (
    SLEEP_MAIN_COLS
    + ["sent", "intercept"]
    + SLEEP_INTERACTION_COLS
)
SLEEP_REWARD        = "SleepReward"                  # reward column for sleep agent
# Mood
MOOD_MAIN_COLS = [
    "LastMoodScore",
    "HoursSinceLastMood",
    "StepsPastWeek",
    "avgSleepSoFar",
    "stdDevSleepSoFar",
    "MinutesSleepLast24Hours",
    "Steps1HourBefore",
    "exercisedPast24Hours",
    "underslept",
    "is_weekend",
    "MoodLast24Hours",
]

MOOD_INTERACTION_COLS = [
    "sent_is_weekend",
    "sent_exercisedPast24Hours",
    "sent_underslept",
    "sent_Steps1HourBefore",
    "sent_MoodLast24Hours",
]

MOOD_FEATURE_COLS = (
    MOOD_MAIN_COLS
    + ["sent", "intercept"]
    + MOOD_INTERACTION_COLS
)

MOOD_REWARD         = "NextMoodScore"               # reward column for mood agent


