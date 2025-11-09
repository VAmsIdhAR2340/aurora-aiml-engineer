"""Data analysis script to find anomalies in Aurora member data"""
import requests
import pandas as pd
import sys
from datetime import datetime

def log(message):
    """Print with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

# Start
log("=" * 70)
log("Starting Aurora Data Analysis")
log("=" * 70)

# Fetch ALL data using skip/limit parameters
url = "https://november7-730026606190.europe-west1.run.app/messages/"

log("Fetching ALL messages from Aurora API...")

try:
    response = requests.get(
        url,
        params={'skip': 0, 'limit': 10000},  # Get all messages at once
        timeout=30
    )
    response.raise_for_status()
    
    data = response.json()
    
    if isinstance(data, dict):
        all_messages = data.get('items', [])
        total = data.get('total', len(all_messages))
        log(f"SUCCESS: Fetched {len(all_messages)} messages (total: {total})")
    elif isinstance(data, list):
        all_messages = data
        log(f"SUCCESS: Fetched {len(all_messages)} messages")
    else:
        log(f"ERROR: Unexpected response format: {type(data)}")
        sys.exit(1)
        
except Exception as e:
    log(f"ERROR fetching data: {e}")
    sys.exit(1)

if not all_messages:
    log("ERROR: No messages fetched!")
    sys.exit(1)

log("")

# Convert to DataFrame
log("Converting to DataFrame...")
df = pd.DataFrame(all_messages)
log(f"SUCCESS: DataFrame created with {len(df)} rows, {len(df.columns)} columns")
log(f"Columns: {', '.join(df.columns)}")
log("")

# === ANALYSIS 1: Duplicate messages ===
log("ANALYSIS 1: Checking for duplicate messages...")
duplicates = df[df.duplicated(subset=['message'], keep=False)]
unique_duplicates = df[df.duplicated(subset=['message'], keep='first')]
log(f"  - Total duplicate rows: {len(duplicates)}")
log(f"  - Unique messages appearing multiple times: {len(unique_duplicates)}")
log(f"  - Percentage: {len(duplicates)/len(df)*100:.2f}%")

if len(unique_duplicates) > 0 and len(unique_duplicates) < 10:
    log(f"  - Examples:")
    for idx, row in unique_duplicates.head(5).iterrows():
        log(f"      * {row['user_name']}: '{row['message'][:60]}...'")
log("")

# === ANALYSIS 2: Timestamp anomalies ===
log("ANALYSIS 2: Checking timestamp anomalies...")
df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
now = pd.Timestamp.now()
future_messages = df[df['timestamp'] > now]
log(f"  - Messages from the future: {len(future_messages)}")

old_messages = df[df['timestamp'] < pd.Timestamp('2020-01-01')]
log(f"  - Very old messages (before 2020): {len(old_messages)}")

date_range = f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
log(f"  - Date range: {date_range}")

# Check for same timestamps
same_time = df.groupby('timestamp').size()
max_same_time = same_time.max()
log(f"  - Max messages with same timestamp: {max_same_time}")
log("")

# === ANALYSIS 3: Missing data ===
log("ANALYSIS 3: Checking for missing data...")
for col in df.columns:
    missing = df[col].isna().sum()
    if missing > 0:
        log(f"  - Missing {col}: {missing}")
if df.isna().sum().sum() == 0:
    log("  - SUCCESS: No missing data found!")
log("")

# === ANALYSIS 4: User ID consistency ===
log("ANALYSIS 4: Checking user ID consistency...")
user_mapping = df.groupby('user_id')['user_name'].nunique()
inconsistent = user_mapping[user_mapping > 1]
log(f"  - User IDs with multiple names: {len(inconsistent)}")
if len(inconsistent) > 0:
    log("  - WARNING: Same user_id maps to different names!")
    for uid in list(inconsistent.index)[:3]:
        names = df[df['user_id'] == uid]['user_name'].unique()
        log(f"      * {uid[:16]}...: {', '.join(names)}")
log("")

# === ANALYSIS 5: Name encoding issues ===
log("ANALYSIS 5: Checking name encoding issues...")
weird_chars = df[df['user_name'].str.contains(r'[^\x00-\x7F]', na=False)]
log(f"  - Names with non-ASCII characters: {len(weird_chars)}")
if len(weird_chars) > 0:
    unique_weird = weird_chars['user_name'].unique()
    log(f"  - Examples: {', '.join(unique_weird[:5])}")
log("")

# === ANALYSIS 6: Message lengths ===
log("ANALYSIS 6: Analyzing message lengths...")
df['msg_length'] = df['message'].str.len()
log(f"  - Shortest: {df['msg_length'].min()} chars")
log(f"  - Longest: {df['msg_length'].max()} chars")
log(f"  - Average: {df['msg_length'].mean():.1f} chars")
log(f"  - Median: {df['msg_length'].median():.0f} chars")

# Show extremes
if df['msg_length'].min() < 20:
    shortest = df.loc[df['msg_length'].idxmin()]
    log(f"  - Shortest message: '{shortest['message']}'")

longest = df.loc[df['msg_length'].idxmax()]
log(f"  - Longest (first 80 chars): '{longest['message'][:80]}...'")
log("")

# === ANALYSIS 7: Member activity ===
log("ANALYSIS 7: Analyzing member activity...")
log(f"  - Total unique members: {df['user_name'].nunique()}")
log(f"  - Total messages: {len(df)}")
log(f"  - Average messages per member: {len(df)/df['user_name'].nunique():.1f}")

top_users = df['user_name'].value_counts().head(10)
log(f"  - Top 10 most active members:")
for idx, (name, count) in enumerate(top_users.items(), 1):
    percent = count/len(df)*100
    log(f"      {idx}. {name}: {count} messages ({percent:.1f}%)")
log("")

# === ANALYSIS 8: PII detection ===
log("ANALYSIS 8: Checking for PII exposure...")
phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
card_pattern = r'\b\d{4}[\s-]*\d{4}[\s-]*\d{4}[\s-]*\d{4}\b'

has_phone = df['message'].str.contains(phone_pattern, na=False, regex=True)
has_email = df['message'].str.contains(email_pattern, na=False, regex=True)
has_card = df['message'].str.contains(card_pattern, na=False, regex=True)

log(f"  - Messages with phone numbers: {has_phone.sum()}")
log(f"  - Messages with emails: {has_email.sum()}")
log(f"  - Messages with potential card numbers: {has_card.sum()}")

if (has_phone.sum() + has_email.sum() + has_card.sum()) > 0:
    log("  - WARNING: PII detected in plain text messages!")
log("")

# === Save results ===
output_file = 'member_data_analysis.csv'
df.to_csv(output_file, index=False)
log(f"Saved analysis to: {output_file}")
log("")

# === Summary ===
log("=" * 70)
log("ANALYSIS COMPLETE")
log("=" * 70)
log("KEY FINDINGS:")
log(f"  * Total Messages: {len(df)}")
log(f"  * Unique Members: {df['user_name'].nunique()}")
log(f"  * Date Range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
log(f"  * Duplicate Messages: {len(unique_duplicates)} ({len(unique_duplicates)/len(df)*100:.2f}%)")
log(f"  * Missing Data: {'None' if df.isna().sum().sum() == 0 else 'Found'}")
log(f"  * PII Instances: {has_phone.sum() + has_email.sum() + has_card.sum()}")
log("")
log("Ready to document findings in README.md!")
log("=" * 70)
