import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/spurs_matches_feature_set_d.csv")
df["date"] = pd.to_datetime(df["date"])

print("Matches ordered by days since last match (longest rest first):")
df_sorted = df.sort_values("days_since_last_match", ascending=False).reset_index(drop=True)
print(df_sorted[["date", "opponent_team", "venue", "days_since_last_match"]].to_string(index=False))

# a normal top-flight rest gap is ~3-8 days (midweek/weekend cycle); a typical
# FIFA international break adds up to ~14 days, so mark that as the reference
# line for "this gap is unusual" rather than just routine fixture spacing.
# Not all gaps past this line are international breaks (season boundaries,
# COVID, the Qatar World Cup break also show up here), hence the general name.
EXTENDED_BREAK_THRESHOLD_DAYS = 14

plt.figure(figsize=(14, 5))
plt.plot(df["date"], df["days_since_last_match"], marker="o", markersize=3, linewidth=0.8)
plt.axhline(EXTENDED_BREAK_THRESHOLD_DAYS, color="red", linestyle="--", linewidth=1, label=f"{EXTENDED_BREAK_THRESHOLD_DAYS}-day extended break threshold")
plt.title("Days since last match, over time")
plt.xlabel("Date")
plt.ylabel("Days since last match")
plt.legend()
plt.tight_layout()

plot_path = "plots/days_since_last_match.png"
plt.savefig(plot_path)
plt.close()
print(f"\nSaved plot to {plot_path}")
