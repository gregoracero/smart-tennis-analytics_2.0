import re
import pandas as pd

INPUT = "data/parquet/analytics_matches.parquet"
OUTPUT = "data/parquet/analytics_matches_tiebreaks.parquet"

df = pd.read_parquet(INPUT)

winner_tb_played = []
winner_tb_won = []

loser_tb_played = []
loser_tb_won = []

for score in df["score"].fillna(""):

    played = len(
        re.findall(r'7-6\(|6-7\(', score)
    )

    won_by_winner = len(
        re.findall(r'7-6\(', score)
    )

    won_by_loser = len(
        re.findall(r'6-7\(', score)
    )

    winner_tb_played.append(played)
    winner_tb_won.append(won_by_winner)

    loser_tb_played.append(played)
    loser_tb_won.append(won_by_loser)

df["winner_tb_played"] = winner_tb_played
df["winner_tb_won"] = winner_tb_won

df["loser_tb_played"] = loser_tb_played
df["loser_tb_won"] = loser_tb_won

df.to_parquet(
    OUTPUT,
    index=False
)

print(df.shape)
