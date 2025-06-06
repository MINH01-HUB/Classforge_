import pandas as pd
import random
from itertools import combinations

def simulate_relationships(df, friend_threshold=0.8, conflict_threshold=0.9):
    """
    Generate synthetic Friend/Conflict/Neutral edges based on similarity and distress.
    Returns a DataFrame with columns: source, target, type.
    """
    edges = []

    for i, j in combinations(df.index, 2):
        s1 = df.loc[i]
        s2 = df.loc[j]

        # Similarity score for friendship (closer SES + achievement)
        sim_score = 1 - abs(s1['achievement'] - s2['achievement']) / 100 \
                      - abs(s1['SES'] - s2['SES']) / 10

        # Distress conflict flag
        distress_flag = (s1['psychological_distress'] + s2['psychological_distress']) == 2
        ses_gap = abs(s1['SES'] - s2['SES'])

        if sim_score > friend_threshold:
            edges.append({"source": s1['student_id'], "target": s2['student_id'], "type": "Friend"})
        elif distress_flag and ses_gap >= 3:
            edges.append({"source": s1['student_id'], "target": s2['student_id'], "type": "Conflict"})
        elif random.random() < 0.02:  # occasional neutral edges
            edges.append({"source": s1['student_id'], "target": s2['student_id'], "type": "Neutral"})

    return pd.DataFrame(edges)
