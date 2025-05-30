import csv
from datetime import datetime
from scripts.kp_favorability_rules import (
    create_chart,
    get_significator_houses,
    get_conjunctions,
    get_ruling_planets,
    evaluate_period
)

# Match details
match_date = "2025-05-02"
location = {"lat": 23.0225, "lon": 72.5714}  # Ahmedabad
ascendant_sign = "Libra"

# Path to the sub lord timeline CSV
timeline_csv = "data/processed/muhurta_timeline.csv"
output_csv = "results/favorability_timeline.csv"

# Read timeline and compute favorability
with open(timeline_csv, "r") as infile, open(output_csv, "w", newline="") as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ["Favorability"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        dt = datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M")
        chart = create_chart(dt, location["lat"], location["lon"])

        sub_lord = row["Sub_Lord"]
        sub_sub_lord = row["Sub_Sub_Lord"]
        period_data = {
            "Sub_Lord": sub_lord,
            "Sub_Sub_Lord": sub_sub_lord,
            "Sub_Lord_Houses": get_significator_houses(sub_lord, chart),
            "Sub_Sub_Lord_Houses": get_significator_houses(sub_sub_lord, chart),
            "Sub_Lord_Retrograde": row["Sub_Lord_Retrograde"] == "True",
            "Sub_Sub_Lord_Retrograde": row["Sub_Sub_Lord_Retrograde"] == "True",
            "Sub_Lord_Combust": row["Sub_Lord_Combust"] == "True",
            "Sub_Sub_Lord_Combust": row["Sub_Sub_Lord_Combust"] == "True",
            "Sub_Lord_Conjunct": get_conjunctions(sub_lord, chart),
            "Sub_Sub_Lord_Conjunct": get_conjunctions(sub_sub_lord, chart),
            "Sub_Lord_Star_Lord": row["Sub_Lord_Star_Lord"],
            "Sub_Sub_Lord_Star_Lord": row["Sub_Sub_Lord_Star_Lord"],
            "Ruling_Planets": get_ruling_planets(dt, location["lat"], location["lon"])
        }

        favorability = evaluate_period(period_data, ascendant_sign)
        row["Favorability"] = favorability
        writer.writerow(row)

print(f"✅ Favorability timeline generated: {output_csv}")
