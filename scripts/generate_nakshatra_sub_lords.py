import csv

# Vimshottari Dasha sequence
dasha_sequence = [
    "Ketu", "Venus", "Sun", "Moon", "Mars",
    "Rahu", "Jupiter", "Saturn", "Mercury"
]

# 27 Nakshatras and their ruling planets
nakshatras = [
    ("Ashwini", "Ketu"),
    ("Bharani", "Venus"),
    ("Krittika", "Sun"),
    ("Rohini", "Moon"),
    ("Mrigashira", "Mars"),
    ("Ardra", "Rahu"),
    ("Punarvasu", "Jupiter"),
    ("Pushya", "Saturn"),
    ("Ashlesha", "Mercury"),
    ("Magha", "Ketu"),
    ("Purva Phalguni", "Venus"),
    ("Uttara Phalguni", "Sun"),
    ("Hasta", "Moon"),
    ("Chitra", "Mars"),
    ("Swati", "Rahu"),
    ("Vishakha", "Jupiter"),
    ("Anuradha", "Saturn"),
    ("Jyeshtha", "Mercury"),
    ("Mula", "Ketu"),
    ("Purva Ashadha", "Venus"),
    ("Uttara Ashadha", "Sun"),
    ("Shravana", "Moon"),
    ("Dhanishta", "Mars"),
    ("Shatabhisha", "Rahu"),
    ("Purva Bhadrapada", "Jupiter"),
    ("Uttara Bhadrapada", "Saturn"),
    ("Revati", "Mercury")
]

# Write to CSV
with open("config/nakshatra_sub_lords.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Nakshatra", "Pada", "Planet", "Sub_Lord", "Sub_Sub_Lord"])

    dasha_index = 0  # Start from Ketu

    for nak_name, ruling_planet in nakshatras:
        for pada in range(1, 5):
            sub_lord = dasha_sequence[dasha_index % len(dasha_sequence)]
            sub_sub_lord = dasha_sequence[(dasha_index + 1) % len(dasha_sequence)]
            writer.writerow([nak_name, pada, ruling_planet, sub_lord, sub_sub_lord])
            dasha_index += 1
