# KP Cricket Predictor 🏏🔮

This project uses KP Astrology (Krishnamurti Paddhati) to predict cricket match momentum by analyzing Muhurat charts.

## 🔧 Features

- Calculate Ascendant & Moon longitude using Swiss Ephemeris
- Determine Moon's Nakshatra, Sub Lord, and Sub-Sub Lord
- Generate Sub Lord mapping (108 Padas) from KP Dasha sequence
- Designed to evaluate match periods favoring Ascendant or Descendant

## 📁 Project Structure

KP_cricket_predictor/
├── scripts/
│ └── chart_generator.py
│ └── generate_nakshatra_sub_lords.py
├── config/
│ └── nakshatra_sub_lords.csv
├── README.md

markdown
Copy
Edit

## 🔮 Planned

- KP logic to determine who is favored: Ascendant or Descendant
- Fantasy team picker based on Muhurat periods
- Match schedule automation via date/location input

## 🧠 Tech Stack

- Python
- Swiss Ephemeris (`pyswisseph`)
- Pandas
- VS Code + CodeGPT

## ✨ Example

To run chart generation:

```bash
python scripts/chart_generator.py
🚀 How to Contribute
Star 🌟 the repo

Clone & explore

Open an issue or discussion

📜 License
MIT License

yaml
Copy
Edit

---

Just paste that into a new file called `README.md` at the root of your repo.

Then push it:
```bash
git add README.md
git commit -m "Add README with full project overview"
git push