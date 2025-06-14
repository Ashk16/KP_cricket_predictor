# KP Cricket Predictor - Project Documentation

## 1. High-Level Project Goal

The primary objective of this project is to develop a sophisticated astrological prediction model for T20 cricket matches based on Krishnamurti Paddhati (KP) astrology. Beyond simple prediction, the project aims to create a robust, data-driven framework for systematically testing, analyzing, and refining the model's core astrological rules.

The system is designed to move beyond anecdotal evidence and provide statistically significant insights into which astrological factors are most influential during critical phases of a match.

---

## 2. Core Components

The project is composed of several key scripts, each with a distinct responsibility:

-   **`app/app.py`**: The user-facing front-end built with Streamlit. It provides an interactive interface for generating a dynamic, real-time astrological timeline for a given match based on the underlying rules engine. It is designed for "what-if" analysis and visualizing predictions for a single game.

-   **`scripts/chart_generator.py`**: A foundational utility that calculates KP astrological charts for a specific date, time, and location. It is the source for all planetary positions, house cusps, and Dasha lord calculations (Star Lord, Sub Lord, Sub-Sub Lord).

-   **`scripts/kp_favorability_rules.py`**: This is the "brain" of the prediction model. It contains the core astrological logic, including house significations, planetary weights, dignity calculations, and the final scoring algorithm that produces the `asc_score` and `desc_score`. **This file is the primary target for our improvement cycle.**

-   **`scripts/training_supervisor.py`**: A powerful batch-processing script designed to run our prediction model against thousands of historical matches. It iterates through the `match_index.csv`, processes the corresponding Cricsheet JSON data, and generates a detailed, ball-by-ball analysis file (`*_analyzed.csv`) for each match, saving it to the `training_analysis/` directory.

-   **`scripts/rule_optimizer.py`**: The "intelligent analysis" engine. This script ingests the thousands of files produced by the supervisor and performs advanced statistical analysis to identify weaknesses and patterns in the prediction model. Its key features include:
    -   **Flip Test**: Automatically determines the correct Ascendant team for each match to maximize prediction accuracy.
    -   **Correlation Filter**: Discards low-quality matches where predictions show little correlation with reality, preventing data pollution.
    -   **Weighted Impact Score**: Moves beyond binary correct/incorrect predictions by weighting wickets and boundaries more heavily to focus on game-changing events.
    -   **Contextual Influence Analysis**: Our most advanced feature. It analyzes how the influence of Sub Lords and Sub-Sub Lords changes based on the strength of the ruling Star Lord, providing data to refine the 50-30-20 weighting system dynamically.
    -   The final output is `contextual_influence_analysis.csv`, which serves as the basis for our rule refinement hypotheses.

---

## 3. The Model Improvement Cycle

This is our core methodology for improving the prediction model. It is a data-driven, iterative loop.

**Step 1: Data Generation (`training_supervisor.py`)**
> We begin by running the supervisor script over our entire dataset of historical T20 matches. This applies the *current* set of rules in `kp_favorability_rules.py` to every ball of every game, creating a rich dataset of predictions vs. reality stored in the `training_analysis/` folder.

**Step 2: Analysis & Insight (`rule_optimizer.py`)**
> Next, we run the optimizer script. It processes all the data from Step 1, applying its advanced filters and analyses (Flip Test, Impact Score, Contextual Analysis). The output is a single, clean CSV file (`contextual_influence_analysis.csv`) that provides clear, actionable insights into how our rules are performing under various astrological conditions.

**Step 3: Hypothesis & Refinement (Manual)**
> This is the human-in-the-loop step. We analyze the results from Step 2 to formulate a specific, testable hypothesis.
> - *Example Hypothesis:* "The data shows that when the Star Lord is neutral (score between -1 and 1), the Sub Lord's influence is a much better predictor of outcomes. Therefore, we should increase the Sub Lord's weight in the `asc_score` calculation under this specific condition."
> We then implement this change directly in the `kp_favorability_rules.py` file.

**Step 4: Repeat the Cycle**
> With the new rule implemented, we return to Step 1. We re-run the entire pipeline to generate new analysis data based on our modified rules. By comparing the new results to the previous baseline, we can statistically measure whether our change improved the model's predictive accuracy.

---

## 4. Current Status & Next Steps

-   **Current State**: We have fixed all known bugs in the application and the training pipeline. The `training_supervisor.py` script is currently running in the background, generating a fresh, corrected set of analysis files using our most up-to-date code.

-   **Immediate Next Step**: Once the supervisor has generated at least 50 match files, we will execute `scripts/rule_optimizer.py`. This will give us our first `contextual_influence_analysis.csv`.

-   **Following Step**: We will perform **Step 3: Hypothesis & Refinement**. We will carefully analyze the `contextual_influence_analysis.csv` to identify the most promising area for our first rule adjustment and then implement it. 