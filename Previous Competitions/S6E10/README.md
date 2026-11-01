# Predicting Airline Satisfaction

## Playground Series - Season 6, Episode 10

![Header](https://www.kaggle.com/competitions/125224/images/header)

## Overview

Welcome to the 2026 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

**Your Goal:** Predicting Airline Satisfaction

## Evaluation

Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

### Submission File

For each id in the test set, you must predict a probability for the `satisfaction` variable. The file should contain a header and have the following format:

```csv
id,satisfaction
699635,0.2
699636,0.3
699637,0.1
```

## Timeline

- **Start Date** - October 1, 2026
- **Entry Deadline** - Same as the Final Submission Deadline
- **Team Merger Deadline** - Same as the Final Submission Deadline
- **Final Submission Deadline** - October 31, 2026

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## About the Tabular Playground Series

The goal of the Tabular Playground Series is to provide the Kaggle community with a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science. The duration of each competition will generally only last a few weeks, and may have longer or shorter durations depending on the challenge. The challenges will generally use fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc.

### Synthetically-Generated Datasets

Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series six years ago, and that goal is to produce datasets that have far fewer artifacts. Please feel free to give us feedback on the datasets for the different competitions so that we can continue to improve!

## Dataset Description

The dataset for this competition (both train and test) was inspired by the Airline satisfaction dataset. Feature distributions are close to, but not exactly the same, as the original.

## Files

- `train.csv` - the training set, with `satisfaction` as target
- `test.csv` - the test set, used to predict the probability for `satisfaction`
- `sample_submission.csv` - a sample submission file in the correct format
- `data.csv` - the original real-world inspired data used to generate the synthetic competition data

---

## Competition Dataset

### Overview

The competition dataset (train.csv and test.csv) is a synthetically generated dataset based on airline passenger satisfaction survey responses. It contains passenger demographics, travel details, service quality ratings, and delay information associated with overall satisfaction.

### 🔍 Use Cases

- Predicting passenger satisfaction (binary classification)
- Exploratory data analysis on service quality ratings
- Feature engineering with aggregated service scores
- Modeling the impact of flight delays on satisfaction
- Building tabular ML and explainability workflows

### Features

| Column                            | Type    | Description                                                                           |
| :-------------------------------- | :------ | :------------------------------------------------------------------------------------ |
| id                                | Integer | Unique identifier                                                                     |
| Age                               | Integer | Passenger age in years                                                                |
| Flight Distance                   | Integer | Flight distance in miles                                                              |
| Inflight wifi service             | Integer | Satisfaction with WiFi on board (1-5, 0 = not rated)                                  |
| Departure/Arrival time convenient | Integer | Satisfaction with departure/arrival time (1-5, 0 = not rated)                         |
| Ease of Online booking            | Integer | Satisfaction with online booking process (1-5, 0 = not rated)                         |
| Gate location                     | Integer | Satisfaction with gate location (1-5, 0 = not rated)                                  |
| Food and drink                    | Integer | Satisfaction with food and beverages (1-5, 0 = not rated)                             |
| Online boarding                   | Integer | Satisfaction with online check-in (1-5, 0 = not rated)                                |
| Seat comfort                      | Integer | Satisfaction with seat comfort (1-5, 0 = not rated)                                   |
| Inflight entertainment            | Integer | Satisfaction with entertainment options (1-5, 0 = not rated)                          |
| On-board service                  | Integer | Satisfaction with onboard service (1-5, 0 = not rated)                                |
| Leg room service                  | Integer | Satisfaction with legroom (1-5, 0 = not rated)                                        |
| Baggage handling                  | Integer | Satisfaction with baggage handling (1-5, 0 = not rated)                               |
| Checkin service                   | Integer | Satisfaction with check-in service (1-5, 0 = not rated)                               |
| Cleanliness                       | Integer | Satisfaction with cabin cleanliness (1-5, 0 = not rated)                              |
| Departure Delay in Minutes        | Float   | Departure delay in minutes                                                            |
| Arrival Delay in Minutes          | Float   | Arrival delay in minutes                                                              |
| Gender                            | String  | Passenger gender (Female/Male)                                                        |
| Customer Type                     | String  | Type of passenger (Loyal Customer/disloyal Customer)                                  |
| Type of Travel                    | String  | Purpose of travel (Personal Travel/Business Travel)                                   |
| Class                             | String  | Flight class (Business/Eco/Eco Plus)                                                  |
| satisfaction                      | Boolean | **Target variable**: Passenger satisfaction (True = Satisfied, False = Not Satisfied) |

---

## Original Dataset: Airline Satisfaction Dataset

### About Dataset

#### Dataset Overview

This dataset contains 129,880 survey responses collected by an airline from passengers after their flights. The survey was designed to evaluate various aspects of the passenger experience — from booking and check-in to in-flight services and baggage handling.

The data is anonymized and contains no personally identifiable information (PII). This makes it safe and easy to share for educational and research purposes.

#### Potential Use Cases

- Binary Classification — predict whether a passenger is satisfied
- Regression — predict any numeric column (service score, flight distance, etc.)
- EDA Practice — explore distributions, correlations, and delays
- Feature Engineering — create a "Total Service Score" by averaging all service ratings
- Model Comparison — test Logistic Regression, Random Forest, XGBoost, Neural Networks

### Passenger Information

| Feature         | Description                           |
| :-------------- | :------------------------------------ |
| Gender          | Passenger gender (Female/Male)        |
| Customer Type   | Type of passenger (Loyal/Disloyal)    |
| Age             | Passenger age in years (numerical)    |
| Type of Travel  | Purpose of travel (Personal/Business) |
| Class           | Flight class (Business/Eco/Eco Plus)  |
| Flight Distance | Flight distance in miles (numerical)  |

### Service Quality Ratings

All service ratings are numerical: 1–5 or 0 (0 = not rated).

| Feature                           | Description                              |
| :-------------------------------- | :--------------------------------------- |
| Inflight wifi service             | Satisfaction with WiFi on board          |
| Departure/Arrival time convenient | Satisfaction with departure/arrival time |
| Ease of Online booking            | Satisfaction with online booking process |
| Gate location                     | Satisfaction with gate location          |
| Food and drink                    | Satisfaction with food and beverages     |
| Online boarding                   | Satisfaction with online check-in        |
| Seat comfort                      | Satisfaction with seat comfort           |
| Inflight entertainment            | Satisfaction with entertainment options  |
| On-board service                  | Satisfaction with onboard service        |
| Leg room service                  | Satisfaction with legroom                |
| Baggage handling                  | Satisfaction with baggage handling       |
| Checkin service                   | Satisfaction with check-in service       |
| Cleanliness                       | Satisfaction with cabin cleanliness      |

### Delay Ratings

| Feature                    | Description                |
| :------------------------- | :------------------------- |
| Departure Delay in Minutes | Departure delay in minutes |
| Arrival Delay in Minutes   | Arrival delay in minutes   |

### Target Variable

| Feature      | Description                                                      |
| :----------- | :--------------------------------------------------------------- |
| satisfaction | Passenger satisfaction (True = Satisfied, False = Not Satisfied) |
