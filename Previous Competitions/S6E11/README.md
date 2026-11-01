# Predicting Credit Card Fraud

## Playground Series - Season 6, Episode 11

![Header](https://www.kaggle.com/competitions/125225/images/header)

## Overview

Welcome to the 2026 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

**Your Goal:** Predicting credit card fraud.

## Evaluation

Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

### Submission File

For each `id` in the test set, you must predict a probability for the `is_fraud` variable. The file should contain a header and have the following format:

```csv
id,is_fraud
699960,0.3
699961,0.9
699962,0.2
```

## Timeline

- **Start Date** - November 1, 2026
- **Entry Deadline** - Same as the Final Submission Deadline
- **Team Merger Deadline** - Same as the Final Submission Deadline
- **Final Submission Deadline** - November 30, 2026

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## About the Tabular Playground Series

The goal of the Tabular Playground Series is to provide the Kaggle community with a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science. The duration of each competition will generally only last a few weeks, and may have longer or shorter durations depending on the challenge. The challenges will generally use fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc.

### Synthetically-Generated Datasets

Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series six years ago, and that goal is to produce datasets that have far fewer artifacts. Please feel free to give us feedback on the datasets for the different competitions so that we can continue to improve!

## Dataset Description

The dataset for this competition (both train and test) was inspired by the Credit Card Fraud Detection 2026 dataset. Feature distributions are close to, but not exactly the same, as the original.

## Files

- `train.csv` - the training set, with `is_fraud` as target
- `test.csv` - the test set, used to predict the probability for `is_fraud`
- `sample_submission.csv` - a sample submission file in the correct format
- `credit_card_fraud_2026.csv` - the original dataset used to inspire the synthetic competition data

---

## Competition Dataset

### Overview

The competition dataset (train.csv and test.csv) is a synthetically generated dataset of credit card transactions, carrying amount, velocity, location, device, authentication and merchant-risk signals used to predict whether a transaction is fraudulent. Unlike the classic fraud benchmark, nothing is PCA-anonymised into `V1…V28`: every column is named and readable, so a feature-importance chart becomes a story rather than "V14 matters, nobody knows why".

- **Train Set**: 699,960 rows × 26 columns
- **Test Set**: 299,984 rows × 25 columns (same columns as train, minus `is_fraud`)
- **Missing Values**: none, in either file
- **Positive Rate**: 1.6700% (11,689 of 699,960 train rows) - the same prior the host ships as the constant column in `sample_submission.csv`
- **Id Split**: train `0 - 699,959`, test `699,960 - 999,943`

### 🔍 Use Cases

- Predicting credit card fraud (binary classification under severe class imbalance)
- Reporting PR-AUC, precision and recall alongside the scored ROC AUC, and explaining why
- Tuning the decision threshold to buy recall at a known false-positive cost
- Comparing SMOTE, class weights and undersampling on the same model
- Feature engineering from the velocity signals `hours_since_last_txn` and `txn_count_last_24h`
- Model explainability for a risk team, on named features instead of anonymous components

### Features

| Column                         | Type    | Description                                                                                                                                   |
| :----------------------------- | :------ | :-------------------------------------------------------------------------------------------------------------------------------------------- |
| id                             | Integer | Unique transaction identifier (train `0 - 699,959`, test `699,960 - 999,943`)                                                                  |
| amount_usd                     | Float   | Transaction amount in USD (1.00 - 6,872.69)                                                                                                    |
| hours_since_last_txn           | Float   | Elapsed hours since the cardholder's previous transaction (0.01 - 51.98)                                                                        |
| txn_count_last_24h             | Integer | Transaction count in the trailing 24 hours (0 - 11)                                                                                             |
| distance_from_home_km          | Float   | Distance of the transaction from the cardholder's home (0 - 57.93 km)                                                                           |
| card_age_months                | Integer | Age of the card in months (25 - 73)                                                                                                             |
| customer_age                   | Integer | Cardholder age in years (18 - 81)                                                                                                               |
| account_balance_usd            | Float   | Account balance in USD (58.25 - 99,701.40)                                                                                                      |
| cvv_retry_count                | Integer | Repeated CVV failures on the card (0 - 3)                                                                                                       |
| velocity_score                 | Float   | Behavioural velocity signal (0.0 - 72.9)                                                                                                        |
| time_of_day_hour               | Integer | Hour of the day the transaction was made (0 - 23)                                                                                               |
| day_of_week                    | Integer | Day of the week the transaction was made (0 - 6)                                                                                                |
| merchant_risk_score            | Float   | Risk level of the accepting merchant (0 - 100)                                                                                                  |
| prior_disputes                 | Integer | Disputes already recorded on the card (0 - 4)                                                                                                   |
| merchant_category              | String  | Category of the accepting merchant - 12 generic labels, including Crypto Exchange, Gift Cards and Streaming                                     |
| card_type                      | String  | Card network - Visa, Mastercard, Amex, Discover, RuPay                                                                                          |
| auth_method                    | String  | Authentication used - 3D Secure, Biometric, OTP, PIN, No Authentication                                                                         |
| channel                        | String  | Payment channel - Contactless, In-App, Online, POS, ATM                                                                                         |
| device_type                    | String  | Device the transaction was made from - iPhone, Android Phone, Tablet, Mac, Windows PC, Smart Watch, POS Terminal, ATM Machine                   |
| is_foreign_transaction         | Boolean | Transaction was made outside the cardholder's country                                                                                           |
| is_new_merchant                | Boolean | Cardholder has not transacted with this merchant before                                                                                         |
| used_vpn                       | Boolean | VPN was used for the transaction                                                                                                                |
| ip_country_mismatch            | Boolean | IP country differs from the billing country                                                                                                     |
| billing_shipping_mismatch      | Boolean | Billing and shipping details disagree                                                                                                           |
| is_ai_generated_scam_attempt   | Boolean | AI-driven social-engineering attempt on this transaction                                                                                         |
| is_fraud                       | Float   | **Target variable**: transaction is fraudulent (1.0 = fraud, 0.0 = normal)                                                                      |

---

## Original Dataset: Credit Card Fraud Detection 2026

### About Dataset

#### Dataset Overview

Every fraud dataset hides behind `V1…V28`. This one does not - 20,000 transactions in 2026 with columns you can actually read, reason about and explain to a risk team. Every feature is named and meaningful, and the data carries 2026-era patterns the 2013-era classics predate: an `is_ai_generated_scam_attempt` flag for AI-driven social-engineering, modern auth methods (3D Secure, OTP, biometric), contactless and in-app channels, VPN usage and crypto-exchange merchants.

**1.70% of transactions are fraud (339 of 20,000).** That imbalance is the point - it is what makes this genre worth practising on. Accuracy will happily tell you 98.3% and mean nothing; you will need precision/recall, PR-AUC, resampling and threshold tuning to get anywhere. **26 columns, 20,000 rows, zero missing values.**

The original file is keyed by `transaction_id` (`1 - 20,000`) where the competition files use `id`, and its tails reach further than the competition's: `hours_since_last_txn` to 87.05, `distance_from_home_km` to 216.19 km, `txn_count_last_24h` to 12 and `account_balance_usd` to 127,126.

#### Potential Use Cases

- Build a fraud classifier and report PR-AUC, not accuracy - then explain why
- Find which three signals catch the most fraud, and plot the feature importance
- Tune the decision threshold: how much recall can you buy per false positive?
- Compare SMOTE, class weights and undersampling on the same model
- Test whether AI-generated scam attempts are harder to catch than ordinary fraud
- Test whether strong authentication (biometric, 3D Secure) actually suppresses fraud
- Engineer your own velocity features from `hours_since_last_txn` and `txn_count_last_24h`

### Transaction Signals

| Feature                  | Description                                         |
| :----------------------- | :-------------------------------------------------- |
| amount_usd               | Transaction amount in USD                           |
| velocity_score           | Behavioural velocity signal                         |
| hours_since_last_txn     | Time since the previous transaction                 |
| txn_count_last_24h       | Transaction volume in the trailing 24 hours         |
| cvv_retry_count          | Repeated CVV failures                               |
| time_of_day_hour         | Hour of the day of the transaction                  |
| day_of_week              | Day of the week of the transaction                  |

### Cardholder & Account

| Feature             | Description                        |
| :------------------ | :--------------------------------- |
| customer_age        | Age of the cardholder in years     |
| card_age_months     | Age of the card in months          |
| account_balance_usd | Account balance in USD             |
| prior_disputes      | Disputes already recorded on the card |
| card_type           | Card network                       |

### Location & Device Risk

| Feature                     | Description                                 |
| :-------------------------- | :------------------------------------------ |
| distance_from_home_km       | Distance from the cardholder's home         |
| ip_country_mismatch         | IP country differs from the billing country |
| billing_shipping_mismatch   | Billing and shipping details disagree       |
| is_foreign_transaction      | Transaction made outside the home country   |
| used_vpn                    | VPN was used for the transaction            |
| device_type                 | Device the transaction was made from        |

### Authentication & Merchant

| Feature                        | Description                                   |
| :----------------------------- | :-------------------------------------------- |
| auth_method                    | 3D Secure, OTP, biometric, PIN or none        |
| channel                        | Contactless, in-app, online, POS or ATM       |
| merchant_category              | Generic category of the accepting merchant    |
| merchant_risk_score            | Risk level of the accepting merchant          |
| is_new_merchant                | First transaction with this merchant          |
| is_ai_generated_scam_attempt   | AI-driven social-engineering attempt          |

### Target Variable

| Feature    | Description                                                          |
| :--------- | :------------------------------------------------------------------- |
| is_fraud   | Transaction is fraudulent (1 = fraud, 0 = not fraud)                 |

**Note on data:** This dataset is fully synthetic, generated with NumPy (see `gen.py`) for machine-learning practice. It is not derived from any real cardholder, bank, payment processor or transaction record, and no real customer data is included. Merchant categories are generic labels, not companies. Use freely under CC0.

---

## Citation

Yao Yan, Walter Reade, Elizabeth Park. Predicting Credit Card Fraud. https://www.kaggle.com/competitions/playground-series-s6e11, 2026. Kaggle.
