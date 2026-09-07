# Credit Card Fraud Decision Agent

This repository contains a beginner-friendly Week 1 project that demonstrates how an agent can make a credit-card fraud decision using:

- historical probabilities calculated from a CSV dataset,
- sequential Bayesian belief updates,
- deterministic decision costs,
- and a small set of explainable actions.

The agent reasons about two hidden states:

- `LEGITIMATE`
- `FRAUDULENT`

Its possible actions are:

- `APPROVE`
- `GET_MORE_EVIDENCE`
- `BLOCK`
- `HUMAN_REVIEW`

## Project diagram

The complete process is available as an editable Excalidraw file:

[Open the Credit Card Fraud Agent process diagram](./credit-card-fraud-agent-process.excalidraw)

You can open it by visiting [excalidraw.com](https://excalidraw.com/) and choosing **Open** or **Import**, then selecting the `.excalidraw` file.

Original shared Excalidraw link: [view online](https://excalidraw.com/#json=EVjFhksgkkTuiSkspnFs5,SJkDpFbTXfQ7mov6l36jOA)

The diagram shows this flow:

```text
CSV data
  ↓
Feature engineering
  ↓
Historical priors and likelihoods
  ↓
Initial evidence
  ↓
Bayesian belief update
  ↓
Expected decision costs
  ↓
Policy action
  ↓
Evaluation → Failure analysis → Probability decision record
```

## Repository structure

```text
data/
    credit_card_fraud_10k.csv
decisions/
    probability-decision-record.md
experiments/
    agent_simulation.ipynb
credit-card-fraud-agent-process.excalidraw
discussion-record.md
project-tracking.md
research-file.md
README.md
```

## Important scope boundary

This is a simple Week 1 educational agent. It does not use machine-learning classifiers, neural networks, Random Forest, XGBoost, entropy ranking, or Expected Information Gain.

The feature thresholds, evidence order, decision costs, and uncertainty margin are explicitly documented design assumptions. Class counts, priors, and evidence likelihoods are calculated from the dataset.

The original raw CSV remains unchanged.
