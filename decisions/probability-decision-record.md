# Probability Decision Record

## Record status

Complete using actual transaction `122` from `data/credit_card_fraud_10k.csv`.

Policy: `POLICY_B`

Policy version: `V1-POLICY-B-APPROVE-FRAUD-COST-20`

Dataset version: `credit_card_fraud_10k.csv`, 10,000 rows, inspected 2026-09-07

## Visual decision trace

```text
Prior
  ↓ EARLY_HOUR=True
Posterior: fraud 0.050509
  ↓ FOREIGN_TRANSACTION=True
Posterior: fraud 0.241011
  ↓ costs too close
GET_MORE_EVIDENCE
  ↓ LOCATION_MISMATCH=False
Posterior: fraud 0.152915
  ↓ APPROVE
  ↓ reveal label
Actual state: FRAUDULENT
```

## Observed transaction

```text
transaction_id = 122
transaction_hour = 1
foreign_transaction = 1
location_mismatch = 0
device_trust_score = 36
velocity_last_24h = 2
```

Derived evidence:

```text
EARLY_HOUR = True
FOREIGN_TRANSACTION = True
LOCATION_MISMATCH = False
LOW_DEVICE_TRUST = True
HIGH_VELOCITY = False
```

The hidden `is_fraud` label was not passed to the decision functions.

## Hidden states and prior beliefs

```text
LEGITIMATE: 9,849 / 10,000 = 0.984900
FRAUDULENT:   151 / 10,000 = 0.015100
```

## Relevant likelihoods

```text
P(EARLY_HOUR=True | LEGITIMATE) = 0.236674
P(EARLY_HOUR=True | FRAUDULENT) = 0.821192

P(FOREIGN_TRANSACTION=True | LEGITIMATE) = 0.090974
P(FOREIGN_TRANSACTION=True | FRAUDULENT) = 0.543046

P(LOCATION_MISMATCH=False | LEGITIMATE) = 0.920296
P(LOCATION_MISMATCH=False | FRAUDULENT) = 0.523179
```

## Sequential belief updates

### Step 1: EARLY_HOUR=True

```text
LEGITIMATE unnormalized = 0.984900 × 0.236674 = 0.233100
FRAUDULENT unnormalized = 0.015100 × 0.821192 = 0.012400
normalizer = 0.245500

P(LEGITIMATE) = 0.949491
P(FRAUDULENT) = 0.050509
```

Policy B expected costs:

```text
EC(APPROVE) = 0.050509 × 20 = 1.010183
EC(BLOCK) = 0.949491 × 6 = 5.696945
Action after this update: APPROVE
```

The agent still processes `FOREIGN_TRANSACTION` because both initial evidence items must be revealed.

### Step 2: FOREIGN_TRANSACTION=True

```text
LEGITIMATE unnormalized = 0.949491 × 0.090974 = 0.086379
FRAUDULENT unnormalized = 0.050509 × 0.543046 = 0.027429
normalizer = 0.113808

P(LEGITIMATE) = 0.758989
P(FRAUDULENT) = 0.241011
```

Policy B expected costs:

```text
EC(APPROVE) = 0.241011 × 20 = 4.820212
EC(BLOCK) = 0.758989 × 6 = 4.553936
cost gap = 0.266276
```

Because the gap is at most the design margin `0.5` and unused evidence remains:

```text
Action: GET_MORE_EVIDENCE
```

### Step 3: LOCATION_MISMATCH=False

False evidence uses complement likelihoods:

```text
P(LOCATION_MISMATCH=False | LEGITIMATE) = 1 - 0.079704 = 0.920296
P(LOCATION_MISMATCH=False | FRAUDULENT) = 1 - 0.476821 = 0.523179
```

```text
P(LEGITIMATE) = 0.847085
P(FRAUDULENT) = 0.152915
```

Policy B expected costs:

```text
EC(APPROVE) = 0.152915 × 20 = 3.058298
EC(BLOCK) = 0.847085 × 6 = 5.082510
Action: APPROVE
```

The policy stops here because the costs are no longer within the uncertainty margin.
`LOW_DEVICE_TRUST` and `HIGH_VELOCITY` remain unused.

## Final action and evaluation label

```text
Final action: APPROVE
Actual state revealed afterward: FRAUDULENT
Error type: FALSE_NEGATIVE
```

The error is not fabricated: transaction `122` is an actual row in the CSV. The policy approved because the posterior after the collected evidence made APPROVE lower-cost, even though the hidden historical label was fraudulent.
