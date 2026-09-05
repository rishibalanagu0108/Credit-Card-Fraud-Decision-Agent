# Credit Card Fraud Decision Agent — Research File

## Stage 1: Problem definition

### Problem statement

The agent observes a credit-card transaction and must decide whether to `APPROVE`, `GET_MORE_EVIDENCE`, `BLOCK`, or send the case to `HUMAN_REVIEW` because the true fraud state is unknown during decision-making.

### Project objective

Build a beginner-friendly Week 1 agent using historical probabilities, sequential Bayesian belief updating, and deterministic cost-based decision logic. The agent must remain manually traceable for one transaction.

## Dataset choice

The raw dataset is `data/credit_card_fraud_10k.csv`. Stage 1 verification found 10,000 rows, all expected columns, and zero missing values. The original CSV is not modified.

## Stage 2: Historical priors and likelihoods

### Visual: where the probabilities come from

```mermaid
flowchart LR
    R[Historical rows] --> C[Count rows by is_fraud]
    C --> P[Prior: state count / all rows]
    R --> E[Count evidence rows within each state]
    E --> L[Likelihood: evidence count / state count]
```

Plain-text version:

```text
All historical rows
      |
      +--> Count LEGITIMATE and FRAUDULENT -> priors
      |
      +--> Count evidence within each state -> likelihoods
```

### Class counts and priors

These are **DATASET-DERIVED**:

| Hidden state | Historical count | Calculation | Prior |
|---|---:|---|---:|
| `LEGITIMATE` | 9,849 | `9,849 / 10,000` | `0.984900` |
| `FRAUDULENT` | 151 | `151 / 10,000` | `0.015100` |
| **Total** | **10,000** | `9,849 + 151` | **1.000000** |

The prior is our starting belief before looking at transaction evidence. It reflects the class frequency in this historical dataset.

### Likelihoods

Each likelihood is calculated as:

```text
P(evidence=True | state)
    = number of rows where evidence is True and state matches
      / number of rows where state matches
```

The following values are **DATASET-DERIVED**:

| Evidence | State | True count | State count | `P(True \| State)` | `P(False \| State)` |
|---|---|---:|---:|---:|---:|
| `EARLY_HOUR` | `LEGITIMATE` | 2,331 | 9,849 | 0.236674 | 0.763326 |
| `EARLY_HOUR` | `FRAUDULENT` | 124 | 151 | 0.821192 | 0.178808 |
| `FOREIGN_TRANSACTION` | `LEGITIMATE` | 896 | 9,849 | 0.090974 | 0.909026 |
| `FOREIGN_TRANSACTION` | `FRAUDULENT` | 82 | 151 | 0.543046 | 0.456954 |
| `LOCATION_MISMATCH` | `LEGITIMATE` | 785 | 9,849 | 0.079704 | 0.920296 |
| `LOCATION_MISMATCH` | `FRAUDULENT` | 72 | 151 | 0.476821 | 0.523179 |
| `LOW_DEVICE_TRUST` | `LEGITIMATE` | 3,230 | 9,849 | 0.327952 | 0.672048 |
| `LOW_DEVICE_TRUST` | `FRAUDULENT` | 129 | 151 | 0.854305 | 0.145695 |
| `HIGH_VELOCITY` | `LEGITIMATE` | 3,178 | 9,849 | 0.322672 | 0.677328 |
| `HIGH_VELOCITY` | `FRAUDULENT` | 88 | 151 | 0.582781 | 0.417219 |

For example:

```text
P(FOREIGN_TRANSACTION=True | FRAUDULENT)
= 82 fraudulent rows with foreign_transaction = 1 / 151 fraudulent rows
= 82 / 151
= 0.543046
```

For false evidence, we use the complement of the true-evidence probability:

```text
P(FOREIGN_TRANSACTION=False | FRAUDULENT)
= 1 - P(FOREIGN_TRANSACTION=True | FRAUDULENT)
= 1 - 0.543046
= 0.456954
```

The two values add to 1 for each evidence/state pair. This lets the later Bayesian update handle both `True` and `False` observations.

### Stage 2 verification

```text
class count sum: 10,000
prior sum: 1.0
all likelihoods in [0, 1]: yes
all true/false complements sum to 1: yes
```

## Hidden states

The hidden state is represented by the historical label:

| Historical label | Hidden state |
|---:|---|
| `is_fraud = 0` | `LEGITIMATE` |
| `is_fraud = 1` | `FRAUDULENT` |

The label will be hidden while the agent decides and revealed only afterward for evaluation.

### Visual: what the agent can see

```mermaid
flowchart LR
    T[Raw transaction] --> E[Five binary evidence flags]
    T --> H[Historical label: is_fraud]
    E --> A[Agent observes evidence]
    H -. hidden during decision .-> A
    A --> D[Later: evaluate action against true state]
```

Plain-text version:

```text
Raw transaction
      |
      v
Five evidence flags  --->  Agent makes a decision
                                  ^
                                  |
                    is_fraud stays hidden until evaluation
```

The important separation is that `is_fraud` helps us learn historical probabilities, but it is not available to the agent while it is deciding about a new transaction.

## Selected columns for V1

| Column | Use | Reason |
|---|---|---|
| `transaction_hour` | Selected evidence input | Used to derive `EARLY_HOUR`. |
| `foreign_transaction` | Selected evidence input | Direct geographic risk signal. |
| `location_mismatch` | Selected evidence input | Direct mismatch signal. |
| `device_trust_score` | Selected evidence input | Used to derive low device trust. |
| `velocity_last_24h` | Selected evidence input | Used to derive unusually high recent activity. |
| `is_fraud` | Historical target / hidden state | Used to calculate historical probabilities and evaluate decisions; not revealed during a live decision. |

## Excluded from V1 evidence

| Column | Reason for exclusion |
|---|---|
| `transaction_id` | Identifier only; it does not describe transaction risk. |
| `merchant_category` | Excluded to keep the first version small and manually traceable. |
| `cardholder_age` | Excluded from the initial evidence model to avoid unnecessary demographic complexity. |
| `amount` | Excluded from V1 so the first model focuses on the specified behavioral and location signals. |

## Feature-engineering rules

These rules are **DESIGN ASSUMPTIONS**. They create binary observations; they are not probabilities derived from the dataset.

| Evidence variable | Rule | Meaning |
|---|---|---|
| `EARLY_HOUR` | `transaction_hour <= 5` | The transaction occurred from midnight through 05:00. |
| `FOREIGN_TRANSACTION` | `foreign_transaction == 1` | The transaction is marked foreign. |
| `LOCATION_MISMATCH` | `location_mismatch == 1` | The transaction is marked as having a location mismatch. |
| `LOW_DEVICE_TRUST` | `device_trust_score < 50` | The device trust score is below 50. |
| `HIGH_VELOCITY` | `velocity_last_24h >= 3` | At least 3 recent transactions are recorded in the 24-hour velocity field. |

The exact boundary choices (`<= 5`, `< 50`, and `>= 3`) are V1 design choices. Stage 2 will calculate how often each resulting evidence variable occurs for each historical hidden state.

### Visual: raw columns becoming evidence

```text
transaction_hour       --(<= 5)-->  EARLY_HOUR
foreign_transaction    --(== 1)-->  FOREIGN_TRANSACTION
location_mismatch      --(== 1)-->  LOCATION_MISMATCH
device_trust_score     --(< 50)-->  LOW_DEVICE_TRUST
velocity_last_24h      --(>= 3)-->  HIGH_VELOCITY
```

The arrows above are feature-engineering rules. They do not say how likely fraud is. Stage 2 will estimate those probabilities from rows grouped by `is_fraud`.

### Visual: the Week 1 reasoning pipeline

```mermaid
flowchart TD
    S1[1. Inspect data and define evidence] --> S2[2. Calculate priors and likelihoods]
    S2 --> S3[3. Update belief with Bayes]
    S3 --> S4[4. Compare expected costs]
    S4 --> S5[5. Collect evidence sequentially]
    S5 --> S6[6. Evaluate 50 held-out cases]
    S6 --> S7[7. Inspect failures]
    S7 --> S8[8. Record one complete decision]
```

Plain-text version:

```text
Inspect data
    -> Calculate historical probabilities
    -> Update belief
    -> Compare decision costs
    -> Get more evidence when needed
    -> Evaluate
    -> Analyze failures
    -> Record one complete decision
```

## Stage 1 verification output

The following was verified directly from the CSV:

```text
Rows: 10,000
Required columns present: yes
Missing values: 0
transaction_hour range: 0–23
foreign_transaction values: 0, 1
location_mismatch values: 0, 1
device_trust_score range: 25–99
velocity_last_24h range: 0–9
is_fraud values: 0, 1
```

Derived evidence counts in the 10,000-row dataset are included only as a Stage 1 sanity check, not as likelihoods:

```text
EARLY_HOUR: 2,455 true
FOREIGN_TRANSACTION: 978 true
LOCATION_MISMATCH: 857 true
LOW_DEVICE_TRUST: 3,359 true
HIGH_VELOCITY: 3,266 true
```

## Not implemented yet

- Bayesian posterior updates
- Decision costs and uncertainty margin
- Agent action selection
- Held-out evaluation
- Failure analysis
- Reddit/community research

These will be added only in their assigned stages.
