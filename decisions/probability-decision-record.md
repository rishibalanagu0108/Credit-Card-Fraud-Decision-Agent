# Probability Decision Record

This record will be populated in Stage 8 using one actual uncertain transaction. No decision record is claimed yet.

Status: **PENDING — Bayesian policy has not been implemented.**

## Decision trace preview

The completed record will follow this visual sequence:

```text
Prior belief
    |
    v
First observed evidence
    |
    v
Updated posterior
    |
    v
Expected APPROVE/BLOCK costs
    |
    +--> act now
    |
    +--> GET_MORE_EVIDENCE -> new posterior -> action
```

The actual state will be recorded only after the action is complete.
