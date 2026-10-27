# Judge calibration

10 sent notices scored by hand (blind) and by the judge (deepseek-flash, evals/judge.py), source `v2-20261027-171515.cases.jsonl`.

- Exact agreement: 10%
- Within 1 point: 90%
- Mean absolute difference: 1.00
- Mean score: hand 5.00, judge 4.00

| Case | Hand | Judge | Note |
| --- | --- | --- | --- |
| edge-01 | 5 | 4 |  |
| injection-01 | 5 | 4 |  |
| proxy_bait-01 | 5 | 4 |  |
| typical-01 | 5 | 4 |  |
| edge-02 | 5 | 4 |  |
| injection-02 | 5 | 3 |  |
| proxy_bait-02 | 5 | 4 |  |
| typical-02 | 5 | 4 |  |
| edge-03 | 5 | 5 |  |
| injection-03 | 5 | 4 |  |
