# job-jobcomplex-full

JOB & JOBComplex (FULL): all 143 queries × {luby, single}.

- **Started**: 2026-05-25T16-48-03
- **Git revision**: `60cf406b8afc90b860c75c938ecf2001f1b25320`
- **Run group**: Paper
- **n_seeds**: 3
- **Configs**: single, dp, geqo
- **Reward modes**: neg_log
- **UCT aggregations**: best
- **Queries** (143): 29a, 29b, 29c, 28a, 28b, 28c, 33a, 33b, 33c, 24a, 24b, 26a, 26b, 26c, 27a, 27b, 27c, 30a, 30b, 30c, 22a, 22b, 22c, 22d, 23a, 23b, 23c, 31a, 31b, 31c, 19a, 19b, 19c, 19d, 20a, 20b, 20c, 13a, 13b, 13c, 13d, 15a, 15b, 15c, 15d, 21a, 21b, 21c, 25a, 25b, 25c, 7a, 7b, 7c, 9a, 9b, 9c, 9d, 11a, 11b, 11c, 11d, 12a, 12b, 12c, 14a, 14b, 14c, 16a, 16b, 16c, 16d, 8a, 8b, 8c, 8d, 10a, 10b, 10c, 17a, 17b, 17c, 17d, 17e, 17f, 18a, 18b, 18c, 32a, 32b, 1a, 1b, 1c, 1d, 2a, 2b, 2c, 2d, 4a, 4b, 4c, 5a, 5b, 5c, 6a, 6b, 6c, 6d, 6e, 6f, 3a, 3b, 3c, 5, 20, 9, 11, 12, 17, 25, 8, 13, 19, 23, 29, 21, 28, 1, 3, 4, 6, 7, 15, 18, 24, 26, 2, 10, 14, 16, 27, 30, 22

## Selected MCTS cells
### combo `d10_tk10_S_sb480`
```
single:
  depth = 10
  expand_strategy = cost
  exploration_constant = 1.41
  full_budget = False
  patience = 3
  rollouts_per_leaf = 1
  start_budget = 480
  top_k = 10
```
