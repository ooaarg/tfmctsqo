# Host contract

`mctsopt_core/include/mctsopt.h` is the native interface. It has no DBMS includes
or global mutable search state. A host supplies per-call state through `user`;
concurrent searches require independent host state too. The public structs are an
experimental source API, not a stable binary ABI. Link the engine into each adapter.

## Relation ownership and costing

`MctsoptRel` is an opaque, stable host handle. Handles and their relation identities
must remain valid until the caller has consumed the result and trace. The host owns
all allocation. `alloc` must return suitably aligned storage or raise a host error;
it must not return NULL. All allocated blocks survive until the arena is released.
The initial list has at least two non-null handles, in native optimizer order.
Each handle is one initial clump, even when it represents a prejoined subproblem.

`join(left, right, store)` returns NULL for an illegal pair. Otherwise it returns a
native relation, which may have no usable path (`cost` then returns `DBL_MAX`).
`store` distinguishes expansion from rollout cache insertion. `rank` evaluates a
candidate for top-k filtering, producing its ranking score and estimated rows;
its cache and native memo effects must agree with the source implementation.
The ranking enum is passed to the host because it may need different native
metadata and cache behavior for cost, row, mixed and selectivity ranking.

`desirable` expresses preference, not legality: other pairs are a fallback when no
desirable candidate exists. `overlap` checks relation-set intersection; `complete`
checks whether a terminal expansion covers the entire current join subproblem.
Initial clumps must form a partition, and each successful join must preserve the
union of its inputs. This makes completion of rollouts follow from the partition.
Native `cost` may change as memoized paths improve; handles are not immutable costs.

## Speculative scopes

`begin_scope(temporary)` saves memo/list state; `end_scope` restores it. Temporary
scopes isolate rollouts. They must not invalidate returned handles, even when those
handles were produced inside the scope. The PG adapter retains its evaluation arena
until PostgreSQL cleans up the parent planner context. `restart` resets the native
cost cache and per-phase hit counter without invalidating earlier winning plans.

Callback order matters. By default, top-k ranks candidates only when k is positive
and their count exceeds k; top_k=0 skips ranking and retains every action. Optional
`rank_all=true` ranks all unpruned actions while preserving their order, allowing
native memo enrichment to be studied separately from pruning. Pruning uses stable
insertion and does not reorder ties. Speculative path
construction and cache lookup are kept distinct. A replacement host can implement a
different cost model, but that is not a parity-preserving replacement of the PG host.

## Randomness and search controls

The core uses a 48-bit LCG matching `erand48`, seeded by the original SplitMix-style
mix. Nonzero seed is deterministic. Zero asks `entropy` for a seed at each reseeding
point; this is intentionally not reproducible. Luby rollouts preserve the original
XOR/reseed sequence. Avoid fast-math, floating-point reassociation and contraction
when checking exact traces across builds.

`depth=0` means unlimited in the core. The inherited PG GUC accepts 2 through 100;
use 100 for JOB's effectively unlimited-depth checks. The `rollouts` count applies
to depth-limited/full-budget repeated rollout groups. Ordinary expansion performs
one rollout, matching the original. Full-budget search uses classic UCB after
frontier exhaustion; other search iterations use UCT-Extreme or mean aggregation.

`legacy_selection=false` fixes premature exhaustion of live sibling subtrees and
selection of live children after failed-rollout scores. `legacy_normalization=false`
uses current reward bounds for normalized best/average scoring; ordinary `neg_log`
and `neg_cost` keep their historical score paths. Both switches default to false
in zero-initialized configs; a host requiring an attribution control can explicitly
select historical behavior. These fixes can change previously matching host traces.

The result status distinguishes success, no complete plan, and invalid inputs.
Invalid configurations or cumulative counter overflow are rejected before callbacks.
The host must still bound native planner memory and wall time through allocation
and `interrupt`. PG does so through its memory contexts and CHECK_FOR_INTERRUPTS.

## Explicit initialization and diagnostics

`MctsoptHost.initial_plan` is an optional callback appended after `snapshot`. It
returns an existing complete native incumbent, or NULL. Search reads it once before
phases; a valid retained incumbent has an empty result order and best phase/iteration
of -1. Hosts must recognize that successful empty-order result. A pure host must
pass NULL and independently ensure its memo contains no greedy-produced plan:
disabling the callback alone does not remove initialized memo alternatives.

`mctsopt_greedy(config, host, initial, n)` is an explicit helper, never called by
`mctsopt_search`. It chooses the minimum finite cost of admissible merge candidates,
using stable ties and the existing shape/orientation/overlap/connectivity rules and
rollout relaxation. Ranking is fixed to MCTSOPT_COST independently of config.ranking.
It does not use RNG, restart or initial_plan callbacks. Its order and relation handles
have the same arena lifetime as search results; serialize/copy them before teardown.
NO_PLAN and failed/relaxed/cross-product steps are reported explicitly. Desirability
is captured for every candidate before ranking can mutate the host graph.

`MctsoptResult.iterations` and the historical `rollouts` field count evaluated
iterations/batches; `rollout_calls` counts actual rollout invocations. Ordinary
expansion has one invocation even if config.rollouts is larger; depth-limited and
full-budget groups can have several. `failures` is cumulative across phases.
`attempted_iterations`, `completed_phases`, `exhausted_phases`, and `stop_reason`
provide actual-work accounting; `mctsopt_stop_reason_name` formats the enum. Patience
counts phases without strict incumbent-cost improvement, not milliseconds or a
query-execution timeout. Cost stagnation does not prove execution-time stagnation.

The appended host callback preserves existing aggregate source initialization:
missing trailing members are zero-initialized. Struct sizes and MctsoptNode layout
have changed, so this is not binary compatibility. Rebuild core and adapter together.
The unchanged PG positional initializer receives initial_plan=NULL and its designated
config receives the new flags as false. An isolated PostgreSQL 19devel rebuild passed both SQL regression suites. Five
seeds across default, random, and Luby rollout produced identical plans, answers,
and non-timing traces before and after the rank-biased rollout addition. Its legacy `result.rollouts * config.rollouts` statistic is not an exact invocation count.

Rank-biased rollout is opt-in in the C core (`rank_biased_rollout=false` when
omitted). `rollout_epsilon` weights uniform sampling against inverse cost midrank.
The original random/Luby RNG path is unchanged. The optional trailing
`candidate_observation` callback runs after completed rollouts and terminal
observations; it must not change search rewards. PostgreSQL leaves it NULL.

## Traces and materialization

Trace callbacks are optional and active only with `trace=true`. Iteration events
occur before that iteration's improvement. Phase events include its best merge
order. Snapshot events expose the search tree, including pruned candidates; callers
must copy any data they wish to retain beyond the arena's lifetime.

The result contains the winning native handle, cost and ordered list of merge pairs.
The adapter materializes/replays that order according to its DBMS. The PG adapter
preserves the reference's replay side effects and returns its original incumbent.
Changing that behavior is a separate correctness experiment, not part of extraction.

## DuckDB host

`duckdb_mctsopt/core_adapter.cpp` uses interned native relation sets as handles and
looks up current memo nodes to read costs/cardinalities. It retains speculative
memo updates and joins across phases; its scope/restart callbacks do not roll
those updates back. Search allocations are owned by a per-call DuckDB arena.
Preparation retains leaf plans only and enables required graph edges without any
native search. Pure mode stops preparation there; explicit greedy mode invokes
native or shape-aware initialization once before search. The core order is replayed
before releasing its arena. The winning
native memo plan can recombine explored subplans, so search shape constraints do
not guarantee final native plan shape. A search returning NO_PLAN, or failure of an
explicitly requested initializer, triggers fresh native search without selecting
extensions again. If that also fails, DuckDB retains its input logical plan. Both
outcomes have counters and WARNING log events. Cancellation and other errors
propagate. Initializer failure skips Monte Carlo search and is distinguished by
the greedy_initialization_failed fallback reason. The extension enforces
its region-local min_relations threshold before selecting this search. See the
[DuckDB README](../duckdb_mctsopt/README.md) for settings and native hook requirements.

The DuckDB default additionally retains complete candidate trees, including eligible
attempted roots rejected by native cost. It selects from a cost/risk frontier under
a cost guard and verifies the frozen tree on replay. `retention_enabled=false`
restores memo-only selection. Metadata collection reads existing filters and scan
statistics; it does not rewrite predicates. The candidate-cost access patch exports
cached cost terms used for retention and replay checks, without changing native costs.

The C header supports C++ callers through linkage guards. The DuckDB build enables
exception unwinding through the C engine so native cancellation and allocation
errors release the C++ arena. DuckDB-specific defaults and diagnostics are listed
in the adapter README. The core trace callbacks are not exposed in SQL yet.
