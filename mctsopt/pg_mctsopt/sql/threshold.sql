\getenv libdir PG_LIBDIR
\getenv dlsuffix PG_DLSUFFIX
\set module :libdir '/pg_mctsopt' :dlsuffix
LOAD :'module';
SET pg_mctsopt.min_relations = 12;
SET pg_mctsopt.random_seed = 42;
SET pg_mctsopt.start_budget = 25;
SET pg_mctsopt.phases = 1;
SET pg_mctsopt.luby_enabled = false;
SET join_collapse_limit = 64;
SET from_collapse_limit = 64;
SET statement_timeout = '30s';
DO $$
BEGIN
    FOR i IN 1..14 LOOP
        EXECUTE format('CREATE TEMP TABLE t%s AS SELECT generate_series(1, 10) AS id', i);
    END LOOP;
END
$$;
CREATE FUNCTION pg_temp.search_stats(query text) RETURNS jsonb
LANGUAGE plpgsql AS $$
DECLARE
    plan jsonb;
BEGIN
    EXECUTE 'EXPLAIN (ANALYZE, FORMAT JSON) ' || query INTO plan;
    RETURN plan->0->'pg_mctsopt.mcts';
END
$$;
-- Inspect stable fields rather than costs and timings.
SELECT pg_temp.search_stats('SELECT count(*) FROM t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14
WHERE t1.id=t2.id
  AND t2.id=t3.id
  AND t3.id=t4.id
  AND t4.id=t5.id
  AND t5.id=t6.id
  AND t6.id=t7.id
  AND t7.id=t8.id
  AND t8.id=t9.id
  AND t9.id=t10.id
  AND t10.id=t11.id
  AND t11.id=t12.id
  AND t12.id=t13.id
  AND t13.id=t14.id') AS stats \gset
SELECT (:'stats'::jsonb->>'MCTS Relations')::int AS relations,
       (:'stats'::jsonb->>'MCTS Iterations')::int > 0 AS searched;
SELECT count(*) FROM t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14
WHERE t1.id=t2.id
  AND t2.id=t3.id
  AND t3.id=t4.id
  AND t4.id=t5.id
  AND t5.id=t6.id
  AND t6.id=t7.id
  AND t7.id=t8.id
  AND t8.id=t9.id
  AND t9.id=t10.id
  AND t10.id=t11.id
  AND t11.id=t12.id
  AND t12.id=t13.id
  AND t13.id=t14.id;
-- A smaller query must not inherit statistics from the preceding search.
SELECT pg_temp.search_stats('SELECT count(*) FROM t1,t2,t3,t4,t5,t6,t7,t8
WHERE t1.id=t2.id
  AND t2.id=t3.id
  AND t3.id=t4.id
  AND t4.id=t5.id
  AND t5.id=t6.id
  AND t6.id=t7.id
  AND t7.id=t8.id') IS NULL AS skipped;
SET pg_mctsopt.enabled = false;
SELECT pg_temp.search_stats('SELECT count(*) FROM t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14
WHERE t1.id=t2.id
  AND t2.id=t3.id
  AND t3.id=t4.id
  AND t4.id=t5.id
  AND t5.id=t6.id
  AND t6.id=t7.id
  AND t7.id=t8.id
  AND t8.id=t9.id
  AND t9.id=t10.id
  AND t10.id=t11.id
  AND t11.id=t12.id
  AND t12.id=t13.id
  AND t13.id=t14.id') IS NULL AS disabled;
SELECT count(*) FROM t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14
WHERE t1.id=t2.id
  AND t2.id=t3.id
  AND t3.id=t4.id
  AND t4.id=t5.id
  AND t5.id=t6.id
  AND t6.id=t7.id
  AND t7.id=t8.id
  AND t8.id=t9.id
  AND t9.id=t10.id
  AND t10.id=t11.id
  AND t11.id=t12.id
  AND t12.id=t13.id
  AND t13.id=t14.id;
