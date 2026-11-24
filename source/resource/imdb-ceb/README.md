# IMDb CEB Query Resource

Source: `https://github.com/Reminiscent/join-order-benchmark/tree/main/imdb_pg_dataset/ceb-imdb-3k`

Layout:

- `queries/ceb-imdb-3k/`: unchanged upstream CEB IMDb 3k SQL workload.
- `queries/rel9_seed42_200/`: reproducible 200-query subset with at least 9 relations per query.
- `queries/rel9_seed42_200/selected_queries.csv`: manifest with source path, output path, template, and relation count.
- `script/select_rel_queries.py`: sampler used to build the subset.

Obtain the upstream workload from GitHub:

```bash
mkdir -p source/resource/imdb-ceb/{queries,script}
tmpdir="$(mktemp -d)"
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/Reminiscent/join-order-benchmark.git "$tmpdir/repo"
git -C "$tmpdir/repo" sparse-checkout set imdb_pg_dataset/ceb-imdb-3k
cp -a "$tmpdir/repo/imdb_pg_dataset/ceb-imdb-3k" \
  source/resource/imdb-ceb/queries/ceb-imdb-3k
rm -rf "$tmpdir"
```

Regenerate the default subset:

```bash
python3 source/resource/imdb-ceb/script/select_rel_queries.py --force
```

Choose another size or seed:

```bash
python3 source/resource/imdb-ceb/script/select_rel_queries.py \
  --min-relations 9 \
  --limit 500 \
  --seed 123 \
  --output source/resource/imdb-ceb/queries/rel9_seed123_500
```
