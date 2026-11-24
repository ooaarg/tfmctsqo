# Model checkpoint

`supervised.pt` is split into parts smaller than 8 MB so that no single file is
large.

Reassemble with:

```bash
cat supervised.pt.gz.part* | gunzip > supervised.pt
```
