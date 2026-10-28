# Optional GENRAY regression fixture

The local regression uses the COMPX GENRAY `ec_iter_central_cd` example at upstream commit
`ee443d16e5aaf9bc7227fb3e95581e3ba374df89`.

Expected local files are:

- `genray.dat`
- `equilib.dat`
- `gold-genray.nc`
- `upstream_commit.txt`

The public upstream repository did not expose a clear SPDX license when this fixture was audited.
Therefore these files are not included in a new Git history or source distribution by default.
Obtain them from the upstream GENRAY repository only under terms you are authorized to use, place
them in this directory, and run `python -m pytest tests/test_genray_io.py`.

The tests check the imported frequency, injected/absorbed power, current, ray path and SI conversion.
They skip explicitly when `gold-genray.nc` is absent; no result is fabricated and no network download
occurs during testing.
