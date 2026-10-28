# Optional OpenADAS ADF11 hydrogen data

The four `.dat` files referenced by `manifest.json` are intentionally not
redistributed in this repository or its Python packages.

OPEN-ADAS states that downloaded files are for personal use and must not be
redistributed with modelling or analysis code without prior written
permission. Review the current terms before downloading:

- https://open.adas.ac.uk/terms-and-conditions

For authorized local scientific use, download each file from the URL in
`manifest.json`, place it in this directory, and verify its SHA-256 against
the manifest. The following command then validates and reads the files:

```powershell
python -m ecstartup inspect-adas data/openadas/adf11_h_96/manifest.json `
  --Te-eV 20 --ne-m3 1e19
```

Tests that require these files skip explicitly when they are absent.
