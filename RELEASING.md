# Publishing a new release with a Zenodo DOI

1. Copy the contents of this folder into your clone of `iitrky/EdgeLight-GAN_ScientificReports_GitHub`, replacing the old files, then:

```bash
git add -A
git commit -m "Replace code with the implementation that matches the manuscript"
git push
```

2. On GitHub, create a release with the tag `v1.1.0`. Zenodo (switched on for this repository) archives it and issues a new version DOI.
3. On the Zenodo record, use the **concept DOI** ("Cite all versions") in the paper, because it always points to the latest version. If you prefer the version DOI, use the one for `v1.1.0`.
4. If the DOI in the manuscript differs from the one in `README.md` and `CITATION.cff`, update all three.
