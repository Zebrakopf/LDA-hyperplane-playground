"""Streamlit UI package.

    controls.py  session state, config assembly, cached computation
    plots.py     interactive Plotly figures

The entrypoint is `app.py` at the repository ROOT, not here: that is Streamlit
convention, and it keeps this package importable by tests without launching a
server. Static matplotlib figures for headless batch runs live in
`scripts/figures.py` instead, so a cluster job never imports a UI stack
(docs/decisions.md D7).

Nothing in this package computes anything scientific. Numbers come from `core/`.
"""
