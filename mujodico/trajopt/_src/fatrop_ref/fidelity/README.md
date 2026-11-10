# IP-PAS vs. official FATROP fidelity checks

Compare iteration traces of `IpPasSolver` with CasADi's bundled FATROP 1.1.8
(`pip install casadi`). Run from the repository root with this directory on
`PYTHONPATH`:

    PYTHONPATH=mujodico/trajopt/_src/fatrop_ref/fidelity python .../toy_cmp.py 10
    PYTHONPATH=... python .../pend_cmp.py 40 2.0        # N, |u|max (restoration path)
    PYTHONPATH=... python .../contact_cmp.py PushBox2d 1 150   # task seed iters

Rows marked `!!` differ from FATROP (objective beyond 1e-7 rel. or step type).
`contact_cmp.py` wraps the MJX dynamics/constraints as CasADi callbacks with
exact Jacobians and zero second derivatives (Gauss-Newton = IP-PAS 'cost').
