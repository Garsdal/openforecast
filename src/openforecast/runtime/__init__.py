"""Execution engine and provider clients (Steps 8 and 9).

The engine validates, plans views and delegates to a ``ProviderClient`` — which
may be in-process or a subprocess in its own isolated uv environment.
"""
