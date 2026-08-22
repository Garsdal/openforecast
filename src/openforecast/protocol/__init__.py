"""The provider wire protocol (Step 9).

Protocol version, request/response messages and the error envelope. This is the
innermost layer: it may not import any other OpenForecast subpackage, and it
knows nothing about any specific provider.
"""
