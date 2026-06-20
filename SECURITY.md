# Security Policy

## Supported Versions

This is an early prototype. Security fixes target the default branch.

## Reporting a Vulnerability

Please open a private security advisory on GitHub if the issue involves:

- code execution through rendered HTML,
- unsafe model or file loading,
- credential exposure,
- supply-chain or dependency compromise.

For ordinary bugs, use GitHub Issues.

## Safety Notes

The app renders model text into escaped HTML. Do not remove escaping in
`token_uncertainty/rendering.py` unless replacement tests cover XSS behavior.
