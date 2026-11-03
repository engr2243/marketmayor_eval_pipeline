"""
Adapters onto the real production call sites.

A target renders a case into exactly what production would send, calls the
production function with an injected client, and hands back the raw output
alongside what the production verifier made of it. No target reimplements a
prompt, a schema or a validation rule.
"""
