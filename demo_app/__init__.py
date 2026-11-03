"""
A stand-in for the application under test.

Market Mayor's own prompts, schema and extraction code are private and are not
in this repository. This package is a representative substitute: the same shape
of pipeline — a prompt, a structured-output schema, a transcript builder, an
injectable LLM client, and a verifier that drops any fact whose quote cannot be
found in the message it cites — so the scoring code under `evals/` can be read,
run and tested end to end.

It is written generically and from scratch. No prompt, schema, field name, rule
or rejection-reason string here is taken from the production system, and the
numbers it produces are not Market Mayor's numbers. Point the target adapter at
the real application to get those.
"""
