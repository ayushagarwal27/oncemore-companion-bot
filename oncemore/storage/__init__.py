"""Storage layer.

Postgres is the system of record. Redis is a cache and a checkpoint store:
if it is empty the system is slower and more expensive, never wrong. That
property is the justification for running two datastores at all.
"""
