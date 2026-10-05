# FTS projection migration repair plan

Scope: candidate `candidate/ryonakae-v2026.9.24-rebuild`; preserve all pre-existing dirty changes. Integration target is `ryonakae`, the active runtime branch; `main` contains only the official release history. Commit and normal fast-forward push follow successful verification and independent review. Live runtime/DB/gateway changes remain outside this candidate task.

## Root cause established by RED

- `_migrate_misaligned_fts_source()` wraps `_ensure_fts_schema()` (which currently uses `executescript`) in a savepoint; SQLite commits the savepoint away, so an empty store cannot reopen.
- Projection DDL rebuilds from the new `fts_content` view before the bounded column backfill. Historical multimodal rows with NULL `fts_content` therefore rebuild empty and become searchable only after a later optimize.
- Projection pending keys are cleared by the structural helpers before the column backfill is durable, so an interruption can publish an incomplete route.

## Invariants for the repair

1. Raw `messages.content` remains the replay source; `fts_content` and all FTS views/indexes contain text-only projection.
2. Every projection swap publishes a durable pending key first, drops its writer triggers, and clears the key only after structural rebuild **and** bounded backfill/finalization are complete.
3. A pending optional surface is offline: no writer triggers, no route flags, no stale/demotion recovery may re-enable it.
4. Tokenizer/FTS5 unavailability and trigger-drop/restore failures fail closed, with durable state observable from a second connection.
5. Startup stays bounded: no unconditional full backfill; use existing admitted/chunked rebuild paths and resume markers.
6. Empty stores use transactional statement-by-statement DDL; no `executescript` inside savepoints or assumed implicit commits.

## TDD slices

1. Existing reopen boundary: RED receipt already captured; make empty and historical multimodal reopen GREEN.
2. Pending lifecycle: keep projection markers through backfill and resume; add/extend focused regressions for route quarantine and optional surfaces.
3. Recovery/quarantine: tokenizerless trigram/CJK, residual triggers, FTS5-unavailable cleanup, stale/demotion pending guards.
4. Peer/fence: refresh open handles from durable pending state and keep structural migration under admitted write authority; verify sibling writer/observer regressions where deterministic.
5. Bounded finalization: replace full FTS5 `rebuild` with durable row-id chunks. Keep FTS writer triggers detached until publication; capture intervening canonical mutations durably and reconcile updates/deletes against the text actually indexed. Resume after interruption without duplicate insertion. Verify base and available optional indexes with strict integrity checks after publication; absent tokenizers leave optional surfaces fenced. Yield between write transactions so foreground writers can proceed.
6. Delivery: rerun focused and adjacent STATE tests on WAL and DELETE runtimes, obtain independent review, resolve the historical state decisions, and reconcile old `ryonakae` ancestry without force-push. Only the approved upstream wording-test exception is waived.

Verification: approved hermetic runner `python3 /Users/ryo.nakae/.hermes/tmp/pytest-release-hermetic.py --repo <candidate> --label <unique> <testfiles...>`. Record RED/GREEN receipt paths and keep the known unrelated session-recovery `.recover` blocker separate.
