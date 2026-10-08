"""Unit tests for orchestrator resilience and observability wiring.

These tests verify that multi-database routing, rate limiting, metrics
instrumentation, retry backoff, and security config pass-through are actually
integrated into the request processing flow (not just standalone components).
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pg_mcp.config.settings import ResilienceConfig, SecurityConfig, ValidationConfig
from pg_mcp.models.errors import DatabaseError, SecurityViolationError
from pg_mcp.models.query import QueryRequest, ReturnType
from pg_mcp.models.schema import ColumnInfo, DatabaseSchema, TableInfo
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.orchestrator import QueryOrchestrator
from pg_mcp.services.sql_validator import SQLValidator


def make_schema(database_name: str = "test_db") -> DatabaseSchema:
    """Create a minimal single-table schema for the given database."""
    return DatabaseSchema(
        database_name=database_name,
        tables=[
            TableInfo(
                schema_name="public",
                table_name="users",
                columns=[
                    ColumnInfo(
                        name="id",
                        data_type="integer",
                        is_nullable=False,
                        is_primary_key=True,
                    ),
                    ColumnInfo(name="name", data_type="varchar(255)", is_nullable=False),
                ],
            )
        ],
        version="15.0",
    )


def make_generator(return_value: tuple[str, int | None]) -> AsyncMock:
    """Create an AsyncMock SQL generator returning the given (sql, tokens)."""
    generator = AsyncMock()
    generator.generate.return_value = return_value
    return generator


def make_passing_validator() -> MagicMock:
    """Create a validator mock that accepts every SQL statement."""
    validator = MagicMock()
    validator.validate_or_raise.return_value = None
    return validator


def make_rejecting_validator() -> MagicMock:
    """Create a validator mock that rejects every SQL statement."""
    validator = MagicMock()
    validator.validate_or_raise.side_effect = SecurityViolationError("DELETE not allowed")
    return validator


def make_request(
    database: str = "test_db", return_type: ReturnType = ReturnType.RESULT
) -> QueryRequest:
    """Create a query request for the given database and return type."""
    return QueryRequest(
        question="Get all users",
        database=database,
        return_type=return_type,
    )


def build_orchestrator(
    *,
    executors: dict[str, Any] | None = None,
    pools: dict[str, Any] | None = None,
    rate_limiter: Any = None,
    metrics: Any = None,
    resilience_config: ResilienceConfig | None = None,
    generator: Any = None,
    validator: Any = None,
    schema: DatabaseSchema | None = None,
) -> QueryOrchestrator:
    """Build an orchestrator wired with the given test doubles.

    Defaults model a single-database happy path: SQL generation succeeds on
    the first attempt with 100 tokens, validation passes, and execution
    returns one row.
    """
    cache = MagicMock()
    cache.get.return_value = schema or make_schema()
    executor = AsyncMock()
    executor.execute.return_value = ([{"id": 1, "name": "Alice"}], 1)
    return QueryOrchestrator(
        sql_generator=generator or make_generator(("SELECT id, name FROM users;", 100)),
        sql_validator=validator or make_passing_validator(),
        sql_executors=executors if executors is not None else {"test_db": executor},
        result_validator=AsyncMock(),
        schema_cache=cache,
        pools=pools if pools is not None else {"test_db": MagicMock()},
        resilience_config=resilience_config or ResilienceConfig(),
        validation_config=ValidationConfig(enabled=False),
        rate_limiter=rate_limiter,
        metrics=metrics,
    )


class _TimedOutSlot:
    """Async context manager simulating a rate limiter acquire timeout."""

    async def __aenter__(self) -> None:
        raise TimeoutError("Rate limiter timeout exceeded")

    async def __aexit__(self, *exc_info: object) -> None:
        return None


class _AcquiredSlot:
    """Async context manager simulating successful slot acquisition."""

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc_info: object) -> None:
        return None


class LLMTimeoutLimiter:
    """Fake MultiRateLimiter whose LLM slot always times out."""

    def for_llm(self, *, timeout: float | None = None) -> _TimedOutSlot:
        return _TimedOutSlot()

    def for_queries(self, *, timeout: float | None = None) -> _AcquiredSlot:
        return _AcquiredSlot()


class QueryTimeoutLimiter:
    """Fake MultiRateLimiter whose query slot always times out."""

    def for_llm(self, *, timeout: float | None = None) -> _AcquiredSlot:
        return _AcquiredSlot()

    def for_queries(self, *, timeout: float | None = None) -> _TimedOutSlot:
        return _TimedOutSlot()


class TestGetExecutor:
    """Test per-database executor lookup."""

    def test_returns_executor_for_configured_database(self) -> None:
        """Test that the executor bound to the database name is returned."""
        executor = MagicMock()
        orchestrator = build_orchestrator(executors={"db1": executor})

        assert orchestrator._get_executor("db1") is executor

    def test_missing_executor_raises_database_error(self) -> None:
        """Test that a missing executor raises DatabaseError with details."""
        orchestrator = build_orchestrator(executors={"db1": MagicMock()})

        with pytest.raises(DatabaseError) as exc_info:
            orchestrator._get_executor("db2")

        assert exc_info.value.code.value == "database_error"
        assert exc_info.value.details["database"] == "db2"
        assert exc_info.value.details["configured_databases"] == ["db1"]


class TestMultiDatabaseRouting:
    """Test that queries route to the executor of the requested database."""

    @pytest.mark.asyncio
    async def test_routes_to_requested_database_executor(self) -> None:
        """Test that only the requested database's executor runs the SQL."""
        exec_db1, exec_db2 = AsyncMock(), AsyncMock()
        exec_db1.execute.return_value = ([{"id": 1}], 1)
        exec_db2.execute.return_value = ([{"id": 2, "name": "Bob"}], 1)

        orchestrator = build_orchestrator(
            executors={"db1": exec_db1, "db2": exec_db2},
            pools={"db1": MagicMock(), "db2": MagicMock()},
        )

        response = await orchestrator.execute_query(make_request(database="db2"))

        assert response.success is True
        assert response.data is not None
        exec_db2.execute.assert_called_once()
        exec_db1.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_executor_returns_database_error(self) -> None:
        """Test that executing against a database without an executor fails."""
        orchestrator = build_orchestrator(
            executors={},
            pools={"test_db": MagicMock()},
        )

        response = await orchestrator.execute_query(make_request(database="test_db"))

        assert response.success is False
        assert response.error is not None
        assert response.error.code == "database_error"
        assert response.error.details["configured_databases"] == []


class TestRateLimitingIntegration:
    """Test rate limiter integration into the request flow."""

    @pytest.mark.asyncio
    async def test_llm_slot_timeout_returns_rate_limit_error(self) -> None:
        """Test that an LLM limiter timeout surfaces as rate_limit_exceeded."""
        generator = make_generator(("SELECT 1;", 10))
        orchestrator = build_orchestrator(
            generator=generator,
            rate_limiter=LLMTimeoutLimiter(),
        )

        response = await orchestrator.execute_query(make_request(return_type=ReturnType.SQL))

        assert response.success is False
        assert response.error is not None
        assert response.error.code == "rate_limit_exceeded"
        assert response.error.details == {"resource": "llm"}
        generator.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_slot_timeout_returns_rate_limit_error(self) -> None:
        """Test that a query limiter timeout surfaces as rate_limit_exceeded."""
        executor = AsyncMock()
        orchestrator = build_orchestrator(
            executors={"test_db": executor},
            rate_limiter=QueryTimeoutLimiter(),
        )

        response = await orchestrator.execute_query(make_request())

        assert response.success is False
        assert response.error is not None
        assert response.error.code == "rate_limit_exceeded"
        assert response.error.details == {"resource": "queries"}
        executor.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_real_rate_limiter_wraps_llm_and_query_calls(self) -> None:
        """Test the real MultiRateLimiter wraps generation and execution."""
        limiter = MultiRateLimiter(query_limit=3, llm_limit=2)
        orchestrator = build_orchestrator(rate_limiter=limiter)

        response = await orchestrator.execute_query(make_request())

        assert response.success is True
        # Let pending release tasks run before inspecting counters
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert limiter.llm_limiter.active_count == 0
        assert limiter.query_limiter.active_count == 0
        assert limiter.llm_limiter.available == 2
        assert limiter.query_limiter.available == 3
        stats = limiter.get_all_stats()
        assert stats["llm"]["total_requests"] == 1
        assert stats["queries"]["total_requests"] == 1
        assert stats["llm"]["total_rejections"] == 0


class TestMetricsInstrumentation:
    """Test metrics emission across the request flow."""

    @pytest.mark.asyncio
    async def test_success_records_full_metric_set(self) -> None:
        """Test all metrics recorded on a successful executed query."""
        metrics = MagicMock()
        orchestrator = build_orchestrator(metrics=metrics)

        response = await orchestrator.execute_query(make_request())

        assert response.success is True
        metrics.increment_query_request.assert_called_once_with(
            status="success", database="test_db"
        )
        metrics.increment_llm_call.assert_called_once_with("generation")
        metrics.observe_llm_latency.assert_called_once()
        assert metrics.observe_llm_latency.call_args.args[0] == "generation"
        assert metrics.observe_llm_latency.call_args.args[1] >= 0
        metrics.increment_llm_tokens.assert_called_once_with("generation", 100)
        metrics.observe_db_query_duration.assert_called_once()
        assert metrics.observe_db_query_duration.call_args.args[0] >= 0
        metrics.query_duration.observe.assert_called_once()
        metrics.increment_sql_rejected.assert_not_called()

    @pytest.mark.asyncio
    async def test_sql_only_skips_db_duration_metric(self) -> None:
        """Test SQL-only responses skip the DB duration metric."""
        metrics = MagicMock()
        orchestrator = build_orchestrator(metrics=metrics)

        response = await orchestrator.execute_query(make_request(return_type=ReturnType.SQL))

        assert response.success is True
        metrics.increment_query_request.assert_called_once_with(
            status="success", database="test_db"
        )
        metrics.observe_db_query_duration.assert_not_called()
        metrics.query_duration.observe.assert_called_once()

    @pytest.mark.asyncio
    async def test_execution_error_records_error_status(self) -> None:
        """Test failed requests record the error status metric."""
        metrics = MagicMock()
        executor = AsyncMock()
        executor.execute.side_effect = DatabaseError("connection lost")
        orchestrator = build_orchestrator(
            executors={"test_db": executor},
            metrics=metrics,
        )

        response = await orchestrator.execute_query(make_request())

        assert response.success is False
        metrics.increment_query_request.assert_called_once_with(status="error", database="test_db")
        metrics.query_duration.observe.assert_called_once()

    @pytest.mark.asyncio
    async def test_sql_rejection_recorded_per_attempt(self) -> None:
        """Test each rejected validation attempt records a rejection metric."""
        metrics = MagicMock()
        orchestrator = build_orchestrator(
            generator=make_generator(("DELETE FROM users;", 10)),
            validator=make_rejecting_validator(),
            metrics=metrics,
            resilience_config=ResilienceConfig(max_retries=1, retry_delay=0.1, backoff_factor=1.0),
        )

        response = await orchestrator.execute_query(make_request(return_type=ReturnType.SQL))

        assert response.success is False
        assert response.error is not None
        assert response.error.code == "security_violation"
        # Rejected once per attempt (initial + 1 retry)
        assert metrics.increment_sql_rejected.call_count == 2
        metrics.increment_sql_rejected.assert_called_with("security_violation")
        metrics.increment_query_request.assert_called_once_with(status="error", database="test_db")

    @pytest.mark.asyncio
    async def test_missing_tokens_skips_token_metric(self) -> None:
        """Test token metrics are skipped when usage metadata is absent."""
        metrics = MagicMock()
        orchestrator = build_orchestrator(
            generator=make_generator(("SELECT 1;", None)),
            metrics=metrics,
        )

        response = await orchestrator.execute_query(make_request(return_type=ReturnType.SQL))

        assert response.success is True
        assert response.tokens_used is None
        metrics.increment_llm_call.assert_called_once_with("generation")
        metrics.increment_llm_tokens.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_before_resolution_records_unknown_database(self) -> None:
        """Test errors before database resolution fall back to 'unknown'."""
        metrics = MagicMock()
        orchestrator = build_orchestrator(metrics=metrics)

        response = await orchestrator.execute_query(make_request(database="nonexistent"))

        assert response.success is False
        metrics.increment_query_request.assert_called_once_with(status="error", database="unknown")


class TestRetryBackoff:
    """Test exponential backoff between retry attempts."""

    @pytest.mark.asyncio
    async def test_backoff_delay_grows_exponentially(self) -> None:
        """Test sleep uses retry_delay * backoff_factor ** attempt."""
        orchestrator = build_orchestrator(
            generator=make_generator(("DELETE FROM users;", 10)),
            validator=make_rejecting_validator(),
            resilience_config=ResilienceConfig(max_retries=2, retry_delay=0.1, backoff_factor=2.0),
        )

        with (
            patch(
                "pg_mcp.services.orchestrator.asyncio.sleep", new_callable=AsyncMock
            ) as mock_sleep,
            pytest.raises(SecurityViolationError),
        ):
            await orchestrator._generate_sql_with_retry(
                question="Delete all users",
                schema=make_schema(),
                request_id="test-123",
            )

        # Two sleeps between three attempts: 0.1 * 2**0, then 0.1 * 2**1
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(0.1)
        mock_sleep.assert_any_call(0.2)

    @pytest.mark.asyncio
    async def test_no_backoff_on_first_attempt_success(self) -> None:
        """Test no sleep occurs when validation passes on the first attempt."""
        orchestrator = build_orchestrator()

        with patch(
            "pg_mcp.services.orchestrator.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep:
            sql, validation, tokens = await orchestrator._generate_sql_with_retry(
                question="Get all users",
                schema=make_schema(),
                request_id="test-123",
            )

        assert sql == "SELECT id, name FROM users;"
        assert validation.is_valid is True
        assert tokens == 100
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_tokens_accumulate_across_retries(self) -> None:
        """Test token usage is summed across all LLM attempts."""
        generator = AsyncMock()
        generator.generate.side_effect = [
            ("DELETE FROM users;", 50),
            ("SELECT id FROM users;", 60),
        ]
        validator = MagicMock()
        validator.validate_or_raise.side_effect = [
            SecurityViolationError("DELETE not allowed"),
            None,
        ]
        orchestrator = build_orchestrator(
            generator=generator,
            validator=validator,
            resilience_config=ResilienceConfig(max_retries=1, retry_delay=0.1, backoff_factor=1.0),
        )

        sql, _validation, tokens = await orchestrator._generate_sql_with_retry(
            question="Get all users",
            schema=make_schema(),
            request_id="test-123",
        )

        assert sql == "SELECT id FROM users;"
        assert tokens == 110


class TestSecurityConfigWiring:
    """Test that SecurityConfig fields reach the SQL validator (server wiring)."""

    def test_blocked_tables_columns_and_explain_enforced(self) -> None:
        """Test config-driven blocklists and EXPLAIN policy are enforced."""
        # Same wiring as server.py lifespan: config fields passed explicitly
        config = SecurityConfig(
            blocked_tables=["passwords", "audit_events"],
            blocked_columns=["users.ssn"],
            allow_explain=False,
            _env_file=None,
        )
        validator = SQLValidator(
            config=config,
            blocked_tables=config.blocked_tables,
            blocked_columns=config.blocked_columns,
            allow_explain=config.allow_explain,
        )

        with pytest.raises(SecurityViolationError):
            validator.validate_or_raise("SELECT * FROM passwords")
        with pytest.raises(SecurityViolationError):
            # Qualified blocklist entries require the table.column form
            validator.validate_or_raise("SELECT users.ssn FROM users")
        with pytest.raises(SecurityViolationError):
            validator.validate_or_raise("EXPLAIN SELECT 1")
        # Non-sensitive queries are unaffected
        validator.validate_or_raise("SELECT id, name FROM users")

    def test_allow_explain_enabled_permits_explain(self) -> None:
        """Test EXPLAIN statements pass when the policy allows them."""
        config = SecurityConfig(allow_explain=True, _env_file=None)
        validator = SQLValidator(
            config=config,
            blocked_tables=config.blocked_tables,
            blocked_columns=config.blocked_columns,
            allow_explain=config.allow_explain,
        )

        validator.validate_or_raise("EXPLAIN SELECT 1")
