"""Unit tests for configuration management.

Tests for all configuration classes to ensure proper validation,
defaults, and environment variable parsing.
"""

import os

import pytest
from pydantic import ValidationError

from pg_mcp.config.settings import (
    CacheConfig,
    DatabaseConfig,
    ObservabilityConfig,
    OpenAIConfig,
    ResilienceConfig,
    SecurityConfig,
    Settings,
    ValidationConfig,
    get_settings,
    reset_settings,
)


class TestDatabaseConfig:
    """Tests for DatabaseConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        # _env_file=None: skip local .env so pure code defaults are tested
        config = DatabaseConfig(_env_file=None)
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.name == "postgres"
        assert config.user == "postgres"
        assert config.min_pool_size == 5
        assert config.max_pool_size == 20

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = DatabaseConfig(
            host="db.example.com",
            port=5433,
            name="mydb",
            user="myuser",
            password="secret",
        )
        assert config.host == "db.example.com"
        assert config.port == 5433
        assert config.name == "mydb"
        assert config.user == "myuser"
        assert config.password == "secret"

    def test_dsn_generation(self) -> None:
        """Test DSN string generation."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            name="testdb",
            user="testuser",
            password="testpass",
        )
        dsn = config.dsn
        assert dsn == "postgresql://testuser:testpass@localhost:5432/testdb"

    def test_safe_dsn_masks_password(self) -> None:
        """Test safe DSN masks password."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            name="testdb",
            user="testuser",
            password="secret123",
        )
        safe_dsn = config.safe_dsn
        assert "secret123" not in safe_dsn
        assert "***" in safe_dsn
        assert "testuser" in safe_dsn

    def test_invalid_port(self) -> None:
        """Test invalid port number is rejected."""
        with pytest.raises(ValidationError):
            DatabaseConfig(port=0)

        with pytest.raises(ValidationError):
            DatabaseConfig(port=99999)

    def test_invalid_pool_size(self) -> None:
        """Test invalid pool size is rejected."""
        with pytest.raises(ValidationError):
            DatabaseConfig(min_pool_size=0)

        with pytest.raises(ValidationError):
            DatabaseConfig(max_pool_size=101)


class TestOpenAIConfig:
    """Tests for OpenAIConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        # _env_file=None: skip local .env so pure code defaults are tested
        config = OpenAIConfig(api_key="sk-test123", _env_file=None)
        assert config.model == "gpt-4o-mini"
        assert config.max_tokens == 2000
        assert config.temperature == 0.0
        assert config.timeout == 30.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = OpenAIConfig(
            api_key="sk-custom",
            model="gpt-4",
            max_tokens=4000,
            temperature=0.7,
            timeout=60.0,
        )
        assert config.model == "gpt-4"
        assert config.max_tokens == 4000
        assert config.temperature == 0.7
        assert config.timeout == 60.0

    def test_empty_api_key_rejected(self) -> None:
        """Test empty API key is rejected."""
        with pytest.raises(ValidationError, match="must not be empty"):
            OpenAIConfig(api_key="")

    def test_whitespace_api_key_rejected(self) -> None:
        """Test whitespace-only API key is rejected."""
        with pytest.raises(ValidationError, match="must not be empty"):
            OpenAIConfig(api_key="   ")

    def test_invalid_api_key_format(self) -> None:
        """Test API key must start with sk-."""
        with pytest.raises(ValidationError, match="must start with 'sk-'"):
            OpenAIConfig(api_key="invalid-key")

    def test_invalid_max_tokens(self) -> None:
        """Test invalid max_tokens is rejected."""
        with pytest.raises(ValidationError):
            OpenAIConfig(api_key="sk-test", max_tokens=50)

        with pytest.raises(ValidationError):
            OpenAIConfig(api_key="sk-test", max_tokens=5000)

    def test_invalid_temperature(self) -> None:
        """Test invalid temperature is rejected."""
        with pytest.raises(ValidationError):
            OpenAIConfig(api_key="sk-test", temperature=-0.1)

        with pytest.raises(ValidationError):
            OpenAIConfig(api_key="sk-test", temperature=2.1)


class TestSecurityConfig:
    """Tests for SecurityConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SecurityConfig()
        assert config.allow_write_operations is False
        assert config.max_rows == 10000
        assert config.max_execution_time == 30.0
        assert "pg_sleep" in config.blocked_functions
        assert "pg_read_file" in config.blocked_functions

    def test_custom_blocked_functions(self) -> None:
        """Test custom blocked functions."""
        config = SecurityConfig(
            blocked_functions=["func1", "func2"],
        )
        assert config.blocked_functions == ["func1", "func2"]

    def test_parse_blocked_functions_from_string(self) -> None:
        """Test parsing blocked functions from comma-separated string."""
        config = SecurityConfig(
            blocked_functions="func1, func2, func3",  # type: ignore
        )
        assert "func1" in config.blocked_functions
        assert "func2" in config.blocked_functions
        assert "func3" in config.blocked_functions

    def test_allow_write_operations(self) -> None:
        """Test enabling write operations."""
        config = SecurityConfig(allow_write_operations=True)
        assert config.allow_write_operations is True

    def test_invalid_max_rows(self) -> None:
        """Test invalid max_rows is rejected."""
        with pytest.raises(ValidationError):
            SecurityConfig(max_rows=0)

        with pytest.raises(ValidationError):
            SecurityConfig(max_rows=100001)

    def test_blocked_tables_and_columns_defaults(self) -> None:
        """Test blocked tables/columns default to empty and EXPLAIN is off."""
        config = SecurityConfig(_env_file=None)
        assert config.blocked_tables == []
        assert config.blocked_columns == []
        assert config.allow_explain is False

    def test_parse_blocked_tables_from_string(self) -> None:
        """Test parsing blocked tables from comma-separated string."""
        config = SecurityConfig(
            blocked_tables="passwords, audit_events, api_keys",  # type: ignore
            _env_file=None,
        )
        assert config.blocked_tables == ["passwords", "audit_events", "api_keys"]

    def test_parse_blocked_columns_from_string(self) -> None:
        """Test parsing blocked columns from comma-separated string."""
        config = SecurityConfig(
            blocked_columns="users.ssn, credit_cards.number",  # type: ignore
            _env_file=None,
        )
        assert config.blocked_columns == ["users.ssn", "credit_cards.number"]

    def test_blocked_lists_ignore_blank_entries(self) -> None:
        """Test blank entries are dropped when parsing blocklists."""
        config = SecurityConfig(
            blocked_tables=" passwords , ,audit_events,",  # type: ignore
            _env_file=None,
        )
        assert config.blocked_tables == ["passwords", "audit_events"]

    def test_allow_explain_flag(self) -> None:
        """Test allow_explain can be enabled."""
        config = SecurityConfig(allow_explain=True, _env_file=None)
        assert config.allow_explain is True


class TestValidationConfig:
    """Tests for ValidationConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ValidationConfig()
        assert config.max_question_length == 10000
        assert config.min_confidence_score == 70

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ValidationConfig(
            max_question_length=5000,
            min_confidence_score=80,
        )
        assert config.max_question_length == 5000
        assert config.min_confidence_score == 80

    def test_invalid_confidence_score(self) -> None:
        """Test invalid confidence score is rejected."""
        with pytest.raises(ValidationError):
            ValidationConfig(min_confidence_score=-1)

        with pytest.raises(ValidationError):
            ValidationConfig(min_confidence_score=101)


class TestCacheConfig:
    """Tests for CacheConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CacheConfig()
        assert config.schema_ttl == 3600
        assert config.max_size == 100
        assert config.enabled is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = CacheConfig(
            schema_ttl=7200,
            max_size=200,
            enabled=False,
        )
        assert config.schema_ttl == 7200
        assert config.max_size == 200
        assert config.enabled is False

    def test_invalid_ttl(self) -> None:
        """Test invalid TTL is rejected."""
        with pytest.raises(ValidationError):
            CacheConfig(schema_ttl=30)

        with pytest.raises(ValidationError):
            CacheConfig(schema_ttl=90000)


class TestResilienceConfig:
    """Tests for ResilienceConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ResilienceConfig()
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.backoff_factor == 2.0
        assert config.circuit_breaker_threshold == 5
        assert config.circuit_breaker_timeout == 60.0
        assert config.query_concurrency_limit == 10
        assert config.llm_concurrency_limit == 5

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ResilienceConfig(
            max_retries=5,
            retry_delay=2.0,
            backoff_factor=3.0,
            query_concurrency_limit=20,
            llm_concurrency_limit=8,
        )
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.backoff_factor == 3.0
        assert config.query_concurrency_limit == 20
        assert config.llm_concurrency_limit == 8

    def test_invalid_values(self) -> None:
        """Test invalid values are rejected."""
        with pytest.raises(ValidationError):
            ResilienceConfig(max_retries=-1)

        with pytest.raises(ValidationError):
            ResilienceConfig(backoff_factor=0.5)

    def test_invalid_concurrency_limits(self) -> None:
        """Test concurrency limits below 1 are rejected."""
        with pytest.raises(ValidationError):
            ResilienceConfig(query_concurrency_limit=0)

        with pytest.raises(ValidationError):
            ResilienceConfig(llm_concurrency_limit=0)


class TestObservabilityConfig:
    """Tests for ObservabilityConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        # _env_file=None: skip local .env so pure code defaults are tested
        config = ObservabilityConfig(_env_file=None)
        # metrics_enabled 在测试环境可能被禁用以避免启动 HTTP 服务器
        # 生产环境应该通过环境变量显式设置
        assert config.metrics_port == 9090
        assert config.log_level == "INFO"
        assert config.log_format == "json"

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ObservabilityConfig(
            metrics_enabled=False,
            metrics_port=8080,
            log_level="DEBUG",
            log_format="text",
        )
        assert config.metrics_enabled is False
        assert config.metrics_port == 8080
        assert config.log_level == "DEBUG"
        assert config.log_format == "text"

    def test_invalid_log_level(self) -> None:
        """Test invalid log level is rejected."""
        with pytest.raises(ValidationError):
            ObservabilityConfig(log_level="INVALID")  # type: ignore

    def test_invalid_log_format(self) -> None:
        """Test invalid log format is rejected."""
        with pytest.raises(ValidationError):
            ObservabilityConfig(log_format="xml")  # type: ignore


class TestSettings:
    """Tests for main Settings class."""

    def test_default_settings(self) -> None:
        """Test default settings initialization."""
        settings = Settings(openai=OpenAIConfig(api_key="sk-test"))
        assert settings.environment == "development"
        assert settings.database is not None
        assert settings.openai is not None
        assert settings.security is not None
        assert settings.validation is not None
        assert settings.cache is not None
        assert settings.resilience is not None
        assert settings.observability is not None

    def test_is_production(self) -> None:
        """Test production environment check."""
        settings = Settings(
            environment="production",
            openai=OpenAIConfig(api_key="sk-test"),
        )
        assert settings.is_production
        assert not settings.is_development

    def test_is_development(self) -> None:
        """Test development environment check."""
        settings = Settings(
            environment="development",
            openai=OpenAIConfig(api_key="sk-test"),
        )
        assert settings.is_development
        assert not settings.is_production

    def test_nested_config_override(self) -> None:
        """Test overriding nested configurations."""
        settings = Settings(
            openai=OpenAIConfig(api_key="sk-test"),
            database=DatabaseConfig(
                host="custom.host",
                port=5433,
            ),
            security=SecurityConfig(
                allow_write_operations=True,
            ),
        )
        assert settings.database.host == "custom.host"
        assert settings.database.port == 5433
        assert settings.security.allow_write_operations is True

    def test_databases_default_empty(self) -> None:
        """Test databases defaults to empty (single-database mode)."""
        settings = Settings(openai=OpenAIConfig(api_key="sk-test"))
        assert settings.databases == []

    def test_parse_databases_from_string(self) -> None:
        """Test comma-separated database names are parsed into a list."""
        settings = Settings(
            databases="blog_small, ecommerce_medium, saas_crm_large",  # type: ignore
            openai=OpenAIConfig(api_key="sk-test"),
        )
        assert settings.databases == ["blog_small", "ecommerce_medium", "saas_crm_large"]

    def test_parse_databases_ignores_blank_entries(self) -> None:
        """Test blank entries are dropped when parsing database names."""
        settings = Settings(
            databases=" blog_small , ,ecommerce_medium,",  # type: ignore
            openai=OpenAIConfig(api_key="sk-test"),
        )
        assert settings.databases == ["blog_small", "ecommerce_medium"]

    def test_get_database_configs_single_database(self) -> None:
        """Test empty databases list falls back to the primary database."""
        settings = Settings(
            openai=OpenAIConfig(api_key="sk-test"),
            database=DatabaseConfig(name="blog_small"),
        )
        configs = settings.get_database_configs()

        assert len(configs) == 1
        assert configs[0].name == "blog_small"

    def test_get_database_configs_multiple_databases(self) -> None:
        """Test each configured database gets its own config sharing credentials."""
        settings = Settings(
            databases=["blog_small", "ecommerce_medium"],  # type: ignore[arg-type]
            openai=OpenAIConfig(api_key="sk-test"),
            database=DatabaseConfig(
                host="db.internal",
                port=5433,
                name="blog_small",
                user="reader",
                password="secret",
            ),
        )
        configs = settings.get_database_configs()

        assert [c.name for c in configs] == ["blog_small", "ecommerce_medium"]
        # All databases share the primary connection settings
        for config in configs:
            assert config.host == "db.internal"
            assert config.port == 5433
            assert config.user == "reader"
            assert config.password == "secret"


class TestSettingsGlobalInstance:
    """Tests for global settings instance management."""

    def teardown_method(self) -> None:
        """Clean up after each test."""
        reset_settings()
        # Clean up environment variables
        for key in list(os.environ.keys()):
            if key.startswith(("DATABASE_", "OPENAI_", "SECURITY_")):
                del os.environ[key]

    def test_get_settings_creates_instance(self) -> None:
        """Test get_settings creates instance."""
        # Set required env var
        os.environ["OPENAI_API_KEY"] = "sk-test123"

        settings = get_settings()
        assert settings is not None
        assert isinstance(settings, Settings)

    def test_get_settings_returns_same_instance(self) -> None:
        """Test get_settings returns singleton."""
        os.environ["OPENAI_API_KEY"] = "sk-test123"

        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_reset_settings(self) -> None:
        """Test reset_settings clears instance."""
        os.environ["OPENAI_API_KEY"] = "sk-test123"

        settings1 = get_settings()
        reset_settings()
        settings2 = get_settings()
        assert settings1 is not settings2

    def test_settings_from_environment(self) -> None:
        """Test loading settings from environment variables."""
        os.environ["OPENAI_API_KEY"] = "sk-env-key"
        os.environ["OPENAI_MODEL"] = "gpt-4"
        os.environ["DATABASE_HOST"] = "env.host.com"
        os.environ["SECURITY_MAX_ROWS"] = "5000"

        reset_settings()
        settings = get_settings()

        # Use get_secret_value() to access SecretStr content
        assert settings.openai.api_key.get_secret_value() == "sk-env-key"
        assert settings.openai.model == "gpt-4"
        assert settings.database.host == "env.host.com"
        assert settings.security.max_rows == 5000
