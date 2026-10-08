"""Configuration management for PostgreSQL MCP Server.

This module defines all configuration settings using Pydantic for validation
and type safety. Configuration is loaded from environment variables with
sensible defaults.
"""

from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """PostgreSQL database connection configuration."""

    model_config = SettingsConfigDict(env_prefix="DATABASE_", env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    name: str = Field(default="postgres", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: str = Field(default="", description="Database password")

    # Connection pool settings
    min_pool_size: int = Field(default=5, ge=1, le=100, description="Minimum pool size")
    max_pool_size: int = Field(default=20, ge=1, le=100, description="Maximum pool size")
    pool_timeout: float = Field(
        default=30.0, ge=1.0, le=300.0, description="Pool acquire timeout in seconds"
    )
    command_timeout: float = Field(
        default=30.0, ge=1.0, le=300.0, description="Command execution timeout in seconds"
    )

    @property
    def dsn(self) -> str:
        """Build PostgreSQL DSN connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def safe_dsn(self) -> str:
        """Build DSN with masked password for logging."""
        return f"postgresql://{self.user}:***@{self.host}:{self.port}/{self.name}"


class OpenAIConfig(BaseSettings):
    """OpenAI API configuration."""

    model_config = SettingsConfigDict(env_prefix="OPENAI_", env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    api_key: SecretStr = Field(default=SecretStr(""), description="OpenAI API key")
    base_url: str | None = Field(
        default=None,
        description="OpenAI-compatible API base URL (e.g. http://host:port/v1). None uses the official OpenAI endpoint.",
    )
    model: str = Field(default="gpt-4o-mini", description="Model to use for SQL generation")
    max_tokens: int = Field(default=2000, ge=100, le=4096, description="Maximum tokens in response")
    temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, description="Temperature for response randomness"
    )
    timeout: float = Field(
        default=30.0, ge=5.0, le=120.0, description="API request timeout in seconds"
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: SecretStr) -> SecretStr:
        """Validate API key is not empty and has correct format."""
        api_key_str = v.get_secret_value()
        if not api_key_str or not api_key_str.strip():
            raise ValueError("OpenAI API key must not be empty")
        if not api_key_str.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v


class SecurityConfig(BaseSettings):
    """Security and access control configuration."""

    model_config = SettingsConfigDict(env_prefix="SECURITY_", env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    allow_write_operations: bool = Field(
        default=False, description="Allow write operations (INSERT, UPDATE, DELETE)"
    )
    blocked_functions: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "pg_sleep",
            "pg_read_file",
            "pg_write_file",
            "lo_import",
            "lo_export",
        ],
        description="List of blocked PostgreSQL functions",
    )
    blocked_tables: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description="List of table names to block access to (comma-separated in env)",
    )
    blocked_columns: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description="List of column names to block access to (comma-separated in env)",
    )
    allow_explain: bool = Field(
        default=False, description="Whether to allow EXPLAIN statements"
    )
    max_rows: int = Field(default=10000, ge=1, le=100000, description="Maximum rows to return")
    max_execution_time: float = Field(
        default=30.0, ge=1.0, le=300.0, description="Maximum query execution time in seconds"
    )
    readonly_role: str | None = Field(
        default=None, description="PostgreSQL role to switch to for read-only access"
    )
    safe_search_path: str = Field(
        default="public", description="Safe search_path to set during query execution"
    )

    @field_validator("blocked_functions", mode="before")
    @classmethod
    def parse_blocked_functions(cls, v: str | list[str]) -> list[str]:
        """Parse comma-separated string or list."""
        if isinstance(v, str):
            return [f.strip() for f in v.split(",") if f.strip()]
        return v

    @field_validator("blocked_tables", "blocked_columns", mode="before")
    @classmethod
    def parse_name_lists(cls, v: str | list[str]) -> list[str]:
        """Parse comma-separated string or list for table/column blocklists."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v


class ValidationConfig(BaseSettings):
    """Query validation configuration."""

    model_config = SettingsConfigDict(env_prefix="VALIDATION_", env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    max_question_length: int = Field(
        default=10000, ge=1, le=50000, description="Maximum question length in characters"
    )
    min_confidence_score: int = Field(
        default=70, ge=0, le=100, description="Minimum confidence score (0-100)"
    )

    # Result validation settings
    enabled: bool = Field(default=True, description="Enable result validation using LLM")
    sample_rows: int = Field(
        default=5, ge=1, le=100, description="Number of sample rows to include in validation"
    )
    timeout_seconds: float = Field(
        default=10.0, ge=1.0, le=60.0, description="Result validation timeout in seconds"
    )
    confidence_threshold: int = Field(
        default=70, ge=0, le=100, description="Minimum confidence for acceptable results"
    )


class CacheConfig(BaseSettings):
    """Schema cache configuration."""

    model_config = SettingsConfigDict(env_prefix="CACHE_", env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    schema_ttl: int = Field(
        default=3600, ge=60, le=86400, description="Schema cache TTL in seconds"
    )
    max_size: int = Field(default=100, ge=1, le=1000, description="Maximum cache entries")
    enabled: bool = Field(default=True, description="Enable schema caching")


class ResilienceConfig(BaseSettings):
    """Resilience and fault tolerance configuration."""

    model_config = SettingsConfigDict(env_prefix="RESILIENCE_", env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    retry_delay: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Initial retry delay in seconds"
    )
    backoff_factor: float = Field(
        default=2.0, ge=1.0, le=10.0, description="Exponential backoff factor"
    )
    circuit_breaker_threshold: int = Field(
        default=5, ge=1, le=100, description="Failures before circuit opens"
    )
    circuit_breaker_timeout: float = Field(
        default=60.0, ge=10.0, le=300.0, description="Circuit breaker timeout in seconds"
    )
    query_concurrency_limit: int = Field(
        default=10, ge=1, le=100, description="Max concurrent database queries"
    )
    llm_concurrency_limit: int = Field(
        default=5, ge=1, le=100, description="Max concurrent LLM API calls"
    )


class ObservabilityConfig(BaseSettings):
    """Observability and monitoring configuration."""

    model_config = SettingsConfigDict(env_prefix="OBSERVABILITY_", env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    metrics_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(
        default=9090, ge=1024, le=65535, description="Metrics HTTP server port"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    log_format: Literal["json", "text"] = Field(default="json", description="Log format")


class Settings(BaseSettings):
    """Main application settings aggregating all config sections."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    environment: Literal["development", "staging", "production"] = Field(
        default="development", description="Application environment"
    )

    # Multi-database support: comma-separated list of database names on the
    # same host/port/user (each gets its own pool). The first entry is the
    # primary/default database. When unset or a single name, behaves exactly
    # like the single-database configuration.
    databases: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description="Comma-separated list of database names to enable (empty = DATABASE_NAME only)",
    )

    # Nested configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    @field_validator("databases", mode="before")
    @classmethod
    def parse_databases(cls, v: str | list[str]) -> list[str]:
        """Parse comma-separated string or list of database names."""
        if isinstance(v, str):
            return [name.strip() for name in v.split(",") if name.strip()]
        return v

    def get_database_configs(self) -> list[DatabaseConfig]:
        """Build DatabaseConfig for every configured database.

        All databases share the host/port/credentials/pool settings from the
        primary DATABASE_* config; only the name differs per database.

        Returns:
            list[DatabaseConfig]: One config per database, primary first.
        """
        names = [n for n in self.databases if n] or [self.database.name]
        return [
            self.database.model_copy(update={"name": name})
            for name in names
        ]

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create global settings instance.

    Returns:
        Settings: The global settings instance.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset global settings instance. Useful for testing."""
    global _settings
    _settings = None
