"""Environment configuration, paths and Azure credential helpers."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env", override=False)

AGENT_DIR = REPO_ROOT / "agent"
# Foundry workspace standard: the azd service's project folder (agent/) is the agent root.
FOUNDRY_DIR = AGENT_DIR / ".foundry"
DATASETS_DIR = FOUNDRY_DIR / "datasets"
EVALUATORS_DIR = FOUNDRY_DIR / "evaluators"
RESULTS_DIR = FOUNDRY_DIR / "results"

# The Evaluations catalog APIs are preview features and need an explicit opt-in header.
EVALUATIONS_PREVIEW_HEADERS = {"Foundry-Features": "Evaluations=V1Preview"}


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _path(name: str, default: Path) -> Path:
    value = _env(name)
    path = Path(value) if value else default
    return path if path.is_absolute() else REPO_ROOT / path


@dataclass(frozen=True)
class Settings:
    project_endpoint: str | None
    subscription_id: str | None
    tenant_id: str | None
    credential_mode: str
    agent_name: str
    judge_deployment: str
    search_endpoint: str | None
    search_user_index: str
    search_group_index: str
    search_api_key: str | None
    search_api_version: str
    corpus_root: Path
    captures_root: Path

    def require(self, *names: str) -> None:
        missing = [n for n in names if not getattr(self, n)]
        if missing:
            env_names = ", ".join(_ENV_NAMES.get(n, n.upper()) for n in missing)
            raise SystemExit(f"Missing configuration: {env_names}. Copy .env.example to .env and fill it in.")


_ENV_NAMES = {
    "project_endpoint": "AZURE_AI_PROJECT_ENDPOINT",
    "subscription_id": "AZURE_SUBSCRIPTION_ID",
    "search_endpoint": "AZURE_SEARCH_ENDPOINT",
}


def load_settings() -> Settings:
    return Settings(
        project_endpoint=_env("AZURE_AI_PROJECT_ENDPOINT"),
        subscription_id=_env("AZURE_SUBSCRIPTION_ID"),
        tenant_id=_env("AZURE_TENANT_ID"),
        credential_mode=(_env("AZURE_CREDENTIAL", "cli") or "cli").lower(),
        agent_name=_env("SIRE_AGENT_NAME", "sire-voice-agent") or "sire-voice-agent",
        judge_deployment=_env("FOUNDRY_JUDGE_DEPLOYMENT", "gpt-5.4-mini") or "gpt-5.4-mini",
        search_endpoint=_env("AZURE_SEARCH_ENDPOINT"),
        search_user_index=_env("AZURE_SEARCH_USER_INDEX", "sire-gamma-users") or "sire-gamma-users",
        search_group_index=_env("AZURE_SEARCH_GROUP_INDEX", "sire-gamma-groups") or "sire-gamma-groups",
        search_api_key=_env("AZURE_SEARCH_API_KEY"),
        search_api_version=_env("AZURE_SEARCH_API_VERSION", "2024-07-01") or "2024-07-01",
        corpus_root=_path("SIRE_CORPUS_ROOT", REPO_ROOT / "test_data/extracted/tenant-gamma-regression-2026-11-11"),
        captures_root=_path("SIRE_CAPTURES_ROOT", REPO_ROOT / "test_data/extracted/tenant-gamma-regression-2026-11-11-captures"),
    )


def get_credential(settings: Settings | None = None):
    """Synchronous credential.

    ``AZURE_CREDENTIAL=cli`` (default) uses the Azure CLI scoped to ``AZURE_SUBSCRIPTION_ID``,
    which picks the right tenant even when the CLI's default subscription is elsewhere.
    ``AZURE_CREDENTIAL=default`` uses DefaultAzureCredential (managed identity, workload identity, ...).
    """
    settings = settings or load_settings()
    if settings.credential_mode == "cli":
        from azure.identity import AzureCliCredential

        kwargs = {"process_timeout": 60}
        if settings.subscription_id:
            kwargs["subscription"] = settings.subscription_id
        return AzureCliCredential(**kwargs)
    from azure.identity import DefaultAzureCredential

    return DefaultAzureCredential()


def get_async_credential(settings: Settings | None = None):
    settings = settings or load_settings()
    if settings.credential_mode == "cli":
        from azure.identity.aio import AzureCliCredential

        kwargs = {"process_timeout": 60}
        if settings.subscription_id:
            kwargs["subscription"] = settings.subscription_id
        return AzureCliCredential(**kwargs)
    from azure.identity.aio import DefaultAzureCredential

    return DefaultAzureCredential()


def project_client(settings: Settings | None = None):
    from azure.ai.projects import AIProjectClient

    settings = settings or load_settings()
    settings.require("project_endpoint")
    return AIProjectClient(endpoint=settings.project_endpoint, credential=get_credential(settings), allow_preview=True)


def long_path(path: Path | str) -> str:
    """Return a path usable beyond Windows' 260-character MAX_PATH limit."""
    p = os.path.abspath(str(path))
    if sys.platform == "win32" and not p.startswith("\\\\?\\"):
        return "\\\\?\\" + p
    return p
