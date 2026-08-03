"""OAuth token storage and the unrestricted-limits escape hatch."""

from __future__ import annotations

import json
import os
import stat

import pytest

from osiris.config import RiskLimits
from osiris.mcp.session import CALLBACK_PORT, FileTokenStorage, _client_metadata


class TestTokenStorage:
    @pytest.fixture
    def storage(self, tmp_path):
        return FileTokenStorage(path=tmp_path / "mcp-auth.json")

    async def test_round_trips_tokens(self, storage):
        from mcp.shared.auth import OAuthToken

        await storage.set_tokens(
            OAuthToken(access_token="secret", token_type="Bearer", expires_in=3600)
        )
        loaded = await storage.get_tokens()

        assert loaded is not None
        assert loaded.access_token == "secret"

    async def test_absent_file_yields_no_tokens(self, storage):
        assert await storage.get_tokens() is None
        assert not storage.has_tokens

    async def test_tokens_are_written_user_only(self, storage):
        """A token is an account credential.

        World-readable on a shared machine is equivalent to a leaked password.
        """
        from mcp.shared.auth import OAuthToken

        await storage.set_tokens(OAuthToken(access_token="s", token_type="Bearer"))
        mode = stat.S_IMODE(os.stat(storage.path).st_mode)

        assert mode == 0o600

    async def test_corrupt_file_degrades_instead_of_crashing(self, storage):
        """Re-authenticating is recoverable; a crash on startup is not."""
        storage.path.parent.mkdir(parents=True, exist_ok=True)
        storage.path.write_text("{ not json")

        assert await storage.get_tokens() is None

    async def test_invalid_token_shape_is_rejected(self, storage):
        storage.path.parent.mkdir(parents=True, exist_ok=True)
        storage.path.write_text(json.dumps({"tokens": {"unexpected": True}}))

        assert await storage.get_tokens() is None

    async def test_client_info_round_trips_alongside_tokens(self, storage):
        from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

        await storage.set_tokens(OAuthToken(access_token="a", token_type="Bearer"))
        await storage.set_client_info(
            OAuthClientInformationFull(
                client_id="cid",
                redirect_uris=[f"http://127.0.0.1:{CALLBACK_PORT}/callback"],
            )
        )

        # Writing one must not clobber the other.
        assert (await storage.get_tokens()).access_token == "a"
        assert (await storage.get_client_info()).client_id == "cid"

    async def test_clear_forces_reauthentication(self, storage):
        from mcp.shared.auth import OAuthToken

        await storage.set_tokens(OAuthToken(access_token="a", token_type="Bearer"))
        storage.clear()

        assert not storage.has_tokens

    def test_tokens_live_outside_the_repository(self):
        """A credential inside the working tree eventually gets committed."""
        from pathlib import Path

        from osiris.config import REPO_ROOT

        default = FileTokenStorage().path
        assert REPO_ROOT not in Path(default).parents


class TestClientMetadata:
    def test_uses_a_loopback_redirect(self):
        meta = _client_metadata()

        assert str(meta.redirect_uris[0]).startswith("http://127.0.0.1:")

    def test_is_a_public_client_relying_on_pkce(self):
        """No client secret exists to protect in a desktop app."""
        meta = _client_metadata()

        assert meta.token_endpoint_auth_method == "none"

    def test_requests_refresh_so_reauth_is_not_daily(self):
        assert "refresh_token" in _client_metadata().grant_types


class TestUnrestrictedLimits:
    def test_defaults_remain_conservative(self):
        """Opting out must be explicit; the default cannot be unbounded."""
        limits = RiskLimits()

        assert limits.max_trade_notional_pct == 0.02
        assert limits.max_symbol_weight == 0.10

    def test_unrestricted_removes_position_and_loss_caps(self):
        limits = RiskLimits.unrestricted()

        assert limits.max_trade_notional_pct == 1.0
        assert limits.max_symbol_weight == 1.0
        assert limits.daily_loss_halt_pct == 1.0

    def test_unrestricted_still_satisfies_internal_coherence(self):
        """The escape hatch must not produce a self-deadlocking config."""
        limits = RiskLimits.unrestricted()

        assert limits.max_symbol_weight * limits.target_position_count >= 1.0
        assert limits.daily_loss_halt_pct <= limits.max_drawdown_halt_pct

    def test_a_single_name_may_become_the_whole_account(self):
        """Documents the actual consequence rather than implying safety."""
        assert RiskLimits.unrestricted().max_symbol_weight >= 1.0

    def test_incoherent_limits_are_still_rejected_unrestricted(self):
        """Configurability is not the same as accepting nonsense.

        A book of 20 names each capped at 1% can never be fully invested, so the
        kernel would demand full investment while forbidding every order.
        """
        with pytest.raises(ValueError, match="never be fully invested"):
            RiskLimits(max_symbol_weight=0.01, target_position_count=20)


class TestDotenvLoading:
    """Regression: `.env` was invisible to plain `os.environ` lookups.

    Pydantic's `env_file` covers only the `OSIRIS_*` settings it declares.
    Third-party credentials are read with `os.environ.get`, so `.env` held the
    keys while the agent reported them unset and ran without research -- warning
    once and then proceeding, which is easy to miss in a wall of log lines.
    """

    def test_config_import_populates_the_environment(self, tmp_path, monkeypatch):
        import importlib

        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        env = tmp_path / ".env"
        env.write_text("OPENROUTER_API_KEY=from-dotenv\n")

        import osiris.config as config

        monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
        config._load_env()

        assert os.environ.get("OPENROUTER_API_KEY") == "from-dotenv"
        importlib.reload(config)

    def test_a_real_shell_variable_wins_over_the_file(self, tmp_path, monkeypatch):
        """`override=False`. An explicit export must beat the checked-in file."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "from-shell")
        env = tmp_path / ".env"
        env.write_text("OPENROUTER_API_KEY=from-dotenv\n")

        import osiris.config as config

        monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
        config._load_env()

        assert os.environ["OPENROUTER_API_KEY"] == "from-shell"

    def test_a_missing_env_file_is_not_an_error(self, tmp_path, monkeypatch):
        """A fresh clone has no .env and must still import."""
        import osiris.config as config

        monkeypatch.setattr(config, "REPO_ROOT", tmp_path / "absent")
        config._load_env()  # must not raise


class TestLimitsScaledToAccountSize:
    """The defaults target a five-figure account.

    At a few hundred dollars they are not conservative but INFEASIBLE, and the
    failure looks like a broken agent rather than a misconfiguration: the kernel
    vetoes nearly every order it proposes.
    """

    def test_a_large_account_keeps_the_defaults(self):
        assert RiskLimits.for_equity(50_000.0) == RiskLimits()

    def test_a_tiny_account_can_fill_a_position_in_one_order(self):
        """The actual bug: $366 equity meant $7.32 orders and $18 targets.

        Roughly 50 orders to build the book against a 60-order daily budget, so
        the book could never be established.
        """
        equity = 366.13
        limits = RiskLimits.for_equity(equity)

        per_order = equity * limits.max_trade_notional_pct
        per_position = equity / limits.target_position_count
        assert per_position <= per_order, "a position must fill in one order"

    def test_a_tiny_account_concentrates_deliberately(self):
        """Fewer names is the honest tradeoff, not an oversight.

        A $366 account cannot be both diversified across 20 names and
        meaningfully invested in any of them.
        """
        limits = RiskLimits.for_equity(366.13)

        assert limits.target_position_count <= 6
        assert limits.max_symbol_weight > RiskLimits().max_symbol_weight

    def test_a_mid_size_account_gets_a_middle_preset(self):
        limits = RiskLimits.for_equity(5_000.0)

        assert 6 <= limits.target_position_count <= 12

    @pytest.mark.parametrize("equity", [100.0, 366.13, 1_500.0, 5_000.0, 25_000.0])
    def test_every_preset_is_internally_coherent(self, equity):
        """A preset that deadlocks the kernel is worse than a bad preset."""
        limits = RiskLimits.for_equity(equity)

        assert limits.max_symbol_weight * limits.target_position_count >= 1.0
        assert limits.min_position_count <= limits.target_position_count

    def test_explicit_overrides_win(self):
        limits = RiskLimits.for_equity(366.13, target_position_count=3)

        assert limits.target_position_count == 3
