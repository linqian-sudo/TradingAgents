"""Pinned upstream identity. The install bundle does not fetch a newer commit."""

from __future__ import annotations

UPSTREAM_REPO = "https://github.com/TauricResearch/TradingAgents"
UPSTREAM_COMMIT = "9fa1196fffbcfd9de5146d2dd2869df84ab99838"
UPSTREAM_VERSION = "0.7.1"
FORK_REPO = "https://github.com/linqian-sudo/TradingAgents"
LICENSE_NAME = "Apache-2.0"
SCHEMA = "tradingagents-research-status/v1"

# Public tickers named for the mapping check. Mapping does not assert that
# Yahoo lists them or return a price.
REFERENCE_SYMBOLS = (
    ("0700.HK", "HK"),
    ("1810.HK", "HK"),
    ("601398.SS", "SS"),
    ("002779.SZ", "SZ"),
    ("688825.SS", "SS"),
)
