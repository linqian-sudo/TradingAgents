"""Research-only entry point for a pinned TradingAgents install.

This package does not place orders, read a portfolio, send notifications,
or choose a paid model plan. A run without an explicit provider and
credentials stops before any model client is constructed.
"""

__all__ = ["__version__"]

__version__ = "0.3.3"
