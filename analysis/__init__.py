try:
    from analysis.technical import TechnicalAnalyzer
except ImportError:
    TechnicalAnalyzer = None  # type: ignore[assignment,misc]

from analysis.patterns import PatternRecognizer
from analysis.onchain import OnChainAnalyzer
from analysis.correlations import MarketCorrelations

try:
    from analysis.quant import QuantAnalyzer
except ImportError:
    QuantAnalyzer = None  # type: ignore[assignment,misc]

__all__ = ["TechnicalAnalyzer", "PatternRecognizer", "OnChainAnalyzer", "MarketCorrelations", "QuantAnalyzer"]
