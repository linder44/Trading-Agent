"""Time and session context module.

Provides Claude with temporal awareness:
- Trading session (Asia/Europe/US) — each has different volatility/volume
- Day of week — weekends are low volume, Mondays are volatile
- Monthly options/futures expiry dates — high volatility around expiry
- Time since last analysis — context for how fresh the data is
"""

from datetime import datetime, timezone


class TimeContextAnalyzer:
    """Generates time-based context for trading decisions."""

    # Major crypto futures expiry: last Friday of each month (approx)
    # Bitcoin options expiry on Deribit: last Friday of month
    QUARTERLY_EXPIRY_MONTHS = {3, 6, 9, 12}

    def get_time_context(self) -> dict:
        """Get current time context for Claude's analysis."""
        now = datetime.now(timezone.utc)

        session = self._get_trading_session(now.hour)
        day_info = self._get_day_context(now)
        expiry_info = self._get_expiry_context(now)

        return {
            "utc_time": now.strftime("%Y-%m-%d %H:%M UTC"),
            "session": session,
            "day_of_week": now.strftime("%A"),
            "day_context": day_info,
            "expiry": expiry_info,
            "hour_utc": now.hour,
        }

    @staticmethod
    def _get_trading_session(hour_utc: int) -> dict:
        """Determine active trading session.

        Asia:   00:00-08:00 UTC (Tokyo 09:00-17:00)
        Europe: 07:00-16:00 UTC (London 07:00-16:00)
        US:     13:00-22:00 UTC (NY 08:00-17:00 EST)
        """
        sessions = []
        if 0 <= hour_utc < 8:
            sessions.append("asia")
        if 7 <= hour_utc < 16:
            sessions.append("europe")
        if 13 <= hour_utc < 22:
            sessions.append("us")

        if not sessions:
            sessions = ["off_hours"]

        # Overlap periods are highest volume
        overlap = ""
        if 7 <= hour_utc < 8:
            overlap = "asia_europe"
        elif 13 <= hour_utc < 16:
            overlap = "europe_us"

        return {
            "active": sessions,
            "overlap": overlap,
            "volatility_expected": (
                "high" if overlap else
                "medium" if "us" in sessions or "europe" in sessions else
                "low"
            ),
            "note": (
                "Пересечение сессий — максимальная ликвидность и волатильность"
                if overlap else
                "Одна сессия активна" if len(sessions) == 1 and sessions[0] != "off_hours" else
                "Нерабочие часы — низкая ликвидность, возможны резкие движения"
                if sessions == ["off_hours"] else
                ""
            ),
        }

    @staticmethod
    def _get_day_context(now: datetime) -> dict:
        """Get day-of-week trading context."""
        weekday = now.weekday()  # 0=Mon, 6=Sun

        if weekday in (5, 6):
            return {
                "type": "weekend",
                "warning": "Выходные — низкий объём, возможны манипуляции. Избегай крупных позиций.",
                "recommended_size": "reduced",
            }
        elif weekday == 0:
            return {
                "type": "monday",
                "note": "Понедельник — часто гэпы и высокая волатильность на открытии. Жди первые 2ч.",
                "recommended_size": "normal",
            }
        elif weekday == 4:
            return {
                "type": "friday",
                "note": "Пятница — возможно закрытие позиций перед выходными. Осторожно с новыми позициями.",
                "recommended_size": "normal",
            }
        else:
            return {
                "type": "midweek",
                "note": "Середина недели — нормальная ликвидность.",
                "recommended_size": "normal",
            }

    def _get_expiry_context(self, now: datetime) -> dict:
        """Check proximity to options/futures expiry."""
        day = now.day
        month = now.month
        # Last Friday of month is typically between 25-31
        days_in_month = self._days_in_month(now.year, month)
        last_friday = days_in_month
        # Find last Friday
        test_date = now.replace(day=days_in_month)
        while test_date.weekday() != 4:  # 4 = Friday
            last_friday -= 1
            test_date = now.replace(day=last_friday)

        days_to_expiry = last_friday - day

        is_quarterly = month in self.QUARTERLY_EXPIRY_MONTHS

        if days_to_expiry <= 0:
            # Already past expiry this month
            return {"days_to_monthly_expiry": 30 + days_to_expiry, "is_expiry_week": False, "is_quarterly": False}

        return {
            "days_to_monthly_expiry": days_to_expiry,
            "is_expiry_week": days_to_expiry <= 3,
            "is_quarterly": is_quarterly and days_to_expiry <= 7,
            "warning": (
                "КВАРТАЛЬНАЯ экспирация через {0} дн — ожидай высокую волатильность и пин-бары!".format(days_to_expiry)
                if is_quarterly and days_to_expiry <= 3 else
                "Экспирация опционов через {0} дн — повышенная волатильность возможна.".format(days_to_expiry)
                if days_to_expiry <= 3 else
                ""
            ),
        }

    @staticmethod
    def _days_in_month(year: int, month: int) -> int:
        """Get number of days in a month."""
        if month == 12:
            return 31
        next_month = datetime(year, month + 1, 1)
        return (next_month - datetime(year, month, 1)).days
