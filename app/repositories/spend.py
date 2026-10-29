"""LLM spend per UTC day, the counter behind DAILY_BUDGET_USD."""

from datetime import date

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DailySpend


async def spent_on(session: AsyncSession, day: date) -> float:
    return await session.scalar(select(DailySpend.cost_usd).where(DailySpend.day == day)) or 0.0


async def add_spend(session: AsyncSession, day: date, cost_usd: float) -> None:
    """Atomic upsert, so concurrent requests never lose an increment."""
    stmt = insert(DailySpend).values(day=day, cost_usd=cost_usd)
    stmt = stmt.on_conflict_do_update(
        index_elements=[DailySpend.day],
        set_={"cost_usd": DailySpend.cost_usd + stmt.excluded.cost_usd},
    )
    await session.execute(stmt)
