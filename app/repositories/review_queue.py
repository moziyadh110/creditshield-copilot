from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ReviewItem


async def add_item(session: AsyncSession, item: ReviewItem) -> ReviewItem:
    session.add(item)
    await session.flush()
    return item


async def list_pending(session: AsyncSession) -> list[ReviewItem]:
    stmt = select(ReviewItem).where(ReviewItem.status == "pending").order_by(ReviewItem.id)
    return list(await session.scalars(stmt))
