from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ReviewItem


async def add_item(session: AsyncSession, item: ReviewItem) -> ReviewItem:
    session.add(item)
    await session.flush()
    return item


async def pending_for(session: AsyncSession, applicant_id: int) -> ReviewItem | None:
    stmt = (
        select(ReviewItem)
        .where(ReviewItem.applicant_id == applicant_id, ReviewItem.status == "pending")
        .order_by(ReviewItem.id)
        .limit(1)
        .with_for_update()
    )
    return await session.scalar(stmt)


async def list_pending(session: AsyncSession) -> list[ReviewItem]:
    stmt = select(ReviewItem).where(ReviewItem.status == "pending").order_by(ReviewItem.id)
    return list(await session.scalars(stmt))


async def get_for_update(session: AsyncSession, review_id: int) -> ReviewItem | None:
    """Row-locked, so two reviewers can't resolve the same item at once."""
    stmt = select(ReviewItem).where(ReviewItem.id == review_id).with_for_update()
    return await session.scalar(stmt)
