from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ReviewItem


def add_item(session: Session, item: ReviewItem) -> ReviewItem:
    session.add(item)
    session.flush()
    return item


def list_pending(session: Session) -> list[ReviewItem]:
    stmt = select(ReviewItem).where(ReviewItem.status == "pending").order_by(ReviewItem.id)
    return list(session.scalars(stmt))
