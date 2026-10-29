from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import RunLog


async def add_run(session: AsyncSession, run: RunLog) -> RunLog:
    session.add(run)
    await session.flush()
    return run


async def latest_sent(
    session: AsyncSession, applicant_id: int, prompt_version: int, model: str
) -> RunLog | None:
    """The newest notice sent for this case with this prompt version and model: the response
    cache. Only sent notices are returned, and a sent notice passed every check."""
    stmt = (
        select(RunLog)
        .where(
            RunLog.applicant_id == applicant_id,
            RunLog.prompt_version == prompt_version,
            RunLog.model == model,
            RunLog.outcome.in_(["sent", "revised"]),
            RunLog.notice.is_not(None),
        )
        .order_by(RunLog.id.desc())
        .limit(1)
    )
    return await session.scalar(stmt)
