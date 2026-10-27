from sqlalchemy.orm import Session

from app.db.models import RunLog


def add_run(session: Session, run: RunLog) -> RunLog:
    session.add(run)
    session.flush()
    return run
