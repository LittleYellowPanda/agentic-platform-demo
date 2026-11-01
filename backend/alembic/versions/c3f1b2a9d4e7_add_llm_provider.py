"""add llm provider to agents and runs

Revision ID: c3f1b2a9d4e7
Revises: 8a9d0734b60f
Create Date: 2026-10-31 15:00:00.000000

Written by hand. Nullable columns: existing agents and runs keep NULL, which means
"the default provider" (LLM_* variables), exactly what they used before.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'c3f1b2a9d4e7'
down_revision: Union[str, Sequence[str], None] = '8a9d0734b60f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('agents', sa.Column('provider', sa.String(length=50), nullable=True))
    op.add_column('agent_runs', sa.Column('provider', sa.String(length=50), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('agent_runs', 'provider')
    op.drop_column('agents', 'provider')
