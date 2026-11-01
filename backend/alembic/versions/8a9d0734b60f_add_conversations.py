"""add conversations

Revision ID: 8a9d0734b60f
Revises: a82118103a6e
Create Date: 2026-10-30 18:30:00.000000

Written by hand (same result as autogenerate would give, reviewed line by line).
Existing runs keep conversation_id = NULL: they belong to no conversation.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '8a9d0734b60f'
down_revision: Union[str, Sequence[str], None] = 'a82118103a6e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('conversations',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('agent_id', sa.Integer(), nullable=False),
    sa.Column('title', sa.String(length=255), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_conversations_agent_id'), 'conversations', ['agent_id'], unique=False)

    # Nullable: no default needed for the existing rows, and the change is instant
    op.add_column('agent_runs', sa.Column('conversation_id', sa.Integer(), nullable=True))
    op.create_index(op.f('ix_agent_runs_conversation_id'), 'agent_runs', ['conversation_id'], unique=False)
    op.create_foreign_key(
        'fk_agent_runs_conversation_id', 'agent_runs', 'conversations',
        ['conversation_id'], ['id'], ondelete='CASCADE',
    )


def downgrade() -> None:
    """Downgrade schema: the reverse, in the reverse order."""
    op.drop_constraint('fk_agent_runs_conversation_id', 'agent_runs', type_='foreignkey')
    op.drop_index(op.f('ix_agent_runs_conversation_id'), table_name='agent_runs')
    op.drop_column('agent_runs', 'conversation_id')
    op.drop_index(op.f('ix_conversations_agent_id'), table_name='conversations')
    op.drop_table('conversations')
