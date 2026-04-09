"""add model versions registry

Revision ID: b3e1c7b2a9f1
Revises: d90defe07520
Create Date: 2026-04-08 00:00:00.000000

"""

from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = "b3e1c7b2a9f1"
down_revision: Union[str, None] = "d90defe07520"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
	"""Upgrade schema."""
	# Intentionally left empty: placeholder migration file.
	pass


def downgrade() -> None:
	"""Downgrade schema."""
	# Intentionally left empty: placeholder migration file.
	pass
