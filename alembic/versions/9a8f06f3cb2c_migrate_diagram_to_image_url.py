"""migrate diagram storage to image url

Revision ID: 9a8f06f3cb2c
Revises: b3e1c7b2a9f1
Create Date: 2026-04-08 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9a8f06f3cb2c"
down_revision: Union[str, None] = "b3e1c7b2a9f1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("diagrams", sa.Column("image_url", sa.Text(), nullable=True))
    op.drop_column("diagrams", "image_base64")


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column("diagrams", sa.Column("image_base64", sa.Text(), nullable=True))
    op.drop_column("diagrams", "image_url")
