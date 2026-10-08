"""Optional historical archive inputs for numerical replay tests."""

from pathlib import Path
from typing import Callable

import pytest


@pytest.fixture
def require_archive_files() -> Callable[..., None]:
    def require(*paths: Path) -> None:
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            pytest.skip(
                "Historical archives not distributed with code release: "
                + ", ".join(missing)
            )

    return require
