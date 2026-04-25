from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import os
import subprocess

import pandas as pd


MYSQL_CLI_CANDIDATES = [
    Path(r"C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe"),
    Path("mysql.exe"),
]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


def load_env_file(path: Path = DEFAULT_ENV_PATH, *, override: bool = False) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        if override or key not in os.environ:
            os.environ[key] = value


load_env_file()


@dataclass(frozen=True)
class MySQLSettings:
    host: str
    port: int
    user: str
    password: str
    database: str

    @classmethod
    def from_env(cls, database_override: str | None = None) -> "MySQLSettings":
        host = os.getenv("MYSQL_HOST", "").strip()
        port_value = os.getenv("MYSQL_PORT", "").strip()
        port = int(port_value) if port_value else 0
        user = os.getenv("MYSQL_USER", "").strip()
        password = os.getenv("MYSQL_PASSWORD", "")
        database = (database_override or os.getenv("MYSQL_DATABASE", "")).strip()
        return cls(host=host, port=port, user=user, password=password, database=database)


def require_connection_settings(settings: MySQLSettings) -> None:
    missing: list[str] = []
    if not settings.host:
        missing.append("MYSQL_HOST")
    if not settings.port:
        missing.append("MYSQL_PORT")
    if not settings.database:
        missing.append("MYSQL_DATABASE")
    if missing:
        raise ValueError(
            "Faltan variables de conexion MySQL en el entorno: " + ", ".join(missing) + "."
        )


def find_mysql_cli() -> Path:
    for candidate in MYSQL_CLI_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No se encontro mysql.exe. Ajusta la ruta del cliente MySQL antes de ejecutar el bootstrap."
    )


def require_credentials(settings: MySQLSettings) -> None:
    require_connection_settings(settings)
    if not settings.user or not settings.password:
        raise ValueError(
            "Faltan credenciales MySQL. Define MYSQL_USER y MYSQL_PASSWORD en el entorno."
        )


def _build_command(
    settings: MySQLSettings,
    database: str | None = None,
    *,
    batch: bool = False,
    raw: bool = False,
    skip_column_names: bool = False,
) -> list[str]:
    command = [
        str(find_mysql_cli()),
        "--protocol=TCP",
        "-h",
        settings.host,
        "-P",
        str(settings.port),
        "-u",
        settings.user,
        "--default-character-set=utf8mb4",
    ]
    if database:
        command.extend(["--database", database])
    if batch:
        command.append("--batch")
    if raw:
        command.append("--raw")
    if skip_column_names:
        command.append("--skip-column-names")
    return command


def execute_sql(
    sql_text: str,
    settings: MySQLSettings,
    *,
    database: str | None = None,
) -> subprocess.CompletedProcess[str]:
    require_credentials(settings)
    env = os.environ.copy()
    env["MYSQL_PWD"] = settings.password
    return subprocess.run(
        _build_command(settings, database=database),
        input=sql_text,
        text=True,
        encoding="utf-8",
        capture_output=True,
        env=env,
        check=True,
    )


def query_dataframe(
    sql_text: str,
    settings: MySQLSettings,
    *,
    database: str | None = None,
) -> pd.DataFrame:
    require_credentials(settings)
    env = os.environ.copy()
    env["MYSQL_PWD"] = settings.password
    result = subprocess.run(
        _build_command(settings, database=database, batch=True, raw=True),
        input=sql_text,
        text=True,
        encoding="utf-8",
        capture_output=True,
        env=env,
        check=True,
    )
    if not result.stdout.strip():
        return pd.DataFrame()
    return pd.read_csv(StringIO(result.stdout), sep="\t")
