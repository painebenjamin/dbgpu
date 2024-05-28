import sys
import click
import traceback

from typing import Dict, Optional, List

from dbgpu.version import version
from dbgpu.constants import *

@click.group(name="dbgpu")
@click.version_option(version=version, message="%(version)s")
def main() -> None:
    """
    DBGPU command-line tools.
    """
    pass

@main.command(short_help="Builds a database of GPUs.")
@click.option("--output", "-o", type=click.Path(), multiple=True, help="Output file path.", default=[DEFAULT_GPU_DB_PATH], show_default=True)
@click.option("--manufacturer", "-m", multiple=True, default=ALL_MANUFACTURERS, show_default=True, help="GPU manufacturers to include. Pass multiple times for multiple manufacturers.")
@click.option("--start-year", "-y", type=int, help="Start year for GPU database.", default=DEFAULT_START_YEAR, show_default=True)
@click.option("--end-year", "-Y", type=int, help="End year for GPU database.", default=DEFAULT_END_YEAR, show_default=True)
@click.option("--courtesy-delay", "-d", type=float, help="Delay in seconds between requests.", default=DEFAULT_COURTESY_DELAY, show_default=True)
@click.option("--proxy", "-p", type=str, help="HTTPS proxy URL.")
@click.option("--timeout", "-t", type=float, help="Timeout in seconds.")
@click.option("--retry-max", "-r", type=int, help="Maximum number of retries.", default=DEFAULT_RETRY_MAX, show_default=True)
@click.option("--retry-delay", "-R", type=float, help="Delay in seconds between retries.", default=DEFAULT_RETRY_DELAY, show_default=True)
def build(
    output: List[str]=[DEFAULT_GPU_DB_PATH],
    manufacturer: List[MANUFACTURER_LITERAL]=ALL_MANUFACTURERS,
    start_year: int=DEFAULT_START_YEAR,
    end_year: int=DEFAULT_END_YEAR,
    courtesy_delay: float=DEFAULT_COURTESY_DELAY,
    proxy: Optional[str]=None,
    timeout: Optional[float]=None,
    retry_max: int=DEFAULT_RETRY_MAX,
    retry_delay: float=DEFAULT_RETRY_DELAY,
) -> None:
    """
    Builds a database of GPUs from TechPowerUp.
    """
    from dbgpu.tpu import TechPowerUp

    proxies: Dict[str, str] = {}
    if proxy:
        proxies["http"] = proxies["https"] = proxy

    tpu = TechPowerUp(
        proxies=proxies,
        timeout=timeout,
        retry_max=retry_max,
        retry_delay=retry_delay,
        manufacturers=manufacturer
    )
    tpu.fetch_to_file(
        *output,
        start_year=start_year,
        end_year=end_year,
        courtesy_delay=courtesy_delay,
        use_tqdm=True
    )
    click.echo(f"GPU database saved to {output}")

@main.command(short_help="Looks up a GPU by name.")
@click.argument("name", type=str)
@click.option("--database", "-d", type=click.Path(), help="Path to GPU database. Will use default if not provided.")
@click.option("--fuzzy", "-f", is_flag=True, help="Use fuzzy matching.")
def lookup(
    name: str,
    database: Optional[str]=None,
    fuzzy: bool=False
) -> None:
    """
    Looks up a GPU by name.
    """
    from dbgpu.db import GPUDatabase
    if database:
        db = GPUDatabase.from_file(database)
    else:
        db = GPUDatabase.default()
    try:
        if fuzzy:
            click.echo(db.search(name))
        else:
            click.echo(db[name])
    except KeyError:
        click.echo(f"GPU '{name}' not found.")

try:
    main()
    sys.exit(0)
except Exception as ex:
    sys.stderr.write(f"{ex}\r\n")
    sys.stderr.write(traceback.format_exc())
    sys.stderr.flush()
    sys.exit(5)
