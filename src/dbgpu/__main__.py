import os
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

@main.command(short_help="Saves a database of GPUs to file.")
@click.argument("output", type=click.Path())
@click.option("--database", "-d", type=click.Path(), help="Path to GPU database. Will use default if not provided.")
def save(
    output: str,
    database: Optional[str]=None
) -> None:
    """
    Saves a database of GPUs to file.

    Accepted formats are JSON, CSV and PKL.
    """
    from dbgpu.db import GPUDatabase
    if database:
        db = GPUDatabase.from_file(database)
    else:
        db = GPUDatabase.default()
    data = [
        spec.to_dict()
        for spec in db.specs
    ]
    name, ext = os.path.splitext(os.path.basename(output))
    if ext == ".json":
        import json
        from dbgpu.util import json_serialize
        with open(output, "w") as f:
            json.dump(data, f, default=json_serialize)
    elif ext == ".csv":
        import csv
        with open(output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(data[0].keys())
            for datum in data:
                writer.writerow(datum.values())
    elif ext == ".pkl":
        import pickle
        with open(output, "wb") as f:
            pickle.dump(data, f)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

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

@main.command(short_help="Compares two GPUs.")
@click.argument("name_1", type=str)
@click.argument("name_2", type=str)
@click.option("--database", "-d", type=click.Path(), help="Path to GPU database. Will use default if not provided.")
@click.option("--fuzzy", "-f", is_flag=True, help="Use fuzzy matching.")
def compare(
    name_1: str,
    name_2: str,
    database: Optional[str]=None,
    fuzzy: bool=False
) -> None:
    """
    Compares the specifications of two GPUs.
    """
    from dbgpu.db import GPUDatabase
    if database:
        db = GPUDatabase.from_file(database)
    else:
        db = GPUDatabase.default()

    try:
        if fuzzy:
            gpu_1 = db.search(name_1)
        else:
            gpu_1 = db[name_1]
    except KeyError:
        click.echo(f"GPU '{name_1}' not found.")
        return

    try:
        if fuzzy:
            gpu_2 = db.search(name_2)
        else:
            gpu_2 = db[name_2]
    except KeyError:
        click.echo(f"GPU '{name_2}' not found.")
        return

    fields = gpu_1.labeled_fields()
    fields_1 = dict(fields)
    fields_1["Name"] = gpu_1.name
    fields_2 = dict(gpu_2.labeled_fields())
    fields_2["Name"] = gpu_2.name
    compared_fields = dict(gpu_1.labeled_comparison_fields(gpu_2))

    all_keys = ["Name"] + [f for (f, v) in fields]

    try:
        import tabulate
        tabulate.PRESERVE_WHITESPACE = True
        click.echo(
            tabulate.tabulate(
                [
                    [
                        key,
                        fields_1.get(key, ""),
                        fields_2.get(key, ""),
                        compared_fields.get(key, "")
                    ]
                    for key in all_keys
                ],
                tablefmt="presto"
            )
        )
    except:
        for key in all_keys:
            click.echo(f"{key}:")
            click.echo(f"  {fields_1.get(key, '')}")
            click.echo(f"  {fields_2.get(key, '')}")
            click.echo(f"  {compared_fields.get(key, '')}")
            click.echo()

try:
    main()
    sys.exit(0)
except Exception as ex:
    sys.stderr.write(f"{ex}\r\n")
    sys.stderr.write(traceback.format_exc())
    sys.stderr.flush()
    sys.exit(5)
