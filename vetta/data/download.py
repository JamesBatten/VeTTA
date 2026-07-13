from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

DEFAULT_DATA_DIR = "data"
_SPINNER_FRAMES = ("|", "/", "-", "\\")


class DataDownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    record_id: str
    record_url: str


@dataclass(frozen=True)
class DownloadResult:
    dataset: str
    destination: Path
    files: tuple[Path, ...]
    dry_run: bool


class ProgressReporter(Protocol):
    def dataset_started(
        self,
        dataset: DatasetSpec,
        destination: Path,
        file_count: int,
    ) -> None: ...

    def dataset_replacing(
        self,
        dataset: DatasetSpec,
        destination: Path,
    ) -> None: ...

    def file_started(
        self,
        file_name: str,
        file_index: int,
        file_count: int,
    ) -> None: ...

    def file_progress(
        self,
        downloaded_bytes: int,
        total_bytes: int | None,
    ) -> None: ...

    def file_completed(
        self,
        destination: Path,
        file_index: int,
        file_count: int,
    ) -> None: ...

    def dataset_completed(
        self,
        dataset: DatasetSpec,
        destination: Path,
        file_count: int,
    ) -> None: ...

    def archive_extracting(self, archive_path: Path) -> None: ...

    def archive_extracted(
        self,
        archive_path: Path,
        extracted_file_count: int,
    ) -> None: ...


class NullProgressReporter:
    def dataset_started(
        self,
        dataset: DatasetSpec,
        destination: Path,
        file_count: int,
    ) -> None:
        _ = dataset, destination, file_count

    def dataset_replacing(
        self,
        dataset: DatasetSpec,
        destination: Path,
    ) -> None:
        _ = dataset, destination

    def file_started(
        self,
        file_name: str,
        file_index: int,
        file_count: int,
    ) -> None:
        _ = file_name, file_index, file_count

    def file_progress(
        self,
        downloaded_bytes: int,
        total_bytes: int | None,
    ) -> None:
        _ = downloaded_bytes, total_bytes

    def file_completed(
        self,
        destination: Path,
        file_index: int,
        file_count: int,
    ) -> None:
        _ = destination, file_index, file_count

    def dataset_completed(
        self,
        dataset: DatasetSpec,
        destination: Path,
        file_count: int,
    ) -> None:
        _ = dataset, destination, file_count

    def archive_extracting(self, archive_path: Path) -> None:
        _ = archive_path

    def archive_extracted(
        self,
        archive_path: Path,
        extracted_file_count: int,
    ) -> None:
        _ = archive_path, extracted_file_count


class ConsoleProgressReporter:
    def __init__(self, stream=None) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._spinner_idx = 0
        self._start_time = 0.0

    def dataset_started(
        self,
        dataset: DatasetSpec,
        destination: Path,
        file_count: int,
    ) -> None:
        print(
            f"\n\033[1;36m=== Downloading {dataset.name} ===\033[0m\n"
            f"Target: {destination}\n"
            f"Files: {file_count}",
            file=self._stream,
            flush=True,
        )

    def dataset_replacing(
        self,
        dataset: DatasetSpec,
        destination: Path,
    ) -> None:
        print(
            f"\033[33mReplacing existing dataset directory: {destination}\033[0m",
            file=self._stream,
            flush=True,
        )

    def file_started(
        self,
        file_name: str,
        file_index: int,
        file_count: int,
    ) -> None:
        self._spinner_idx = 0
        self._start_time = time.monotonic()
        print(
            f"  [{file_index}/{file_count}] {file_name}",
            file=self._stream,
            flush=True,
        )

    def file_progress(
        self,
        downloaded_bytes: int,
        total_bytes: int | None,
    ) -> None:
        spinner = _SPINNER_FRAMES[self._spinner_idx % len(_SPINNER_FRAMES)]
        self._spinner_idx += 1

        elapsed = max(time.monotonic() - self._start_time, 1e-6)
        speed_mib_s = (downloaded_bytes / (1024 * 1024)) / elapsed

        if total_bytes and total_bytes > 0:
            ratio = min(downloaded_bytes / total_bytes, 1.0)
            bar = _render_bar(ratio)
            status = (
                f"\r    {spinner} {bar} {ratio * 100:6.2f}% "
                f"{_format_bytes(downloaded_bytes)}/{_format_bytes(total_bytes)} "
                f"{speed_mib_s:6.2f} MiB/s"
            )
        else:
            status = (
                f"\r    {spinner} {_format_bytes(downloaded_bytes)} downloaded "
                f"{speed_mib_s:6.2f} MiB/s"
            )

        print(status, end="", file=self._stream, flush=True)

    def file_completed(
        self,
        destination: Path,
        file_index: int,
        file_count: int,
    ) -> None:
        _ = file_index, file_count
        print(file=self._stream, flush=True)
        print(
            f"    \033[32mOK\033[0m {destination.name}",
            file=self._stream,
            flush=True,
        )

    def dataset_completed(
        self,
        dataset: DatasetSpec,
        destination: Path,
        file_count: int,
    ) -> None:
        print(
            f"\033[1;32mDone\033[0m {dataset.name} -> {destination} "
            f"({file_count} file(s))",
            file=self._stream,
            flush=True,
        )

    def archive_extracting(self, archive_path: Path) -> None:
        print(
            f"    Extracting {archive_path.name} ...",
            file=self._stream,
            flush=True,
        )

    def archive_extracted(
        self,
        archive_path: Path,
        extracted_file_count: int,
    ) -> None:
        print(
            f"    \033[32mOK\033[0m extracted {archive_path.name} "
            f"({extracted_file_count} file(s)); removed archive",
            file=self._stream,
            flush=True,
        )


def _render_bar(ratio: float, width: int = 30) -> str:
    ratio = max(0.0, min(ratio, 1.0))
    filled = int(width * ratio)
    empty = width - filled
    return f"[{'#' * filled}{'-' * empty}]"


def _format_bytes(byte_count: int) -> str:
    return f"{byte_count / (1024 * 1024):.2f} MiB"


HttpGet = Callable[[str], bytes]
MetadataFetcher = Callable[[DatasetSpec], Mapping[str, object]]
ProgressCallback = Callable[[int, int | None], None]
FileDownloader = Callable[..., None]


DATASET_SPECS: dict[str, DatasetSpec] = {
    "SSA_0.1": DatasetSpec(
        name="SSA_0.1",
        record_id="10076802",
        record_url="https://zenodo.org/records/10076802",
    ),
    "SSA_0.2": DatasetSpec(
        name="SSA_0.2",
        record_id="15741230",
        record_url="https://zenodo.org/records/15741230",
    ),
}


def list_supported_datasets() -> tuple[str, ...]:
    return tuple(DATASET_SPECS.keys())


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    dataset = DATASET_SPECS.get(dataset_name)
    if dataset is None:
        supported = ", ".join(list_supported_datasets())
        raise DataDownloadError(
            f"Unknown dataset '{dataset_name}'. Supported datasets: {supported}"
        )
    return dataset


def ensure_directory(path: str | Path) -> Path:
    resolved_path = Path(path)
    resolved_path.mkdir(parents=True, exist_ok=True)
    return resolved_path


def ensure_data_dir(data_dir: str | Path = DEFAULT_DATA_DIR) -> Path:
    return ensure_directory(data_dir)


def normalize_dataset_selection(
    dataset_names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    if not dataset_names:
        return list_supported_datasets()

    seen = set()
    resolved_names = []
    for dataset_name in dataset_names:
        spec = get_dataset_spec(dataset_name)
        if spec.name in seen:
            continue
        seen.add(spec.name)
        resolved_names.append(spec.name)

    return tuple(resolved_names)


def build_zenodo_api_url(record_id: str) -> str:
    return f"https://zenodo.org/api/records/{record_id}"


def default_http_get(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:
        return response.read()


def fetch_zenodo_record_metadata(
    dataset: DatasetSpec,
    *,
    http_get: HttpGet = default_http_get,
) -> Mapping[str, object]:
    api_url = build_zenodo_api_url(dataset.record_id)
    payload = http_get(api_url)
    try:
        metadata = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataDownloadError(
            f"Failed to parse metadata for {dataset.name} from {api_url}."
        ) from exc

    if not isinstance(metadata, dict):
        raise DataDownloadError(
            f"Metadata for {dataset.name} from {api_url} was not a JSON object."
        )

    return metadata


def extract_file_targets(
    record_metadata: Mapping[str, object],
) -> tuple[tuple[str, str], ...]:
    files = record_metadata.get("files")
    if not isinstance(files, list):
        raise DataDownloadError("Record metadata did not contain a valid 'files' list.")

    targets = []
    for idx, file_data in enumerate(files):
        if not isinstance(file_data, Mapping):
            raise DataDownloadError(f"Record file entry {idx} was not an object.")

        file_name = file_data.get("key")
        if not isinstance(file_name, str) or not file_name:
            raise DataDownloadError(f"Record file entry {idx} did not include 'key'.")

        links = file_data.get("links")
        if not isinstance(links, Mapping):
            raise DataDownloadError(f"Record file entry {idx} did not include 'links'.")

        download_url = links.get("self") or links.get("download")
        if not isinstance(download_url, str) or not download_url:
            raise DataDownloadError(
                f"Record file entry {idx} did not include a download URL."
            )

        targets.append((file_name, download_url))

    if not targets:
        raise DataDownloadError("Record metadata contained no downloadable files.")

    return tuple(targets)


def _parse_content_length(raw_value: str | None) -> int | None:
    if raw_value is None:
        return None
    try:
        parsed = int(raw_value)
    except ValueError:
        return None
    if parsed < 0:
        return None
    return parsed


def default_download_file(
    url: str,
    destination: Path,
    *,
    progress_callback: ProgressCallback | None = None,
    chunk_size: int = 1024 * 1024,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(url) as response, destination.open("wb") as file_obj:
        total_bytes = _parse_content_length(response.headers.get("Content-Length"))
        downloaded_bytes = 0
        if progress_callback is not None:
            progress_callback(downloaded_bytes, total_bytes)

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            file_obj.write(chunk)
            downloaded_bytes += len(chunk)
            if progress_callback is not None:
                progress_callback(downloaded_bytes, total_bytes)


def smoke_metadata_fetcher(dataset: DatasetSpec) -> Mapping[str, object]:
    """Offline stand-in for ``fetch_zenodo_record_metadata``.

    Returns a synthetic Zenodo-shaped record so the real ``extract_file_targets``
    runs without any network call. The single archive's key matches the dataset
    name so the extracted layout collapses through ``normalize_dataset_layout``.
    """
    return {
        "files": [
            {
                "key": f"{dataset.name}.zip",
                "links": {"self": f"smoke://{dataset.record_id}/{dataset.name}.zip"},
            }
        ]
    }


def smoke_file_downloader(
    url: str,
    destination: Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Offline stand-in for ``default_download_file``.

    Instead of fetching bytes over the network it writes a tiny but real zip
    archive (members nested under a ``<dataset>/`` top dir, matching how the
    Zenodo archives are shaped) so the genuine staging / extraction / layout /
    finalize code all runs against it.
    """
    del url
    destination.parent.mkdir(parents=True, exist_ok=True)
    top = destination.stem  # equals the dataset name for the smoke target
    with zipfile.ZipFile(destination, "w") as archive:
        archive.writestr(f"{top}/train/sample.txt", b"vetta smoke skip_vessel\n")
        archive.writestr(f"{top}/README.txt", b"vetta smoke skip_vessel\n")
    if progress_callback is not None:
        total = destination.stat().st_size
        progress_callback(0, total)
        progress_callback(total, total)


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return
    path.unlink()


def _supports_progress_callback(file_downloader: FileDownloader) -> bool:
    try:
        signature = inspect.signature(file_downloader)
    except (TypeError, ValueError):
        return False

    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return "progress_callback" in signature.parameters


def _build_staging_dir(root_path: Path, dataset_name: str) -> Path:
    return root_path / f".{dataset_name}.download.tmp"


def _list_files_recursive(directory: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (path for path in directory.rglob("*") if path.is_file()),
            key=lambda path: str(path),
        )
    )


def normalize_dataset_layout(
    dataset_staging_dir: Path,
    dataset_name: str,
) -> None:
    root_entries = tuple(dataset_staging_dir.iterdir())
    if len(root_entries) != 1:
        return

    only_entry = root_entries[0]
    if only_entry.name != dataset_name or not only_entry.is_dir():
        return

    for child in tuple(only_entry.iterdir()):
        destination = dataset_staging_dir / child.name
        child.rename(destination)
    only_entry.rmdir()


def _resolve_member_destination(
    dataset_dir: Path,
    member_name: str,
) -> Path:
    target = (dataset_dir / member_name).resolve()
    base = dataset_dir.resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise DataDownloadError(
            f"Archive member path escapes dataset directory: {member_name}"
        ) from exc
    return target


def extract_zip_archive(
    archive_path: Path,
    destination_dir: Path,
) -> tuple[Path, ...]:
    extracted_files = []
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            destination = _resolve_member_destination(destination_dir, member.filename)
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted_files.append(destination)

    return tuple(extracted_files)


def materialize_downloaded_files(
    downloaded_files: Sequence[Path],
    *,
    progress_reporter: ProgressReporter | None = None,
    remove_archives: bool = True,
) -> tuple[Path, ...]:
    reporter = progress_reporter if progress_reporter is not None else NullProgressReporter()
    materialized_files = []

    for file_path in downloaded_files:
        if file_path.suffix.lower() != ".zip":
            materialized_files.append(file_path)
            continue

        reporter.archive_extracting(file_path)
        extracted_files = extract_zip_archive(file_path, file_path.parent)
        if remove_archives:
            file_path.unlink()
        reporter.archive_extracted(file_path, len(extracted_files))
        materialized_files.extend(extracted_files)

    return tuple(materialized_files)


def _select_progress_reporter(
    progress_reporter: ProgressReporter | None,
    *,
    dry_run: bool,
) -> ProgressReporter:
    if progress_reporter is not None:
        return progress_reporter
    if dry_run:
        return NullProgressReporter()
    return ConsoleProgressReporter()


def _download_file_targets(
    file_targets: Sequence[tuple[str, str]],
    *,
    staging_dir: Path,
    file_downloader: FileDownloader,
    progress_reporter: ProgressReporter,
) -> tuple[Path, ...]:
    supports_progress_callback = _supports_progress_callback(file_downloader)
    staged_files = []
    file_count = len(file_targets)

    for file_idx, (file_name, download_url) in enumerate(file_targets, start=1):
        staged_destination = staging_dir / file_name
        ensure_directory(staged_destination.parent)

        progress_reporter.file_started(file_name, file_idx, file_count)
        if supports_progress_callback:
            file_downloader(
                download_url,
                staged_destination,
                progress_callback=progress_reporter.file_progress,
            )
        else:
            file_downloader(download_url, staged_destination)
        progress_reporter.file_completed(staged_destination, file_idx, file_count)
        staged_files.append(staged_destination)

    return tuple(staged_files)


def _finalize_staged_dataset(
    *,
    staging_dir: Path,
    dataset_dir: Path,
    dataset: DatasetSpec,
    progress_reporter: ProgressReporter,
    clean_existing: bool,
) -> tuple[Path, ...]:
    normalize_dataset_layout(staging_dir, dataset.name)
    staged_final_files = _list_files_recursive(staging_dir)

    if clean_existing and dataset_dir.exists():
        progress_reporter.dataset_replacing(dataset, dataset_dir)
        _remove_path(dataset_dir)

    staging_dir.rename(dataset_dir)

    completed_files = tuple(
        dataset_dir / staged_file.relative_to(staging_dir)
        for staged_file in staged_final_files
    )
    progress_reporter.dataset_completed(dataset, dataset_dir, len(completed_files))
    return completed_files


def _format_cli_result(result: DownloadResult) -> str:
    if result.dry_run:
        return f"[dry-run] Prepared directory for {result.dataset}: {result.destination}"
    return (
        f"Downloaded {result.dataset} into {result.destination} "
        f"({len(result.files)} file(s))."
    )


def download_dataset(
    dataset: DatasetSpec,
    data_root: str | Path,
    *,
    metadata_fetcher: MetadataFetcher = fetch_zenodo_record_metadata,
    file_downloader: FileDownloader = default_download_file,
    dry_run: bool = False,
    progress_reporter: ProgressReporter | None = None,
    clean_existing: bool = True,
) -> DownloadResult:
    root_path = ensure_directory(data_root).resolve()
    dataset_dir = root_path / dataset.name

    if dry_run:
        ensure_directory(dataset_dir)
        return DownloadResult(
            dataset=dataset.name,
            destination=dataset_dir,
            files=tuple(),
            dry_run=True,
        )

    metadata = metadata_fetcher(dataset)
    file_targets = extract_file_targets(metadata)

    reporter = progress_reporter if progress_reporter is not None else NullProgressReporter()
    reporter.dataset_started(dataset, dataset_dir, len(file_targets))

    staging_dir = _build_staging_dir(root_path, dataset.name)
    _remove_path(staging_dir)
    ensure_directory(staging_dir)

    try:
        staged_files = _download_file_targets(
            file_targets,
            staging_dir=staging_dir,
            file_downloader=file_downloader,
            progress_reporter=reporter,
        )

        _ = materialize_downloaded_files(
            staged_files,
            progress_reporter=reporter,
            remove_archives=True,
        )
        completed_files = _finalize_staged_dataset(
            staging_dir=staging_dir,
            dataset_dir=dataset_dir,
            dataset=dataset,
            progress_reporter=reporter,
            clean_existing=clean_existing,
        )

        return DownloadResult(
            dataset=dataset.name,
            destination=dataset_dir,
            files=completed_files,
            dry_run=False,
        )
    except Exception:
        _remove_path(staging_dir)
        raise


def download_datasets(
    dataset_names: Sequence[str] | None = None,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    dry_run: bool = False,
    *,
    metadata_fetcher: MetadataFetcher = fetch_zenodo_record_metadata,
    file_downloader: FileDownloader = default_download_file,
    progress_reporter: ProgressReporter | None = None,
    clean_existing: bool = True,
) -> tuple[DownloadResult, ...]:
    root_path = ensure_data_dir(data_dir)
    selected_datasets = normalize_dataset_selection(dataset_names)
    reporter = _select_progress_reporter(progress_reporter, dry_run=dry_run)

    results = []
    for dataset_name in selected_datasets:
        dataset = get_dataset_spec(dataset_name)
        result = download_dataset(
            dataset=dataset,
            data_root=root_path,
            metadata_fetcher=metadata_fetcher,
            file_downloader=file_downloader,
            dry_run=dry_run,
            progress_reporter=reporter,
            clean_existing=clean_existing,
        )
        results.append(result)

    return tuple(results)


def run_smoke_download(
    dataset_names: Sequence[str] | None = None,
) -> tuple[DownloadResult, ...]:
    """Run the full download pipeline offline, skipping only the network I/O.

    Drives the real ``download_datasets`` path with ``smoke_metadata_fetcher`` /
    ``smoke_file_downloader`` injected, so everything except the actual Zenodo
    metadata fetch and byte transfer executes (enumeration, staging, extraction,
    layout normalisation, and the replace-and-finalize step). Output goes to a
    throwaway temp directory that is pre-seeded so the dataset-replacement branch
    is exercised too, then removed.
    """
    selected = normalize_dataset_selection(dataset_names)
    smoke_root = Path(tempfile.mkdtemp(prefix="vetta_download_smoke_"))
    try:
        for name in selected:
            seeded = smoke_root / name
            seeded.mkdir(parents=True, exist_ok=True)
            (seeded / "stale.txt").write_text("stale\n")
        return download_datasets(
            dataset_names=list(selected),
            data_dir=smoke_root,
            dry_run=False,
            metadata_fetcher=smoke_metadata_fetcher,
            file_downloader=smoke_file_downloader,
        )
    finally:
        shutil.rmtree(smoke_root, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download VeTTA datasets from Zenodo into a local data directory.",
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        choices=list_supported_datasets(),
        help="Dataset to download. Pass multiple times to select multiple datasets.",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Directory where datasets are stored (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create target directories and print actions without downloading files.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List supported datasets and exit.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run the full download pipeline offline into a temp directory with "
            "fake metadata/fetch -- exercises everything except the real network "
            "download. Ignores --data-dir/--dry-run."
        ),
    )
    return parser


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    downloader: Callable[..., tuple[DownloadResult, ...]] = download_datasets,
) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        for dataset_name in list_supported_datasets():
            dataset = get_dataset_spec(dataset_name)
            print(f"{dataset.name}: {dataset.record_url}")
        return 0

    if args.smoke:
        for result in run_smoke_download(args.datasets):
            print(f"[smoke] {_format_cli_result(result)}")
        return 0

    results = downloader(
        dataset_names=args.datasets,
        data_dir=args.data_dir,
        dry_run=args.dry_run,
    )

    for result in results:
        print(_format_cli_result(result))

    return 0
