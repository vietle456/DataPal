import logging
import tempfile
from pathlib import Path

import docker.errors
import requests.exceptions

import docker

logger = logging.getLogger(__name__)

EXECUTION_TIMEOUT_SECONDS = 15


def _snapshot(directory: Path) -> set[Path]:
    """Return the set of all files currently present under *directory*."""
    if not directory.exists():
        return set()
    return {p for p in directory.rglob("*") if p.is_file()}


class SandboxRunner:
    def __init__(self) -> None:
        try:
            self.client = docker.from_env()
        except docker.errors.DockerException as exc:
            logger.error(
                "[SandboxRunner] Cannot connect to Docker daemon — "
                "make sure Docker is running. Detail: %s",
                exc,
            )
            raise

    @staticmethod
    def _force_stop(container) -> None:
        """Best-effort SIGKILL; never raises."""
        if container is None:
            return
        try:
            container.kill()
        except docker.errors.DockerException as kill_err:
            # Already exited / not running is fine; anything else is logged.
            logger.warning("[SandboxRunner] Failed to kill container: %s", kill_err)
        except requests.exceptions.RequestException as kill_err:
            logger.warning("[SandboxRunner] Failed to kill container: %s", kill_err)

    def execute(
        self,
        code_str: str,
        uploads_path: Path,
        sql_results_path: Path,
        artifacts_path: Path,
    ) -> dict:
        # Ensure all bound host directories exist before Docker mounts them.
        uploads_path.mkdir(parents=True, exist_ok=True)
        sql_results_path.mkdir(parents=True, exist_ok=True)
        artifacts_path.mkdir(parents=True, exist_ok=True)

        stdout_str = ""
        stderr_str = ""
        exit_code = 0

        # Snapshot existing output artifacts so we can report only what this run produced.
        before_output = _snapshot(artifacts_path)

        with tempfile.TemporaryDirectory() as tmp:
            temp_dir = Path(tmp)

            # 1. Write generated code
            script_path = temp_dir / "script.py"
            script_path.write_text(code_str, encoding="utf-8")

            container = None
            try:
                # 2. Spawn Docker container with 15s timeout.
                #    Storage directories are bind-mounted directly, so files
                #    written inside the container appear on the host immediately
                #    with no extra copy step required.
                container = self.client.containers.create(
                    image="data-agent-runner:latest",
                    network_mode="none",
                    mem_limit="512m",
                    nano_cpus=1000000000,
                    user="nobody",
                    read_only=True,
                    cap_drop=["ALL"],
                    # In-memory writable scratch space needed by data-science libs
                    # even though the root FS is read-only:
                    #   /tmp  — matplotlib font cache, pandas temp buffers, .pyc files
                    #   /root — plotly / home-dir config fallback for `nobody`
                    # Size-capped at 64 MiB each to limit memory impact.
                    tmpfs={
                        "/tmp": "size=64m,mode=1777",
                        "/root": "size=64m,mode=0700",
                    },
                    volumes={
                        str(temp_dir): {
                            "bind": "/workspace",
                            "mode": "rw",
                        },
                        # Read-only: container only reads uploaded source files.
                        str(uploads_path): {
                            "bind": "/workspace/input",
                            "mode": "ro",
                        },
                        # Read-write: container writes query results (parquet, csv…).
                        str(sql_results_path): {
                            "bind": "/workspace/intermediate",
                            "mode": "rw",
                        },
                        # Read-write: container writes output artifacts (charts, reports…).
                        str(artifacts_path): {
                            "bind": "/workspace/output",
                            "mode": "rw",
                        },
                    },
                    detach=True,
                    # Seconds Docker waits after SIGTERM before SIGKILL on stop.
                    stop_timeout=2,
                )

                container.start()
                res = container.wait(timeout=15)
                exit_code = res.get("StatusCode", 0)

                stdout_str = container.logs(stdout=True, stderr=False).decode(
                    "utf-8", errors="replace"
                )
                stderr_str = container.logs(stdout=False, stderr=True).decode(
                    "utf-8", errors="replace"
                )

            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                # container.wait(timeout=...) raises a *client-side* read timeout; the
                # container itself keeps running, so it must be killed explicitly.
                # (ConnectionError is raised by docker-py on some platforms for the same case.)
                logger.warning("[SandboxRunner] Container execution timed out: %s", e)
                exit_code = 124
                stderr_str = f"Execution timed out after {EXECUTION_TIMEOUT_SECONDS}s"
                self._force_stop(container)
            except docker.errors.DockerException as e:
                logger.error(
                    "[SandboxRunner] Docker error during container execution — "
                    "is Docker running? Detail: %s",
                    e,
                )
                stderr_str = f"Sandbox error: {e}"
                exit_code = 1
                self._force_stop(container)
            except Exception as e:  # noqa: BLE001
                # Never let an unexpected failure crash the MCP server process.
                logger.exception("[SandboxRunner] Unexpected error during execution")
                stderr_str = f"Sandbox error: {e}"
                exit_code = 1
                self._force_stop(container)
            finally:
                if container:
                    try:
                        container.remove(force=True)
                    except docker.errors.DockerException as remove_err:
                        logger.warning(
                            "[SandboxRunner] Failed to remove container %s: %s",
                            getattr(container, "short_id", "?"),
                            remove_err,
                        )

        # 3. Collect Python-produced artifact paths only (new files under artifacts_path).
        #    SQL result files written to sql_results_path are intentionally excluded.
        new_output = _snapshot(artifacts_path) - before_output
        artifacts = [str(p) for p in sorted(new_output)]

        return {
            "stdout": stdout_str,
            "stderr": stderr_str,
            "exit_code": exit_code,
            "artifacts": artifacts,
        }
