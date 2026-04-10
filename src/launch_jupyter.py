import argparse
import subprocess
import time
import re
from pathlib import Path

LOG_DIR = Path.home() / "jupyter_logs"
DEFAULT_PORT = 9999

def submit_slurm_job(slurm_args: str, port: int, env_name: str) -> tuple[str, Path]:
    """Submits the job and returns the job ID."""
    # Create a unique log file for this session
    log_file = LOG_DIR / f"jupyter_{int(time.time())}.log"

    activate_cmd = f'source /etc/profile && mamba activate {env_name}'
    
    # Simple sbatch script wrapper
    # We use --output to capture the token/url
    cmd = (
        f"sbatch {slurm_args} --output={log_file} "
        f"--wrap='{activate_cmd} && jupyter notebook --no-browser --port={port} --ip=0.0.0.0'"
    )
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Slurm submission failed: {result.stderr}")
        
    # Parse Job ID (e.g., "Submitted batch job 12345")
    match = re.search(r"job (\d+)", result.stdout)
    if not match:
        raise ValueError("Could not parse Job ID from output.")
        
    print(f"Job submitted: {match.group(1)}. Waiting for start...")
    return match.group(1), log_file

def wait_for_connection_info(log_file: Path, job_id: str) -> str:
    """Polls the log file for the compute node name and the Jupyter token.
    Raises RuntimeError with the log contents if the job fails."""
    print(f"Watching log file: {log_file}")

    while True:
        if log_file.exists():
            content = log_file.read_text()

            # Check for token
            token_match = re.search(r"token=([a-z0-9]+)", content)
            if token_match:
                return token_match.group(1)

            # Check if the job is still running
            job_state = subprocess.run(
                f"squeue -j {job_id} -h -o %T",
                shell=True, capture_output=True, text=True
            ).stdout.strip()

            if job_state not in ("RUNNING", "PENDING", ""):
                raise RuntimeError(
                    f"Slurm job {job_id} ended with state '{job_state}'. "
                    f"Log output:\n{content}"
                )

            # Job is gone from squeue entirely but no token found
            if not job_state and content:
                raise RuntimeError(
                    f"Slurm job {job_id} exited before Jupyter started. "
                    f"Log output:\n{content}"
                )

        time.sleep(2)

def get_running_node(job_id: str) -> str | None:
    """Checks squeue to see if the job is running and returns the node."""
    cmd = f"squeue -j {job_id} -h -o %N"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    node = res.stdout.strip()
    return node if node else None

def start_tunnel(compute_node: str, remote_port: int, local_port: int) -> None:
    """Sets up an SSH tunnel in the background."""
    print(f"Tunneling {compute_node}:{remote_port} -> localhost:{local_port}...")
    # -f: go to background, -N: do not execute remote command
    cmd = ["ssh", "-f", "-N", "-L", f"{local_port}:localhost:{remote_port}", compute_node]
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Launch Jupyter on Slurm and auto-tunnel.")
    parser.add_argument("--time", default="02:00:00", help="Wall time")
    parser.add_argument("--qos", default=None, help="QOS name")
    parser.add_argument("--gpus", default="1", help="Number of GPUs")
    parser.add_argument("--mem", default="16G", help="Memory")
    parser.add_argument("--env", default=None, help="The mamba environment to activate. Defaults to ./.env")
    args = parser.parse_args()

    env_name = args.env or str(Path.cwd() / '.env')
    
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Submit Job
    qos_flag = f" --qos={args.qos}" if args.qos else ""
    slurm_flags = f"--time={args.time} --gpus-per-node={args.gpus} --mem={args.mem}{qos_flag}"
    job_id, log_file = submit_slurm_job(slurm_flags, DEFAULT_PORT, env_name)
    
    # 2. Wait for Allocation
    print("Waiting for node allocation...")
    node_name = None
    while not node_name:
        node_name = get_running_node(job_id)
        time.sleep(3)
    
    print(f"Allocated Node: {node_name}")
    
    # 3. Establish Tunnel immediately (even before Jupyter is fully up, the tunnel is valid)
    # Kill old SSH tunnels on this port to prevent conflicts
    subprocess.run(f"lsof -ti:{DEFAULT_PORT} -sTCP:LISTEN | xargs -r kill", shell=True, stderr=subprocess.DEVNULL)
    start_tunnel(node_name, DEFAULT_PORT, DEFAULT_PORT)
    
    # 4. Wait for Token
    token = wait_for_connection_info(log_file, job_id)
    
    # 5. Output for VS Code
    url = f"http://127.0.0.1:{DEFAULT_PORT}/?token={token}"
    print("\n" + "="*100)
    print(f"Server connected via tunnel successfully.")
    print(f"URL: {url}")
    print("="*100 + "\n")

if __name__ == "__main__":
    main()