"""Find idle deployments and prune them (only resources Skyhammer created are touched).

Cost: free. Run: uv run python examples/23_manage_deployments.py
CLI: skyhammer deployments list | skyhammer deployments prune --idle-days 7
"""

from skyhammer.deployments import list_deployments
from skyhammer.session import get_session

for d in list_deployments(get_session(), idle_days=7):
    idle = "IDLE" if d.invocations == 0 else f"{d.invocations} calls"
    print(f"{d.kind:26} {d.name:30} {d.status:10} {idle}")
