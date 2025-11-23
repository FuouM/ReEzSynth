import contextlib
import time
from collections import defaultdict


class SynthesisTimer:
    """Simple timer for benchmarking synthesis operations."""

    def __init__(self):
        self.timing_data = defaultdict(float)
        self.call_counts = defaultdict(int)

    def reset(self):
        """Reset all timing data."""
        self.timing_data.clear()
        self.call_counts.clear()

    @contextlib.contextmanager
    def time_operation(self, operation_name: str):
        """Context manager to time an operation."""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.timing_data[operation_name] += elapsed
            self.call_counts[operation_name] += 1

    def get_summary(self) -> dict:
        """Get timing summary."""
        summary = {}
        total_time = sum(self.timing_data.values())

        for op, time_spent in self.timing_data.items():
            summary[op] = {
                "time": time_spent,
                "calls": self.call_counts[op],
                "avg_time": time_spent / self.call_counts[op]
                if self.call_counts[op] > 0
                else 0,
                "percentage": (time_spent / total_time * 100) if total_time > 0 else 0,
            }

        summary["_total"] = total_time
        return summary

    def print_summary(self, title: str = "Synthesis Timing Summary"):
        """Print formatted timing summary."""
        summary = self.get_summary()

        print(f"\n{title}")
        print("=" * len(title))

        # Sort by time spent (descending)
        sorted_ops = sorted(
            [(k, v) for k, v in summary.items() if k != "_total"],
            key=lambda x: x[1]["time"],
            reverse=True,
        )

        total_time = summary["_total"]

        for op, data in sorted_ops:
            time_str = f"{data['time']:.4f}s"
            pct_str = f"{data['percentage']:.1f}%"
            calls_str = f"{data['calls']} calls"
            avg_str = f"{data['avg_time']:.4f}s avg" if data["calls"] > 1 else ""

            print(f"  {op:<25} {time_str:>10} ({pct_str:>5}) {calls_str:>10} {avg_str}")

        print(f"  {'TOTAL':<25} {total_time:.4f}s")
