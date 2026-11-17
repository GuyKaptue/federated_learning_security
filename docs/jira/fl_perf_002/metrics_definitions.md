Resource and communication monitoring.

Metric definitions (all MEASURED on the executing machine; this file):

* wall_s            = t_end - t_start (perf_counter)
* cpu_time_s        = (user+system CPU seconds of this process, children included) at end minus start
                      -> cumulative resource consumption
* cpu_util_mean_pct = mean over samples of process CPU% (100% = one fully used core; may exceed 100)
* cpu_util_peak_pct = max over samples
* rss_mean_mb / rss_peak_mb = mean / max resident set size of the process over samples (MiB)
* comm bytes        = logical serialized payload bytes counted by the protocol code (not NIC traffic,
                      since all virtual clients share one machine)
* throughput_samples_per_s = training samples processed by clients / client compute seconds
* GPU               = recorded as 'unavailable' when no NVIDIA tooling is present (never estimated)

