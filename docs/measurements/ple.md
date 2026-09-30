# PLE table: formats and placement

| configuration | pinned host RAM for the table | boot host-RAM peak | quality vs BF16 | speed vs BF16 |
|---|---|---|---|---|
| BF16, wu1ff loader | 95.4 GiB | ~191 GiB (4 ranks load a pageable copy each) | — | — |
| BF16 + 0006 (direct to pinned) | 95.4 GiB | ~158 GiB | same table bytes | same table bytes |
| FP8 + 0007 | 47.7 GiB | — | rel-L2 2.65 %; MMLU/TQA n.s. vs INT8 and BF16 | not measured separately |
| **INT8 per row + 0008** | **48.9 GiB** | — | rel-L2 0.66 %; MMLU/TQA/needles n.s.; 0 top-1 or answer changes | equal |
| INT8 on NVMe + 0013/0013b | 8 GiB row cache (2 per rank) | — | KL at the INT8 noise floor | decode −2 … −4 %, prefill equal |

Details: [../ple-int8.md](../ple-int8.md), [../ple-nvme.md](../ple-nvme.md). The FP8 row used the table cut from the
official FP8 checkpoint (`Qwen/Qwen3.8-Flash-Next-FP8`); its format study on the BF16 table gives the same 2.65 %.
Pinned memory is not counted against a container memory limit; watch the host's MemAvailable at boot.
