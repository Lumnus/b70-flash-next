# Baseline: vLLM + this series vs llama.cpp on 4× Arc Pro B70

Same prompts and harness; llama.cpp serving the Unsloth UD-Q4_K_XL GGUF with layer split across the 4 cards.

| | llama.cpp (layer split) | vLLM, chunk 256 (wu1ff default) | vLLM, chunk 1024 |
|---|---|---|---|
| decode, 1 stream | 19.5 tok/s | 58.7 tok/s | ~59 tok/s (unchanged) |
| prefill at 18.7K tokens | 548 tok/s | 2,674 tok/s | **4,711 tok/s** |
| prefill at 98K tokens | 354 tok/s | 1,686 tok/s | **3,711 tok/s** |
| 4 streams, aggregate decode | 26.1 tok/s | 193.7 tok/s | ~197 tok/s |
| MMLU / TruthfulQA (paired) | — | not significantly different from llama.cpp | unchanged (p = 1.0) |

`--max-num-batched-tokens 1024` (with capture sizes 512 and 1024 added) nearly doubles long-context prefill; the
256-token chunk was the limiter.

Other findings on this hardware:

1. MTP speculative decoding (3 tokens): single-stream decode +34 % (78.5 tok/s), aggregate −34 %. Worth it only for
   single-user latency.
2. The Triton attention backend boots and captures graphs but prefills 7× slower at 18.7K and 29× slower at 98K.
3. compute-runtime 26.35.39758.10 vs the image's 26.27: decode +2.8 %, 4-stream aggregate +8.7 %, +4.5 GiB host
   MemAvailable at `--gpu-memory-utilization 0.85`; 0.90 fails vLLM's startup free-memory check on 26.35 (the driver
   reserves ~1.3 GiB more per card).
4. Keep cudagraph capture sizes to powers of two. With non-power-of-two sizes some requests in the minutes after a
   boot emitted token 1023 from the first token (a NaN loop; with logprobs requested the server answers HTTP 400
   "Out of range float values are not JSON compliant: nan"). With `[1,2,4,8,256,512,1024]` the same items were 60/60
   clean. `B70_DEFAULT_REPETITION_DETECTION` (0010) bounds the loop if it happens anyway.
5. The W4A16 checkpoint refuses prompts that ask it to disclose a "secret"; use neutral needles for retrieval tests.
