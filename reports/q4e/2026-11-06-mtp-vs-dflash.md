# Native MTP vs DFlash2 on 4 x Arc Pro B70

2026-11-06. Qwen3.8-Flash-Next, Intel W4A16 AutoRound, torch 2.15.0, Triton 3.9.0, compute-runtime 26.35, IGC 2.41.5, FULL_AND_PIECEWISE, chunk 4,096, `--no-async-scheduling` on the MTP arms. Same prompts and stages on every arm. Single runs; differences under about 10% are noise. Decode is tok/s. Part of the [q4e test report](2026-11-05-first-results.md).

## Result

1. DFlash2 K=7 is still the faster drafter on this hardware, clearly on code and on 1-2 streams.
2. MTP K=3 comes close on deep context and with many streams, needs no second model, and its temperature-0 output was bit-identical on the longest prompt, which isn't reported for DFlash2.
3. Raising MTP to K=7 trades prose speed and KV for code speed, and doesn't reach DFlash2.

## Same tree, dense: D16-q4e-8

Only the speculative method differs.

| | DFlash2 K=7 | MTP K=3 |
|---|---|---|
| Decode, prose | 115 | 95 |
| Acceptance, prose | 1.42 of 7 | 1.26 of 3 |
| Decode, code | 264 | 150 |
| Acceptance, code | 4.62 of 7 | 2.56 of 3 |
| Decode at 16K / 61K / 125K | 120 / 87 / 83 | 104 / 97 / 83 |
| Step | 23-27 ms | 25-29 ms |
| Total at 8K, 1/2/3/8/12 streams | 196/277/223/212/207 | 133/186/183/189/187 |
| Total at 32K, 1/2/3/8/12 streams | 200/86/67/52/50 | 129/75/62/49/47 |
| Prefill, 8K / 42K | 6,510 / 6,150 | 6,280 / 5,950 |
| TTFT at 125K | 26.2 s | 27.6 s |
| KV | 384,123 tokens | 273,868 tokens |
| Model per rank | 19.03 GiB | 20.28 GiB |
| Host memory | 214 GiB | 212 GiB |
| Temperature 0, one text | 5 and 30 words | 5, 30 and 1,500 words |

Both arms: needle, tools and burst 0/150 pass.

## Sparse

The MTP K=3 and K=7 columns are on zhaohb's branch with the PRs and fused hyper-connections (S16-Z2h), at 131,072 context. DFlash2 is the kit's engine on the same branch base (S16-q4e-8), at 262,144 context, so KV isn't directly comparable and other engine differences may contribute.

| | DFlash2 K=7, S16-q4e-8 | MTP K=3, S16-Z2h | MTP K=7, S16-Z2h |
|---|---|---|---|
| Decode, prose | 108 | 102 | 83 |
| Acceptance, prose | 1.46 of 7 | 1.35 of 3 | 1.43 of 7 |
| Decode, code | 236 | 159 | 194 |
| Acceptance, code | 4.36 of 7 | 2.67 of 3 | 4.68 of 7 |
| Decode at 16K / 61K / 125K | 102 / 84 / 81 | 108 / 107 / 117 | 99 / 105 / 101 |
| Step | 25-26 ms | 23.6-23.9 ms | 31.1-31.2 ms |
| Total at 8K, 1/2/3/8/12 streams | 171/249/215/186/182 | 137/187/182/165/164 | 139/186/182/135/129 |
| Total at 32K, 1/2/3/8/12 streams | 200/79/61/49/47 | 135/66/54/42/41 | 153/66/54/41/40 |
| Prefill, 8K / 42K | 5,670 / 5,610 | 5,300 / 4,840 | 5,300 / 4,830 |
| TTFT at 125K | 23.1 s | 31.6 s | 31.6 s |
| KV | 359,038 tokens | 246,034 tokens | 175,434 tokens |

All arms: needle, burst 0/150, 0 resets. K=7 acceptance on code is about the same as DFlash2's, so the code gap at that K points to the step, 31 against 25 ms.

## Notes

1. **MTP K=5 to K=8 needs a fix.** On zhaohb's branch, K=5 and K=7 refuse at boot with `QSA ring capacity 12 must divide the attention block size 832`. The K=7 column ran with the widening from vLLM #54912 on a scratch tree.
2. **The kit's own sparse MTP is weaker.** MTP K=3 on S16-q4e-8 gave prose 74, code 96, step 31.5-32.7 ms and acceptance 1.61 overall, at 240,000 context, because 262,144 refused KV sizing. The step gap suggests a missing fused path in that draft layer, but this hasn't been timed. The acceptance gap is unexplained. S16-Z2h is the fair sparse MTP number.
3. **`index_share_for_mtp_iteration`** isn't set in this checkpoint. Turned on at K=3, it fires, and step and acceptance stay within noise.
4. **KV falls as MTP K rises**, from 246,034 tokens at K=3 to 175,434 at K=7 in the same tree, and MTP holds far fewer KV tokens than DFlash2 at the same K. The cause isn't established.
