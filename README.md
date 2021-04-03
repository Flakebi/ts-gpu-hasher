# GPU TS Identity Hasher

Increase the level of TeamSpeak identities faster by using GPUs.

TeamSpeak uses a HashCash-like mechanism to prevent people from creating too many new identities and
spamming servers. As hashing is embarrassingly parallel, we can use GPUs to make it faster!

## How does it work?

Inspired by [TeamSpeakHasher](https://github.com/landave/TeamSpeakHasher) that uses OpenCL,
this project runs Rust on the GPU. Using Rust, we can just use the [sha-1 crate](https://crates.io/crates/sha-1)
and compile it for the GPU.

## Performance

On my RX 480 (gfx803), it runs with 2 GH/s (gigahashes per second). The TeamSpeakHasher reaches 400 MH/s.
One CPU core gets 7 MH/s on an i7-5820k.

## Limitations

Because of ~~lazyness~~ performance reasons, only a single SHA-1 block is hashed and things break
when the offset gets too high (called slow-mode in TeamSpeakHasher).

Due to leveraging the `amdgpu` LLVM backend and the ROCm runtime, only AMD GPUs and only Linux are
supported. A hardware agnostic standard like SPIR-V would be nice, but imposes restrictions[¹](https://github.com/EmbarkStudios/rust-gpu/issues/234#issuecomment-726629418)
that are hard to circumvent without sacrificing performance.

## Dependencies

- A fitting OS and GPU (only AMD on Linux is supported)
- A [geobacter-rs](https://github.com/geobacter-rs/geobacter) toolchain, which means
you need to compile a Rust toolchain yourself.
- The ROCm HSA runtime

## Usage

Run
```
cargo run --release -- -k <public key>
# E.g.
cargo run --release -- -k MEsDAgcAAgEgAiB9zjIHIFBmRZyuwajGa7XfWIUieIvDtE1j837eZvVm0AIgGiWDZPjGrMsigOFt4HgRnn9IwH2HIMhU4pzGiulAQUc=
```

To run on the CPU, add `--cpu` (this mode is mostly for debugging and doesn’t use parallelism).
