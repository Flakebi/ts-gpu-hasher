#![feature(asm, geobacter, llvm_asm)]

extern crate grt_amd as geobacter_runtime_amd;

use std::io::Write;
use std::ops::*;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use clap::Clap;
use generic_array::GenericArray;
use grt_amd::alloc::*;
use grt_amd::module::*;
use grt_amd::prelude::*;
use grt_amd::{GeobacterDeps, HsaAmdGpuAccel};
use grt_core::context::Context;
use lexical_core::ToLexical;
use sha1::{Digest, Sha1};
use tsproto_types::crypto::EccKeyPrivP256;

const SHA1_BLOCK_SIZE: usize = 64;
const SHA1_STATE_SIZE: usize = 5;
/// SHA-1 initial state
const SHA1_INIT_STATE: [u32; SHA1_STATE_SIZE] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];
/// SHA-1 padding is 0x80, more zeros and message length as u64
const SHA1_PADDING: usize = 9;

// Optimization parameter, tuned for gfx803
const COUNT: usize = 1024 * 16;
const PER_THREAD_COUNT: u64 = 1000;
const WG_SIZE: usize = 256;

#[derive(Clone, Debug, clap::Clap)]
#[clap(version = clap::crate_version!(), author = clap::crate_authors!(),
	about = clap::crate_description!())]
struct Options {
	/// The public key to compute the level.
	#[clap(short, long)]
	key: Option<String>,
	/// Start offset
	#[clap(short, long, default_value = "0")]
	offset: u64,
	#[clap(short, long)]
	cpu: bool,
}

#[derive(Clone, Debug, Default)]
struct Entry {
	offset: u64,
	level: u8,
}

fn format_num_separated<T: ToString>(n: T) -> String {
	let mut res = String::new();
	let s = n.to_string();
	let mut pos = s.len();
	let mut not_first = false;

	for c in s.chars() {
		if not_first && pos % 3 == 0 {
			res.push(',');
		}

		res.push(c);
		not_first = true;
		pos -= 1;
	};

	res
}

fn get_hash_cash_level(hash: [u32; SHA1_STATE_SIZE]) -> u8 {
	// Clamp to zero if level is < 32
	// if hash[0] != 0 {
	if hash[0] & 0xffff0000 != 0 {
		return 0;
	}
	let mut res = 0;
	for &d in &hash {
		if d == 0 {
			res += std::mem::size_of_val(&d) as u8 * 8;
		} else {
			// Get trailing zeros per byte
			res += get_hash_cash_level_accurate(&d.to_be_bytes());
			break;
		}
	}
	res
}

fn get_hash_cash_level_accurate(hash: &[u8]) -> u8 {
	let mut res = 0;
	for &d in hash {
		if d == 0 {
			res += std::mem::size_of_val(&d) as u8 * 8;
		} else {
			res += d.trailing_zeros() as u8;
			break;
		}
	}
	res
}

fn get_hash_cash_level_from_str(omega: &[u8], offset: u64) -> u8 {
	let mut hasher = Sha1::new();
	hasher.update(omega);
	hasher.update(offset.to_string().as_bytes());
	let r = hasher.finalize();
	get_hash_cash_level_accurate(&r)
}

fn find_best_level(state: [u32; SHA1_STATE_SIZE], omega: &[u8], msg_len: u64, start: u64) -> Entry {
	// sha1(<omega><offset>)
	let mut block = GenericArray::default();
	block[..omega.len()].copy_from_slice(omega);
	let formatted = start.to_lexical(&mut block[omega.len()..]);
	let len = omega.len() + formatted.len();
	let total_len = (msg_len + formatted.len() as u64) * 8;
	// Set last byte to 0x80 for SHA-1 padding, the rest is initialized with 0
	block[len] = 0x80;
	// Append total length in the end
	block[SHA1_BLOCK_SIZE - 8..].copy_from_slice(&total_len.to_be_bytes());

	let mut best = Entry { offset: start, level: 0 };
	for i in start..(start + PER_THREAD_COUNT) {
		// Only change the suffix of the number
		/*if i <= 1000 {
			// Does not work for first block (starting with 0)
			let formatted = i.to_lexical(&mut block[omega.len()..]);
			len = omega.len() + formatted.len();
			total_len = (msg_len + formatted.len() as u64) * 8;

			// Padding
			block[len] = 0x80;
			block[SHA1_BLOCK_SIZE - 8..].copy_from_slice(&total_len.to_be_bytes());
		}*/

		let mut new_state = state;
		sha1::compress(&mut new_state, &[block]);
		let level = get_hash_cash_level(new_state);
		if level > best.level {
			best.offset = i;
			best.level = level;
		}

		// Increment
		block[len - 1] += 1;
		if block[len - 1] == b'9' + 1 {
			block[len - 1] = b'0';
			block[len - 2] += 1;
			if block[len - 2] == b'9' + 1 {
				block[len - 2] = b'0';
				block[len - 3] += 1;
			}
		}
	}
	best

	// Inline assembly
	/*unsafe {
		/*#[cfg(not(target_arch = "x86_64"))]
		asm!(
			"v_mul_f32 {r}, {r}, 2.0",
			r = inout(vgpr) r,
		);*/
		llvm_asm!(
			"v_mul_f32 $0, $1, 2.0"
			: "=v"(r)
			: "v"(r)
			:
			:
		);
	}*/
}

#[repr(C)] // Ensure we have a universally understood layout
#[derive(GeobacterDeps)]
pub struct Args {
	// copy: DeviceSignal,
	omega: *const [u8],
	state: *const [u32],
	msg_len: u64,
	offset: u64,
	result_level: *mut [u8],
	result_offset: *mut [u64],
	queue: Rc<DeviceSingleQueue>,
	completion: GlobalSignal,
}
impl Completion for Args {
	type CompletionSignal = GlobalSignal;
	fn completion(&self) -> &GlobalSignal { &self.completion }
}
impl Kernel for Args {
	type Grid = Dim1D<Range<u32>>;
	const WORKGROUP: <Self::Grid as GridDims>::Workgroup = Dim1D { x: ..WG_SIZE as _ };
	type Queue = DeviceSingleQueue;

	fn queue(&self) -> &Self::Queue { self.queue.as_ref() }

	/// This is the kernel that is run on the GPU
	fn kernel(&self, vp: KVectorParams<Self>)
	where Self: Sized {
		//use std::geobacter::amdgpu::workitem::ReadFirstLane;
		// These globals are in LDS (workgroup local) memory.
		lds! {
			let mut lds_level: Lds<[u8; WG_SIZE]> = Lds::new();
			let mut lds_offset: Lds<[u64; WG_SIZE]> = Lds::new();
		}

		let omega = if let Some(omega) = self.omega_view() {
			omega
		} else {
			return;
		};

		let state = if let Some(state) = self.state_view() {
			let mut s = [0; SHA1_STATE_SIZE];
			s.copy_from_slice(state);
			s
		} else {
			return;
		};

		// Global id
		let idx = vp.gl_id();
		// Number of workgroup
		let wg_idx = vp.wg_id().x;
		let start = self.offset + idx as u64 * PER_THREAD_COUNT;
		let mut best = find_best_level(state, omega, self.msg_len, start);

		// TODO We can search more efficiently on a SIMD

		// Index inside the workgroup
		let wi = vp.wi();
		// Binary fan in for maximum level
		let log2 = WG_SIZE.next_power_of_two().trailing_zeros();
		for l in 0..log2 {
			let step_diff = 1 << l;
			if wi.x % step_diff == 0 {
				lds_level.with_shared(|mut lds_level| {
					lds_offset.with_shared(|mut lds_offset| {
						let lds_level_slice = lds_level.init(&vp, best.level);
						let lds_offset_slice = lds_offset.init(&vp, best.offset);
						if wi.x % (step_diff << 1) == 0 {
							let step_diff_dim = Dim1D { x: step_diff };
							if lds_level_slice[wi + step_diff] > best.level {
								best.level = lds_level_slice[wi + step_diff_dim];
								best.offset = lds_offset_slice[wi + step_diff_dim];
							}
						}
					});
				});
			}
		}

		if vp.is_wi0() {
			if let (Some(level), Some(offset)) =
				(self.result_level_view(), self.result_offset_view())
			{
				// Store the result in the first thread of each workgroup
				if let (Some(dest_level), Some(dest_offset)) =
					(level.get_mut(wg_idx as usize), offset.get_mut(wg_idx as usize))
				{
					// Load previous maximum
					if best.level > *dest_level {
						*dest_level = best.level;
						*dest_offset = best.offset;
					}
				}
			}
		}
	}
}

impl Args {
	pub fn omega_view(&self) -> Option<&[u8]> { unsafe { self.omega.as_ref() } }
	pub fn state_view(&self) -> Option<&[u32]> { unsafe { self.state.as_ref() } }
	pub fn result_level_view(&self) -> Option<&mut [u8]> { unsafe { self.result_level.as_mut() } }
	pub fn result_offset_view(&self) -> Option<&mut [u64]> {
		unsafe { self.result_offset.as_mut() }
	}
}
unsafe impl Send for Args {}
unsafe impl Sync for Args {}

pub fn time<F, R>(what: &str, f: F) -> R
where F: FnOnce() -> R {
	let start = Instant::now();
	let r = f();
	let elapsed = start.elapsed();
	println!("{} took {}ms ({}μs)", what, elapsed.as_millis(), elapsed.as_micros());

	r
}

pub fn main() {
	use tracing_subscriber::prelude::*;
	use tracing_subscriber::{fmt, EnvFilter};

	let fmt_layer = fmt::layer().with_target(false);
	let filter_layer =
		EnvFilter::try_from_default_env().or_else(|_| EnvFilter::try_new("info")).unwrap();

	tracing_subscriber::registry().with(filter_layer).with(fmt_layer).init();

	let options: Options = Options::parse();
	let omega = if let Some(key) = options.key {
		key
	} else {
		let key = EccKeyPrivP256::create();
		let priv_key = key.to_ts().expect("Cannot convert private key");
		let omega = key.to_pub().to_ts().expect("Cannot convert public key");
		println!("Generated private key {}", priv_key);
		println!("Generated public key {}", omega);
		omega
	};

	// SHA-1 state and the prefix of the block that has to be hashed on the GPU
	let (state, prefix) = {
		let mut state = SHA1_INIT_STATE;
		for block in omega.as_bytes().chunks(SHA1_BLOCK_SIZE) {
			if block.len() == SHA1_BLOCK_SIZE {
				let block = GenericArray::clone_from_slice(block);
				sha1::compress(&mut state, &[block]);
			}
		}
		(state, &omega[omega.len() / SHA1_BLOCK_SIZE * SHA1_BLOCK_SIZE..])
	};

	// Make a multiple of PER_THREAD_COUNT
	let mut offset = options.offset / PER_THREAD_COUNT * PER_THREAD_COUNT;
	// TODO Starts at PER_THREAD_COUNT, so we don't need to special case the beginning
	if offset < PER_THREAD_COUNT {
		offset = PER_THREAD_COUNT
	}
	let max = if options.cpu {
		cpu(&mut offset, state, prefix, &omega)
	} else {
		gpu(&mut offset, state, prefix, &omega)
	};

	println!("Maximum level {} at offset {}, next offset {}", max.level, format_num_separated(max.offset), format_num_separated(offset));

	let level = get_hash_cash_level_from_str(omega.as_bytes(), max.offset);
	assert_eq!(level, max.level, "Computed hash cash level is wrong");
}

fn run<F: FnMut(u64)>(offset: &mut u64, prefix: &str, mut f: F) {
	let running = Arc::new(AtomicBool::new(true));
	let r = running.clone();
	ctrlc::set_handler(move || {
		r.store(false, Ordering::Relaxed);
	}).expect("Error setting Ctrl-C handler");

	let start = Instant::now();
	let mut last_print = start;
	let mut last_print_offset = *offset;
	let mut last_print_invocs = 0;
	while running.load(Ordering::Relaxed) {
		let end_offset = *offset + COUNT as u64 * PER_THREAD_COUNT;
		if prefix.len() + end_offset.to_string().len() > SHA1_BLOCK_SIZE - SHA1_PADDING {
			println!("Counter gets larger than sha-1 block size, aborting");
			break;
		}

		f(*offset);

		*offset = end_offset;
		last_print_invocs += 1;

		// Print stats
		let now = Instant::now();
		let dur = now.duration_since(last_print);
		if dur > Duration::from_secs(2) {
			let hashes = ((*offset - last_print_offset) as f32 / dur.as_secs_f32()) as u32;
			/*print!(
				"\rMaximum level {} at offset {} | {} H/s ({} invocations)",
				max.0, max.1, hashes, last_print_invocs
			);*/
			const PREFIXES: [&str; 5] = ["", "k", "M", "G", "T"];
			let (prefix, hashes_f) = {
				let p = std::cmp::min(PREFIXES.len() - 1, (hashes as f32).log10() as usize / 3);
				(PREFIXES[p], hashes as f32 / (10u32.pow(p as u32 * 3) as f32))
			};
			print!(
				"\r{:.2} {}H/s ({} invocations) | At offset {}",
				hashes_f, prefix, last_print_invocs, format_num_separated(*offset)
			);
			std::io::stdout().flush().expect("Failed to flush stdout");
			last_print = now;
			last_print_offset = *offset;
			last_print_invocs = 0;
		}
	}
	println!();
}

fn gpu(offset: &mut u64, state: [u32; SHA1_STATE_SIZE], prefix: &str, omega: &str) -> Entry {
	let ctxt = time("create context", || Context::new().expect("create context"));

	let accels = HsaAmdGpuAccel::all_devices(&ctxt).expect("HsaAmdGpuAccel::all_devices");
	if accels.len() < 1 {
		panic!("no accelerator devices???");
	}
	let accel = accels.first().unwrap();
	println!("Picking first device out of {}: {}", accels.len(), accel.agent().name().unwrap());

	let mut invoc: FuncModule<Args> = FuncModule::new(&accel);
	invoc.compile_async();
	unsafe {
		invoc.no_acquire_fence();
		invoc.device_release_fence();
	}

	// Lap = locally accessible pool
	let lap_alloc = accel.fine_lap_node_alloc(0);
	let mut omega_buf: LapVec<u8> =
		time("alloc omega_buf", || LapVec::with_capacity_in(prefix.len(), lap_alloc.clone()));
	time("Fill omega_buf", || omega_buf.extend_from_slice(prefix.as_bytes()));
	let mut state_buf: LapVec<u32> =
		time("alloc state_buf", || LapVec::with_capacity_in(state.len(), lap_alloc.clone()));
	time("Fill state_buf", || state_buf.extend_from_slice(&state));
	let mut result_level_buf: LapVec<u8> = time("alloc result_level_buf", || {
		LapVec::with_capacity_in(COUNT / WG_SIZE, lap_alloc.clone())
	});
	time("Fill result_level_buf", || {
		result_level_buf.resize_with(COUNT / WG_SIZE, Default::default)
	});
	let mut result_offset_buf: LapVec<u64> = time("alloc result_offset_buf", || {
		LapVec::with_capacity_in(COUNT / WG_SIZE, lap_alloc.clone())
	});
	time("Fill result_offset_buf", || {
		result_offset_buf.resize_with(COUNT / WG_SIZE, Default::default)
	});

	let mut device_omega_ptr = time("alloc device slice", || unsafe {
		accel
			.alloc_device_local_slice::<u8>(omega_buf.len())
			.expect("HsaAmdGpuAccel::alloc_device_local")
	});
	let mut device_state_ptr = time("alloc device slice", || unsafe {
		accel
			.alloc_device_local_slice::<u32>(state_buf.len())
			.expect("HsaAmdGpuAccel::alloc_device_local")
	});
	let mut device_result_level_ptr = time("alloc device slice", || unsafe {
		accel
			.alloc_device_local_slice::<u8>(result_level_buf.len())
			.expect("HsaAmdGpuAccel::alloc_device_local")
	});
	let mut device_result_offset_ptr = time("alloc device slice", || unsafe {
		accel
			.alloc_device_local_slice::<u64>(result_offset_buf.len())
			.expect("HsaAmdGpuAccel::alloc_device_local")
	});

	println!("host ptr: 0x{:p}, agent ptr: 0x{:p}", omega_buf.as_ptr(), device_omega_ptr);

	let async_copy_signal0 =
		accel.new_host_signal(1).expect("HsaAmdGpuAccel::new_device_signal: async_copy_signal");
	let async_copy_signal1 =
		accel.new_host_signal(1).expect("HsaAmdGpuAccel::new_device_signal: async_copy_signal");
	let mut results_signal0 =
		accel.new_host_signal(1).expect("HsaAmdGpuAccel::new_host_signal: results_signal");
	let mut results_signal1 =
		accel.new_host_signal(1).expect("HsaAmdGpuAccel::new_host_signal: results_signal");

	time("grant gpu access to buffers", || {
		omega_buf.add_access(&*accel).expect("grant_agents_access");
		state_buf.add_access(&*accel).expect("grant_agents_access");
		result_level_buf.add_access(&*accel).expect("grant_agents_access");
		result_offset_buf.add_access(&*accel).expect("grant_agents_access");
	});

	unsafe {
		accel
			.unchecked_async_copy_into(&omega_buf, &mut device_omega_ptr, &(), &async_copy_signal0)
			.expect("HsaAmdGpuAccel::async_copy_into");
		accel
			.unchecked_async_copy_into(&state_buf, &mut device_state_ptr, &(), &async_copy_signal1)
			.expect("HsaAmdGpuAccel::async_copy_into");
		accel
			.unchecked_async_copy_into(&result_level_buf, &mut device_result_level_ptr, &(), &results_signal0)
			.expect("HsaAmdGpuAccel::async_copy_into");
		accel
			.unchecked_async_copy_into(&result_offset_buf, &mut device_result_offset_ptr, &(), &results_signal1)
			.expect("HsaAmdGpuAccel::async_copy_into");
	}

	time("cpu -> gpu async copy", || {
		async_copy_signal0.wait_for_zero(false).expect("unexpected signal status");
		async_copy_signal1.wait_for_zero(false).expect("unexpected signal status");
		results_signal0.wait_for_zero(false).expect("unexpected signal status");
		results_signal1.wait_for_zero(false).expect("unexpected signal status");
	});

	let group_size = invoc.group_size().expect("codegen failure");
	let private_size = invoc.private_size().unwrap();
	let queue = Rc::new(
		accel
			.create_single_queue2(None, group_size, private_size)
			.expect("HsaAmdGpuAccel::create_single_queue"),
	);

	let mut args_pool =
		time("alloc kernargs pool", || ArgsPool::new::<Args>(&accel, 1).expect("ArgsPool::new"));

	let grid = Dim1D { x: 0u32..COUNT as _ };

	time("compiling", || invoc.compile().expect("codegen failed"));

	run(offset, prefix, |offset| {
		let mut invoc = invoc.invoc(&args_pool);

		let kernel_signal =
			GlobalSignal::new(1).expect("HsaAmdGpuAccel::new_host_signal: kernel_signal");
		let args = Args {
			offset,
			msg_len: omega.len() as u64,
			omega: device_omega_ptr.as_ptr(),
			state: device_state_ptr.as_ptr(),
			result_level: device_result_level_ptr.as_ptr(),
			result_offset: device_result_offset_ptr.as_ptr(),
			completion: kernel_signal,
			queue: queue.clone(),
		};

		let wait = /*time("dispatching", ||*/ unsafe {
			invoc.unchecked_call_async(&grid, args).expect("Invoc::call_async")
		}; //);

		// specifically wait (without enqueuing another async copy) here
		// so we can time just the dispatch.
		//time("dispatch wait", move || {
		wait.wait_for_zero(false).expect("wait for zero failed");
		//});

		drop(wait);
		args_pool.wash();
	});

	// now copy the results back to the locked memory:
	results_signal0.reset(accel, 1).expect("Failed to reset signal");
	results_signal1.reset(accel, 1).expect("Failed to reset signal");
	unsafe {
		accel
			.unchecked_async_copy_from(
				&device_result_level_ptr,
				&mut result_level_buf,
				&(),
				&results_signal0,
			)
			.expect("HsaAmdGpuAccel::async_copy_from");
		accel
			.unchecked_async_copy_from(
				&device_result_offset_ptr,
				&mut result_offset_buf,
				&(),
				&results_signal1,
			)
			.expect("HsaAmdGpuAccel::async_copy_from");
	}
	time("gpu -> cpu async copy", || {
		results_signal0.wait_for_zero(false).expect("unexpected signal status");
		results_signal1.wait_for_zero(false).expect("unexpected signal status");
	});

	let max = result_level_buf.iter().copied().zip(result_offset_buf.iter().copied())
		.max_by_key(|(l, _)| *l).expect("Empty result buffer");
	Entry {
		level: max.0,
		offset: max.1,
	}
}

fn cpu(offset: &mut u64, state: [u32; SHA1_STATE_SIZE], prefix: &str, omega: &str) -> Entry {
	let mut max = Entry::default();

	run(offset, prefix, |offset| {
		let r = find_best_level(state, prefix.as_bytes(), omega.len() as u64, offset);
		if r.level > max.level {
			max = r;
		}
	});
	max
}
