#![feature(asm, geobacter, llvm_asm)]

extern crate grt_amd as geobacter_runtime_amd;

use std::mem::size_of;
use std::ops::*;
use std::time::Instant;

use clap::Clap;
use grt_amd::alloc::*;
use grt_amd::module::*;
use grt_amd::prelude::*;
use grt_amd::{GeobacterDeps, HsaAmdGpuAccel};
use grt_core::context::Context;
use lexical_core::ToLexical;
use sha1::{Digest, Sha1};
use tsproto_types::crypto::EccKeyPubP256;

const COUNT: usize = 1024 * 4;
const PER_THREAD_COUNT: u64 = 10;

const WG_SIZE: usize = 256;

#[derive(Clone, Debug, clap::Clap)]
#[clap(version = clap::crate_version!(), author = clap::crate_authors!(),
	about = clap::crate_description!())]
struct Options {
	/// The public key to compute the level.
	///
	/// Provide either key or uid.
	#[clap(short, long)]
	key: Option<String>,
	// TODO Should not be the uid, but the public key…
	/// The uid to compute the level.
	///
	/// Provide either key or uid.
	#[clap(short, long)]
	uid: Option<String>,
	/// Start offset
	#[clap(short, long, default_value = "0")]
	offset: u64,
}

struct Entry {
	offset: u64,
	level: u8,
}

fn get_hash_cash_level(hash: &[u8]) -> u8 {
	let mut res = 0;
	for &d in hash {
		if d == 0 {
			res += 8;
		} else {
			res += d.trailing_zeros() as u8;
			break;
		}
	}
	res
}

fn find_best_level(omega: &[u8], start: u64, count: u64) -> Entry {
	// sha1(<omega><offset>)
	let hasher = {
		let mut hasher = Sha1::new();
		hasher.update(omega);
		hasher
	};
	let mut best = Entry { offset: start, level: 0 };
	for i in start..(start + count) {
		let mut hasher = hasher.clone();
		let mut format_buf = [0u8; 20];
		let formatted = i.to_lexical(&mut format_buf);
		hasher.update(formatted);
		let level = get_hash_cash_level(&hasher.finalize());
		if level > best.level {
			best.offset = i;
			best.level = level;
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
	copy: DeviceSignal,
	omega: *const [u8],
	offset: u64,
	result_level: *mut [u8],
	result_offset: *mut [u64],
	queue: DeviceSingleQueue,
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

	fn queue(&self) -> &Self::Queue { &self.queue }

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

		// Global id
		let idx = vp.gl_id();
		// Number of workgroup
		let wg_idx = vp.wg_id().x;
		let start = self.offset + idx as u64 * PER_THREAD_COUNT;
		let mut best = find_best_level(omega, start, PER_THREAD_COUNT);

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
						// TODO Barrier here?
						// std::geobacter::amdgpu::sync::workgroup_barrier();
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
					*dest_level = best.level;
					*dest_offset = best.offset;
				}
			}
		}
	}
}

impl Args {
	pub fn omega_view(&self) -> Option<&[u8]> { unsafe { self.omega.as_ref() } }
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
	let options: Options = Options::parse();
	let omega = if let Some(key) = options.key {
		if options.uid.is_some() {
			eprintln!("Only one of key and uid can be used.");
			return;
		}
		let key = EccKeyPubP256::from_short(base64::decode(key).expect("Invalid base64 value"));
		key.to_ts().expect("Cannot convert public key")
	} else if let Some(uid) = options.uid {
		uid
	} else {
		eprintln!("One of key and uid must be provided.");
		return;
	};

	let ctxt = time("create context", || Context::new().expect("create context"));

	let accels = HsaAmdGpuAccel::all_devices(&ctxt).expect("HsaAmdGpuAccel::all_devices");
	if accels.len() < 1 {
		panic!("no accelerator devices???");
	}
	let accel = accels.first().unwrap();
	println!("Picking first device out of {}: {}", accels.len(), accel.agent().name().unwrap());

	println!("allocating {} MB of host memory", COUNT * size_of::<f32>() / 1024 / 1024);

	// Lap = locally accessible poos
	let lap_alloc = accel.fine_lap_node_alloc(0);
	let mut omega_buf: LapVec<u8> =
		time("alloc omega_buf", || LapVec::with_capacity_in(omega.len(), lap_alloc.clone()));
	time("Fill omega_buf", || omega_buf.extend_from_slice(omega.as_bytes()));
	let mut result_level_buf: LapVec<u8> =
		time("alloc result_level_buf", || LapVec::with_capacity_in(COUNT / WG_SIZE, lap_alloc.clone()));
	time("Fill result_level_buf", || result_level_buf.resize_with(COUNT / WG_SIZE, Default::default));
	let mut result_offset_buf: LapVec<u64> =
		time("alloc result_offset_buf", || LapVec::with_capacity_in(COUNT / WG_SIZE, lap_alloc.clone()));
	time("Fill result_offset_buf", || result_offset_buf.resize_with(COUNT / WG_SIZE, Default::default));

	let mut invoc: FuncModule<Args> = FuncModule::new(&accel);
	invoc.compile_async();
	unsafe {
		invoc.no_acquire_fence();
		invoc.device_release_fence();
	}

	let mut device_omega_ptr = time("alloc device slice", || unsafe {
		accel
			.alloc_device_local_slice::<u8>(omega_buf.len())
			.expect("HsaAmdGpuAccel::alloc_device_local")
	});
	let device_result_level_ptr = time("alloc device slice", || unsafe {
		accel
			.alloc_device_local_slice::<u8>(result_level_buf.len())
			.expect("HsaAmdGpuAccel::alloc_device_local")
	});
	let device_result_offset_ptr = time("alloc device slice", || unsafe {
		accel
			.alloc_device_local_slice::<u64>(result_offset_buf.len())
			.expect("HsaAmdGpuAccel::alloc_device_local")
	});

	println!("host ptr: 0x{:p}, agent ptr: 0x{:p}", omega_buf.as_ptr(), device_omega_ptr);

	let async_copy_signal =
		accel.new_device_signal(1).expect("HsaAmdGpuAccel::new_device_signal: async_copy_signal");
	let kernel_signal =
		GlobalSignal::new(1).expect("HsaAmdGpuAccel::new_host_signal: kernel_signal");
	let results_signal0 =
		accel.new_host_signal(1).expect("HsaAmdGpuAccel::new_host_signal: results_signal");
	let results_signal1 =
		accel.new_host_signal(1).expect("HsaAmdGpuAccel::new_host_signal: results_signal");

	time("grant gpu access: `omega_buf` and `values`", || {
		omega_buf.add_access(&*accel).expect("grant_agents_access");
		result_level_buf.add_access(&*accel).expect("grant_agents_access");
		result_offset_buf.add_access(&*accel).expect("grant_agents_access");
	});

	unsafe {
		accel
			.unchecked_async_copy_into(&omega_buf, &mut device_omega_ptr, &(), &async_copy_signal)
			.expect("HsaAmdGpuAccel::async_copy_into");
	}

	let group_size = invoc.group_size().expect("codegen failure");
	let private_size = invoc.private_size().unwrap();
	let queue =
		accel.create_single_queue2(None, group_size, private_size).expect("HsaAmdGpuAccel::create_single_queue");

	let args_pool =
		time("alloc kernargs pool", || ArgsPool::new::<Args>(&accel, 1).expect("ArgsPool::new"));

	let args = Args {
		copy: async_copy_signal,
		offset: options.offset,
		// TODO Omega has a known size, we can just put it into Args as a slice
		omega: device_omega_ptr.as_ptr(),
		result_level: device_result_level_ptr.as_ptr(),
		result_offset: device_result_offset_ptr.as_ptr(),
		completion: kernel_signal,
		queue,
	};

	let grid = Dim1D { x: 0u32..COUNT as _ };
	println!("Testing offsets {}..{}", options.offset, options.offset + COUNT as u64 * PER_THREAD_COUNT);

	time("compiling", || invoc.compile().expect("codegen failed"));

	let mut invoc = invoc.into_invoc(&args_pool);

	let wait = time("dispatching", || unsafe {
		invoc.unchecked_call_async(&grid, args).expect("Invoc::call_async")
	});

	// specifically wait (without enqueuing another async copy) here
	// so we can time just the dispatch.
	time("dispatch wait", move || {
		wait.wait_for_zero(false).expect("wait for zero failed");
	});

	// now copy the results back to the locked memory:
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

	let max = time("find maximum", || {
		result_level_buf.iter().copied().zip(result_offset_buf.iter().copied())
			.max_by_key(|(l, _)| *l).expect("Empty result buffer")
	});
	println!("Maximum level {} at offset {}", max.0, max.1);
}
