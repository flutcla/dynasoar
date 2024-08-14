#ifndef ALLOCATOR_ALLOCATOR_HANDLE_H
#define ALLOCATOR_ALLOCATOR_HANDLE_H

#include "allocator/soa_allocator.h"
#include "allocator/configuration.h"
#include "allocator/tuple_helper.h"
#include "allocator/util.h"
#include "bitmap/bitmap.h"

/**
 * Initializes an allocator at a given memory location.
 * @param allocator Location of allocator
 * @param data_buffer Location of the heap
 * @tparam AllocatorT Allocator type
 */
template<typename AllocatorT>
__global__ void init_allocator_kernel(AllocatorT* allocator,
                                      char* data_buffer) {
  new(allocator) AllocatorT(data_buffer);
}

/**
 * Prints debug information about the state of the allocator.
 * @param allocator Pointer to allocator
 * @tparam AllocatorT Allocator type
 */
template<typename AllocatorT>
__global__ void kernel_print_state_stats(AllocatorT* allocator) {
  assert(gridDim.x == 1 && blockDim.x == 1);
  allocator->DBG_print_state_stats();
}

/**
 * A wrapper class for accessing the allocator from host side. Creating a new
 * object of this class allocates and initializes various data buffers on the
 * host and on the device.
 * @tparam AllocatorT Device allocator type
 */
template<typename AllocatorT>
class AllocatorHandle {
 public:
  /**
   * Allocators cannot be copied.
   */
  AllocatorHandle(const AllocatorHandle<AllocatorT>&) = delete;

  /**
   * Initializes the allocator.
   * 1. Allocates a device allocator object (AllocatorT) in device memory.
   * 2. Allocates a heap data buffer in device memory.
   * 3. Initializes the device allocator.
   * If \p unified_memory is set to true, then all device allocations are
   * allocated in CUDA unified memory. This allows programmers to access
   * objects located on device in host code without manual memory transfers.
   * @param unified_memory Enable/disable unified memory
   */
  AllocatorHandle(bool unified_memory = false)
      : unified_memory_(unified_memory) {
#ifndef NDEBUG
    int device_id;
    gpuErrchk(cudaGetDevice(&device_id));

    // Source: https://devblogs.nvidia.com/how-query-device-properties-and-handle-errors-cuda-cc/
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    printf("Current Device Number: %d\n", device_id);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);

    // Query memory information.
    size_t free_mem, total_mem;
    gpuErrchk(cudaMemGetInfo(&free_mem, &total_mem));
    printf("  Total global memory: %f MB\n",
           (total_mem/1000000.0));
    printf("  Available (free) global memory: %f MB\n\n",
           (free_mem/1000000.0));

    AllocatorT::DBG_print_stats();
#endif  // NDEBUG

    if (unified_memory) {
      // Unified memory is accessible from both host and device.
      gpuErrchk(cudaMallocManaged(&allocator_, sizeof(AllocatorT)));
    } else {
      gpuErrchk(cudaMalloc(&allocator_, sizeof(AllocatorT)));
    }

    assert(allocator_ != nullptr);

    if (unified_memory) {
      gpuErrchk(cudaMallocManaged(&data_buffer_, AllocatorT::kDataBufferSize));
    } else {
      gpuErrchk(cudaMalloc(&data_buffer_, AllocatorT::kDataBufferSize));
    }

#ifndef NDEBUG
    void* maybe_out_of_memory = nullptr;  // To show OOM text...
    assert(data_buffer_ != maybe_out_of_memory);

    gpuErrchk(cudaMemGetInfo(&free_mem, &total_mem));
    printf("  Data buffer:  %p\n", data_buffer_);
    printf("  Available (free) global memory after init: %f MB\n\n",
           (free_mem/1000000.0));
#endif  // NDEBUG

    load_cub_buffer_addresses();

    init_allocator_kernel<<<256, 256>>>(allocator_, data_buffer_);
    gpuErrchk(cudaDeviceSynchronize());
  }

  /**
   * Deletes the allocator: Frees all associated CUDA memory.
   */
  ~AllocatorHandle() {
    cudaFree(allocator_);
    cudaFree(data_buffer_);
  }

  long unsigned int DBG_get_enumeration_time() const {
    return allocator_->DBG_get_enumeration_time();
  }

  void DBG_collect_stats() {
    member_func_kernel<AllocatorT, &AllocatorT::DBG_collect_stats>
        <<<1, 1>>>(allocator_);
    gpuErrchk(cudaDeviceSynchronize());
  }

  void DBG_print_collected_stats() {
    member_func_kernel<AllocatorT, &AllocatorT::DBG_print_collected_stats>
        <<<1, 1>>>(allocator_);
    gpuErrchk(cudaDeviceSynchronize());
  }

  /**
   * Returns a device handle to the allocator.
   */
  AllocatorT* device_pointer() const { return allocator_; }

  /**
   * Parallel do-all: Runs a member function T::func for all objects of type T
   * and subtypes in parallel in a CUDA kernel. Spawns a separate kernel for
   * each subtype to avoid branch divergence.
   * @tparam T Base class
   * @tparam func Member function to be run in parallel
   */
  template<class T, void(T::*func)()>
  void parallel_do() {
    allocator_->template parallel_do<true, T, func>();
  }

  /**
   * Parallel do-all: Same as parallel_do(), but the member function takes one
   * argument of type \p P1.
   * @tparam T Base class
   * @tparam P1 Type of first parameter
   * @tparam func Member function to be run in parallel
   * @param p1 Argument value
   */
  template<class T, typename P1, void(T::*func)(P1)>
  void parallel_do(P1 p1) {
    allocator_->template parallel_do<true, T, P1, func>(std::forward<P1>(p1));
  }

  /**
   * Like parallel_do, but limits the number of blocks
   * @tparam T Base class
   * @tparam func Member function to be run in parallel
   * @param bound Number of blocks
   */
  template<class T, void(T::*func)()>
  void parallel_do_bounded(int bound) {
    allocator_->template parallel_do_bounded<true, T, func>(bound);
  }

  /**
   * Same as parallel_do_bounded, but does omits the scan operation.
   * @tparam T Base class
   * @tparam func Member function to be run in parallel
   * @param bound Number of blocks
   */
  template<class T, void(T::*func)()>
  void fast_parallel_do_bounded(int bound) {
    allocator_->template parallel_do_bounded<false, T, func>(bound);
  }

  /**
   * Same as parallel_do, but the bound is the number of *objects*
   * @tparam T Base class
   * @tparam func Member function to be run in parallel
   * @param bound Number of objects
   */
  template<class T, void(T::*func)()>
  void parallel_do_bounded_by_count(int bound) {
    static const int kSize = AllocatorT::template BlockHelper<T>::kSize;
    allocator_->template parallel_do_bounded<true, T, func>((bound + kSize - 1) / kSize);
  }

  /**
   * Same as parallel_do_bounded_by_count, but does omits the scan operation.
   * @tparam T Base class
   * @tparam func Member function to be run in parallel
   * @param bound Number of objects
   */
  template<class T, void(T::*func)()>
  void fast_parallel_do_bounded_by_count(int bound) {
    static const int kSize = AllocatorT::template BlockHelper<T>::kSize;
    allocator_->template parallel_do_bounded<false, T, func>((bound + kSize - 1) / kSize);
  }

  /**
   * Like parallel_do, but enumerates only objects of type \p IterT.
   * @tparam IterT Class who's objects are enumerated
   * @tparam T Base class
   * @tparam func Member function to be run in parallel
   */
  template<class IterT, class T, void(T::*func)()>
  void parallel_do_single_type() {
    allocator_->template parallel_do_single_type<IterT, T, func, true>();
  }

  /**
   * Like parallel_do, but enumerates only objects of type \p IterT.
   * @tparam IterT Class who's objects are enumerated
   * @tparam T Base class
   * @tparam P1 Type of first parameter
   * @tparam func Member function to be run in parallel
   * @param p1 Argument value
   */
  template<class IterT, class T, typename P1, void(T::*func)(P1)>
  void parallel_do_single_type(P1 p1) {
    allocator_->template parallel_do_single_type<IterT, T, P1, func, true>(
        std::forward<P1>(p1));
  }

  /**
   * Fast parallel do-all: Same as parallel_do(), but does omits the scan
   * operation. Assumes that the set of objects of type \p T has not changed
   * since the last parallel do-all operation. (The state of the objects
   * themselves may have been modified since then.)
   * @tparam T Base class
   * @tparam func Member function to be run in parallel
   */
  template<class T, void(T::*func)()>
  void fast_parallel_do() {
    allocator_->template parallel_do<false, T, func>();
  }

  /**
   * Fast parallel do-all: Same as fast_parallel_do(), but the member function
   * takes one argument of type \p P1.
   * @tparam T Base class
   * @tparam P1 Type of first parameter
   * @tparam func Member function to be run in parallel
   * @param p1 Argument value
   */
  template<class T, typename P1, void(T::*func)(P1)>
  void fast_parallel_do(P1 p1) {
    allocator_->template parallel_do<false, T, P1, func>(p1);
  }

  /**
   * Device do: Runs a member function \p func of class \p T for all objects
   * of type \p T. In contrast to parallel_do(), device_do() runs on the host
   * and without parallelization. Requires unified_memory support.
   * @param func Member function to be run
   * @param args Argument values
   * @tparam T Base class
   * @tparam F Type of member function
   * @tparam Args Types of arguments
   */
  template<class T, typename F, typename... Args>
  void device_do(F func, Args... args) {
    assert(unified_memory_);
    allocator_->template device_do<T, F, Args...>(
        func, std::forward<Args>(args)...);
  }

  /**
   * Parallel new: Create \p num_objects new objects of type \p T in parallel.
   * @param num_objects Number of new objects
   * @param args Common constructor arguments passes to all new objects
   * @tparam T Base class
   * @tparam Args Constructor argument types
   */
  template<class T, typename... Args>
  void parallel_new(int num_objects, Args... args) {
    allocator_->template parallel_new<T>(num_objects, args...);
  }

#ifdef OPTION_DEFRAG
  // Defrag/compact all objects of type T. Also updates all affected pointers
  // in the data buffer.
  // Should be invoked from host side.
  template<typename T, int NumRecords>
  void parallel_defrag(int min_num_compactions = kMinDefragCandidates) {
    allocator_->parallel_defrag<T, NumRecords>(min_num_compactions);
  }

  template<typename T>
  void parallel_defrag(int min_num_compactions = kMinDefragCandidates) {
    allocator_->parallel_defrag<T>(min_num_compactions);
  }
#endif  // OPTION_DEFRAG

  /**
   * Count allocated block: counts the number of allocated blocks of type T.
   * @tparam T Type of object
   * @param scan Whether to scan the bitmap
   * @return Number of allocated blocks
   */
  template<class T>
  int count_allocated_block(bool scan = true) {
    static const int kTypeIndex = AllocatorT::template BlockHelper<T>::kIndex;
    if (scan)
    {
      allocator_->allocated_[kTypeIndex].scan();
    }
    auto* num_soa_blocks_ptr = allocator_->allocated_[kTypeIndex].scan_num_bits_ptr();
    auto num_soa_blocks = copy_from_device(num_soa_blocks_ptr);
    gpuErrchk(cudaDeviceSynchronize());
    return num_soa_blocks;
  }

  /**
   * Count free block
   * @param scan Whether to scan the bitmap
   * @return Number of free blocks
   */
  int count_free_block(bool scan = true) {
    if (scan)
    {
      allocator_->global_free_.scan();
    }
    auto* num_soa_blocks_ptr = allocator_->global_free_.scan_num_bits_ptr();
    auto num_soa_blocks = copy_from_device(num_soa_blocks_ptr);
    return num_soa_blocks;
  }

  /**
   * Count allocated object roughly: counts the number of allocated objects of
   * type T. This is a rough estimate based on the number of allocated blocks.
   * @tparam T Type of object
   * @param scan Whether to scan the bitmap
   * @return Number of allocated objects
   */
  template<class T>
  int count_allocated_object_roughly(bool scan = true) {
    auto num_soa_blocks = count_allocated_block<T>(scan);
    static const int kSize = AllocatorT::template BlockHelper<T>::kSize;
    return num_soa_blocks * kSize;
  }

  void DBG_print_state_stats() {
    kernel_print_state_stats<<<1, 1>>>(allocator_);
    gpuErrchk(cudaDeviceSynchronize());
  }

#ifdef OPTION_DEFRAG
  void DBG_print_defrag_time() {
    allocator_->DBG_print_defrag_time();
  }
#endif  // OPTION_DEFRAG

 private:
  /**
   * Pointer to allocator handle in device memory.
   */
  AllocatorT* allocator_ = nullptr;

  /**
   * Data buffer/heap in device memory.
   */
  char* data_buffer_ = nullptr;

  /**
   * Unified memory activated?
   */
  bool unified_memory_ = false;
};

#endif  // ALLOCATOR_ALLOCATOR_HANDLE_H
