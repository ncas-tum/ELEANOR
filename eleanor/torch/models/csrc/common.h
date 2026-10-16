#pragma once

#include <torch/library.h>
#include <ATen/core/Tensor.h>

#include <cstdint>
#include <initializer_list>
#include <utility>

#ifdef __CUDACC__
#define ELEANOR_HD __host__ __device__ __forceinline__
#else
#define ELEANOR_HD inline
#endif

namespace eleanor
{

  // Matches jnp.sign / torch.sign: 0 at 0 (copysign would give +-1).
  ELEANOR_HD float sgn(float x) { return static_cast<float>((x > 0.f) - (x < 0.f)); }

  ELEANOR_HD float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

  ELEANOR_HD float reluf(float x) { return x > 0.f ? x : 0.f; }

  // Read-only view of a per-neuron parameter that broadcasts against the state.
  // The parameter either matches the state's trailing dims (e.g. (1, N) or (N,) against
  // (B, N)) or is a scalar, so for a contiguous state the flat index is idx % numel.
  struct BroadcastParam
  {
    const float *ptr;
    int64_t numel;

    ELEANOR_HD float operator[](int64_t idx) const { return ptr[numel == 1 ? 0 : idx % numel]; }
  };

  inline void check_state(const at::Tensor &ref, const char *ref_name,
                          std::initializer_list<std::pair<const at::Tensor *, const char *>> others,
                          c10::DeviceType device)
  {
    TORCH_CHECK(ref.scalar_type() == at::kFloat, ref_name, " must be float32, got ", ref.scalar_type());
    TORCH_CHECK(ref.device().type() == device, ref_name, " must be on ", device, ", got ", ref.device());
    for (const auto &[t, name] : others)
    {
      TORCH_CHECK(t->sizes() == ref.sizes(), name, " must have shape ", ref.sizes(), " (same as ", ref_name, "), got ", t->sizes());
      TORCH_CHECK(t->scalar_type() == at::kFloat, name, " must be float32, got ", t->scalar_type());
      TORCH_CHECK(t->device() == ref.device(), name, " must be on ", ref.device(), ", got ", t->device());
    }
  }

  // Validates `param` against the state `ref` and returns a contiguous copy (kept alive by
  // the caller) whose data can be wrapped in a BroadcastParam.
  inline at::Tensor broadcast_param(const at::Tensor &param, const char *name, const at::Tensor &ref)
  {
    TORCH_CHECK(param.scalar_type() == at::kFloat, name, " must be float32, got ", param.scalar_type());
    TORCH_CHECK(param.device() == ref.device(), name, " must be on ", ref.device(), ", got ", param.device());

    // Leading size-1 dims broadcast freely; what remains must equal ref's trailing dims.
    auto psizes = param.sizes();
    size_t lead = 0;
    while (lead < psizes.size() && psizes[lead] == 1)
      lead++;
    auto core = psizes.slice(lead);
    auto rsizes = ref.sizes();
    bool ok = param.numel() == 1 ||
              (core.size() <= rsizes.size() && core == rsizes.slice(rsizes.size() - core.size()));
    TORCH_CHECK(ok, name, " with shape ", psizes, " must be a scalar or match the trailing dims of ", rsizes);

    return param.contiguous();
  }

  inline BroadcastParam view(const at::Tensor &contig) { return {contig.data_ptr<float>(), contig.numel()}; }

} // namespace eleanor
