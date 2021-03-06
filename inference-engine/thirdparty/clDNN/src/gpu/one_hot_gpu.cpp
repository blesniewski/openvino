// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "one_hot_inst.h"

#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "one_hot/one_hot_kernel_selector.h"
#include "one_hot/one_hot_kernel_base.h"
#include "cldnn/runtime/error_handler.hpp"
#include <vector>

namespace cldnn {
namespace gpu {

struct one_hot_gpu : typed_primitive_gpu_impl<one_hot> {
    using parent = typed_primitive_gpu_impl<one_hot>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<one_hot_gpu>(*this);
    }

    static primitive_impl* create(const one_hot_node& arg) {
        auto oh_params = get_default_params<kernel_selector::one_hot_params>(arg, 1);
        auto oh_optional_params =
            get_default_optional_params<kernel_selector::one_hot_optional_params>(arg.get_program());

        oh_params.one_hot_axis = arg.get_primitive()->one_hot_axis;
        oh_params.on_value = arg.get_primitive()->on_value;
        oh_params.off_value = arg.get_primitive()->off_value;

        auto output_sizes = arg.get_output_layout().format == format::bfzyx ?
                            arg.get_output_layout().size.sizes(format::bfzyx) :
                            arg.get_output_layout().size.sizes(format::bfyx);

        oh_params.one_hot_limit = output_sizes[oh_params.one_hot_axis];

        auto& kernel_selector = kernel_selector::one_hot_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(oh_params, oh_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with these arguments");

        return new one_hot_gpu(arg, best_kernels[0]);
    }
};

namespace detail {

attach_one_hot_gpu::attach_one_hot_gpu() {
    auto val_fw = one_hot_gpu::create;

    implementation_map<one_hot>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx), val_fw);
    implementation_map<one_hot>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfyx), val_fw);
    implementation_map<one_hot>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfyx), val_fw);
    implementation_map<one_hot>::add(std::make_tuple(engine_types::ocl, data_types::i64, format::bfyx), val_fw);
    implementation_map<one_hot>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
    implementation_map<one_hot>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
    implementation_map<one_hot>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfzyx), val_fw);
    implementation_map<one_hot>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfzyx), val_fw);
    implementation_map<one_hot>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfzyx), val_fw);
    implementation_map<one_hot>::add(std::make_tuple(engine_types::ocl, data_types::i64, format::bfzyx), val_fw);
    implementation_map<one_hot>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx), val_fw);
    implementation_map<one_hot>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx), val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
