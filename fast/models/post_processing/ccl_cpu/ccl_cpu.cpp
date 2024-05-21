#include <torch/script.h>
#include <torch/types.h>
#include <ATen/Parallel.h>
// 2d
#define BLOCK_ROWS 16
#define BLOCK_COLS 16

template <typename T>
inline unsigned char hasBit(T bitmap, unsigned char pos) {
    return (bitmap >> pos) & 1;
}

int32_t find(const int32_t *s_buf, int32_t n) {
    while (s_buf[n] != n)
        n = s_buf[n];
    return n;
}

int32_t find_n_compress(int32_t *s_buf, int32_t n) {
    const int32_t id = n;
    while (s_buf[n] != n) {
        n = s_buf[n];
        s_buf[id] = n;
    }
    return n;
}

void union_(int32_t *s_buf, int32_t a, int32_t b) {
    bool done;
    do {
        a = find(s_buf, a);
        b = find(s_buf, b);

        if (a < b) {
            int32_t old = std::min(s_buf[b], a);
            done = (old == b);
            s_buf[b] = old;
            b = old;
        } else if (b < a) {
            int32_t old = std::min(s_buf[a], b);
            done = (old == a);
            s_buf[a] = old;
            a = old;
        } else
            done = true;

    } while (!done);
}

namespace cc2d {
    void init_labeling(int32_t *label, const uint32_t W, const uint32_t H) {
        at::parallel_for(0, H/2, 0, [&](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                for (uint32_t j = 0; j < W/2; ++j) {
                    const uint32_t row = i * 2;
                    const uint32_t col = j * 2;
                    const uint32_t idx = row * W + col;

                    if (row < H && col < W)
                        label[idx] = idx;
                }
            }
        });
    }

    void merge(uint8_t *img, int32_t *label, const uint32_t W, const uint32_t H) {
        at::parallel_for(0, H/2, 0, [&](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                for (uint32_t j = 0; j < W/2; ++j) {
                    const uint32_t row = i * 2;
                    const uint32_t col = j * 2;
                    const uint32_t idx = row * W + col;

                    if (row >= H || col >= W)
                        return;

                    uint32_t P = 0;

                    if (img[idx])                      P |= 0x777;
                    if (row + 1 < H && img[idx + W])   P |= 0x777 << 4;
                    if (col + 1 < W && img[idx + 1])   P |= 0x777 << 1;

                    if (col == 0)               P &= 0xEEEE;
                    if (col + 1 >= W)           P &= 0x3333;
                    else if (col + 2 >= W)      P &= 0x7777;

                    if (row == 0)               P &= 0xFFF0;
                    if (row + 1 >= H)           P &= 0xFF;

                    if (P > 0) {
                        if (hasBit(P, 0) && img[idx - W - 1]) {
                            union_(label, idx, idx - 2 * W - 2); // top left block
                        }

                        if ((hasBit(P, 1) && img[idx - W]) || (hasBit(P, 2) && img[idx - W + 1]))
                            union_(label, idx, idx - 2 * W); // top bottom block

                        if (hasBit(P, 3) && img[idx + 2 - W])
                            union_(label, idx, idx - 2 * W + 2); // top right block

                        if ((hasBit(P, 4) && img[idx - 1]) || (hasBit(P, 8) && img[idx + W - 1]))
                            union_(label, idx, idx - 2); // just left block
                    }
                }
            }
        });
    }

    void compression(int32_t *label, const int32_t W, const int32_t H) {
        at::parallel_for(0, H/2, 0, [&](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                for (uint32_t j = 0; j < W/2; ++j) {
                    const uint32_t row = i * 2;
                    const uint32_t col = j * 2;
                    const uint32_t idx = row * W + col;

                    if (row < H && col < W)
                        find_n_compress(label, idx);
                }
            }
        });
    }

    void final_labeling(const uint8_t *img, int32_t *label, const int32_t W, const int32_t H) {
        at::parallel_for(0, H/2, 0, [&](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                for (uint32_t j = 0; j < W/2; ++j) {
                    const uint32_t row = i * 2;
                    const uint32_t col = j * 2;
                    const uint32_t idx = row * W + col;

                    if (row >= H || col >= W)
                        return;

                    int32_t y = label[idx] + 1;

                    if (img[idx])
                        label[idx] = y;
                    else
                        label[idx] = 0;

                    if (col + 1 < W) {
                        if (img[idx + 1])
                            label[idx + 1] = y;
                        else
                            label[idx + 1] = 0;

                        if (row + 1 < H) {
                            if (img[idx + W + 1])
                                label[idx + W + 1] = y;
                            else
                                label[idx + W + 1] = 0;
                        }
                    }

                    if (row + 1 < H) {
                        if (img[idx + W])
                            label[idx + W] = y;
                        else
                            label[idx + W] = 0;
                    }
                }
            }
        });
    }

} // namespace cc2d

torch::Tensor connected_componnets_labeling_2d(const torch::Tensor &input) {
    AT_ASSERTM(!input.is_cuda(), "input must be a CPU tensor");
    AT_ASSERTM(input.ndimension() == 2, "input must be a [H, W] shape");
    AT_ASSERTM(input.scalar_type() == torch::kUInt8, "input must be a uint8 type");

    const uint32_t H = input.size(-2);
    const uint32_t W = input.size(-1);

    AT_ASSERTM((H % 2) == 0, "shape must be an even number");
    AT_ASSERTM((W % 2) == 0, "shape must be an even number");

    // label must be uint32_t
    auto label_options = torch::TensorOptions().dtype(torch::kInt32).device(input.device());
    torch::Tensor label = torch::zeros({H, W}, label_options);

    cc2d::init_labeling(
        label.data_ptr<int32_t>(), W, H
    );
    cc2d::merge(
        input.data_ptr<uint8_t>(),
        label.data_ptr<int32_t>(),
        W, H
    );
    cc2d::compression(
        label.data_ptr<int32_t>(), W, H
    );
    cc2d::final_labeling(
        input.data_ptr<uint8_t>(),
        label.data_ptr<int32_t>(),
        W, H
    );

    return label;
}

torch::Tensor connected_componnets_labeling_2d_batch(const torch::Tensor &input) {
    AT_ASSERTM(!input.is_cuda(), "input must be a CPU tensor");
    AT_ASSERTM(input.ndimension() == 3, "input must be a [C, H, W] shape");
    AT_ASSERTM(input.scalar_type() == torch::kUInt8, "input must be a uint8 type");

    const uint32_t H = input.size(-2);
    const uint32_t W = input.size(-1);
    const uint32_t B = input.size(-3);

    AT_ASSERTM((H % 2) == 0, "shape must be an even number");
    AT_ASSERTM((W % 2) == 0, "shape must be an even number");

    // label must be uint32_t
    auto label_options = torch::TensorOptions().dtype(torch::kInt32).device(input.device());
    torch::Tensor label = torch::zeros({B, H, W}, label_options);

    for (int i = 0; i < B; ++i) {
        cc2d::init_labeling(
            label[i].data_ptr<int32_t>(), W, H
        );
        cc2d::merge(
            input[i].data_ptr<uint8_t>(),
            label[i].data_ptr<int32_t>(),
            W, H
        );
        cc2d::compression(
            label[i].data_ptr<int32_t>(), W, H
        );
        cc2d::final_labeling(
            input[i].data_ptr<uint8_t>(),
            label[i].data_ptr<int32_t>(),
            W, H
        );
    }
    return label;
}
