#include "blurs.h"
#include "texture.h"

#include "../shaders/blur.glsl.h"

#include <assert.h>
#include <vector>

// ======== Context

struct BlurContext
{
    sg_pipeline pip_separable;
    sg_pipeline pip_dk_down, pip_dk_up, pip_dk_mix, pip_dk_copy;
    sg_pipeline pip_split_kawase_step;
    sg_sampler smp_linear, smp_nearest;

    ~BlurContext()
    {
        sg_destroy_pipeline(pip_separable);
        sg_destroy_pipeline(pip_dk_down);
        sg_destroy_pipeline(pip_dk_down);
        sg_destroy_pipeline(pip_dk_mix);
        sg_destroy_pipeline(pip_dk_copy);
        sg_destroy_pipeline(pip_split_kawase_step);
        sg_destroy_sampler(smp_linear);
        sg_destroy_sampler(smp_nearest);
    }
};

BlurContext *blur_ctx_initialize()
{
    BlurContext *ctx = new BlurContext{};
    ctx->pip_separable = sg_make_pipeline(sg_pipeline_desc{
        .shader = sg_make_shader(blur_separable_shader_desc(sg_query_backend())),
        .colors[0].pixel_format = SG_PIXELFORMAT_RGBA32F,
        .depth.pixel_format = SG_PIXELFORMAT_NONE,
        .primitive_type = SG_PRIMITIVETYPE_TRIANGLE_STRIP,
        .label = "pipe-blur-separable",
    });
    ctx->pip_dk_down = sg_make_pipeline(sg_pipeline_desc{
        .shader = sg_make_shader(blur_dk_down_shader_desc(sg_query_backend())),
        .colors[0].pixel_format = SG_PIXELFORMAT_RGBA32F,
        .depth.pixel_format = SG_PIXELFORMAT_NONE,
        .primitive_type = SG_PRIMITIVETYPE_TRIANGLE_STRIP,
        .label = "pipe-dk-down",
    });
    ctx->pip_dk_up = sg_make_pipeline(sg_pipeline_desc{
        .shader = sg_make_shader(blur_dk_up_shader_desc(sg_query_backend())),
        .colors[0].pixel_format = SG_PIXELFORMAT_RGBA32F,
        .depth.pixel_format = SG_PIXELFORMAT_NONE,
        .primitive_type = SG_PRIMITIVETYPE_TRIANGLE_STRIP,
        .label = "pipe-dk-up",
    });
    ctx->pip_dk_mix = sg_make_pipeline(sg_pipeline_desc{
        .shader = sg_make_shader(blur_dk_mix_shader_desc(sg_query_backend())),
        .colors[0] = {
            .pixel_format = SG_PIXELFORMAT_RGBA32F,
            .blend = {
                .enabled = true,
                .src_factor_rgb = SG_BLENDFACTOR_SRC_ALPHA,
                .dst_factor_rgb = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
                .src_factor_alpha = SG_BLENDFACTOR_SRC_ALPHA,
                .dst_factor_alpha = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
            },
        },
        .depth.pixel_format = SG_PIXELFORMAT_NONE,
        .primitive_type = SG_PRIMITIVETYPE_TRIANGLE_STRIP,
        .label = "pipe-dk-mix",
    });
    ctx->pip_dk_copy = sg_make_pipeline(sg_pipeline_desc{
        .shader = sg_make_shader(blur_dk_copy_shader_desc(sg_query_backend())),
        .colors[0].pixel_format = SG_PIXELFORMAT_RGBA32F,
        .depth.pixel_format = SG_PIXELFORMAT_NONE,
        .primitive_type = SG_PRIMITIVETYPE_TRIANGLE_STRIP,
        .label = "pipe-dk-copy",
    });
    ctx->pip_split_kawase_step = sg_make_pipeline(sg_pipeline_desc{
        .shader = sg_make_shader(blur_split_kawase_step_shader_desc(sg_query_backend())),
        .colors[0].pixel_format = SG_PIXELFORMAT_RGBA32F,
        .depth.pixel_format = SG_PIXELFORMAT_NONE,
        .primitive_type = SG_PRIMITIVETYPE_TRIANGLE_STRIP,
        .label = "pipe-split-kawase-step",
    });

    ctx->smp_linear = sg_make_sampler(sg_sampler_desc{
        .min_filter = SG_FILTER_LINEAR,
        .mag_filter = SG_FILTER_LINEAR,
        .mipmap_filter = SG_FILTER_NEAREST,
        .wrap_u = SG_WRAP_CLAMP_TO_EDGE,
        .wrap_v = SG_WRAP_CLAMP_TO_EDGE,
        .label = "blur-linear-sampler",
    });
    ctx->smp_nearest = sg_make_sampler(sg_sampler_desc{
        .min_filter = SG_FILTER_NEAREST,
        .mag_filter = SG_FILTER_NEAREST,
        .mipmap_filter = SG_FILTER_NEAREST,
        .wrap_u = SG_WRAP_CLAMP_TO_EDGE,
        .wrap_v = SG_WRAP_CLAMP_TO_EDGE,
        .label = "blur-nearest-sampler",
    });
    return ctx;
}

void blur_ctx_cleanup(BlurContext *ctx) { delete ctx; }

// ======== Separable kernel blur

static float separable_kernel_value(BlurMode mode, float x)
{
    x = fabsf(x);
    switch (mode)
    {
    case BLUR_BOX:
        return x > 1.0f ? 0.0f : 1.0f;
    case BLUR_TENT:
        return x > 1.0f ? 0.0f : 1.0f - x;
    case BLUR_GAUSSIAN:
    {
        constexpr float scale = 1.6f;
        constexpr float two_scale2 = 2.0f * scale * scale;
        x *= 3.0f * scale;
        return 1.0f / sqrtf(float(M_PI) * two_scale2) * expf(-x * x / two_scale2);
    }
    default:
        assert(false);
        return 0.0f;
    }
}

static Texture *calc_separable_weights(BlurMode mode, float radius)
{
    // Kernel size is radius+1, but since it is symmetric we only store
    // one half.
    const int size = ceilf(radius) + 1;
    std::vector<float> result(size);

    float sum = 0.0f;

    // Center weight
    const float center_weight = separable_kernel_value(mode, 0.0f);
    result[0] = center_weight;
    sum += center_weight;

    // Other weights in the positive direction. Add double to the sum, to account for
    // the negative direction as well.
    const float scale = radius > 0.0f ? 1.0f / radius : 0.0f;
    for (int i = 1; i < size; ++i)
    {
        const float weight = separable_kernel_value(mode, i * scale);
        result[i] = weight;
        sum += weight * 2.0f;
    }

    // Normalize the weights
    for (int i = 0; i < size; ++i)
    {
        result[i] /= sum;
    }

    return new Texture(size, 1, 1, result.data(), "separable-kernel");
}

static void separable_pass(
    BlurContext *ctx, Texture *input, Texture *output, BlurMode mode, bool horizontal, float radius)
{
    Texture *weights = calc_separable_weights(mode, radius);

    const fs_separable_params_t par = {
        .uv_step[0] = horizontal ? 1.0f / input->width : 0.0f,
        .uv_step[1] = horizontal ? 0.0f : 1.0f / input->height,
        .kernel_width = weights->width,
    };
    sg_pass pass = {
        .action = { .colors[0] = { .load_action = SG_LOADACTION_DONTCARE } },
        .attachments = {
            .colors[0] = output->view_attachment,
        },
    };
    sg_begin_pass(&pass);
    sg_apply_pipeline(ctx->pip_separable);

    {
        sg_bindings bind = {
            .views[VIEW_tex] = input->view_sample,
            .views[VIEW_tex_kernel] = weights->view_sample,
            .samplers[SMP_smp] = ctx->smp_nearest,
        };
        sg_apply_bindings(&bind);
    }
    sg_apply_uniforms(UB_fs_separable_params, SG_RANGE(par));
    sg_draw(0, 4, 1);
    sg_end_pass();

    delete weights;
}

static void blur_separable(BlurContext *ctx, Texture *input, Texture *output, const BlurParams &params)
{
    Texture *tmp = new Texture(input->width, input->height, "blur-gauss-tmp");
    separable_pass(ctx, input, tmp, params.mode, true, params.radius_x);
    separable_pass(ctx, tmp, output, params.mode, false, params.radius_y);
    delete tmp;
}

// ======== Dual Kawase blur

static Texture *kawase_downsample(
    BlurContext *ctx, const Texture &input, int full_size_x, int full_size_y, int divisor, float ratio)
{
    int size_x = std::max(full_size_x / divisor, 1);
    int size_y = std::max(full_size_y / divisor, 1);
    Texture *output = new Texture(size_x, size_y, "kawase_down");
    // Note: "step" is based on *output* size, which is 2x smaller than input
    float step_x = ratio / size_x;
    float step_y = ratio / size_y;

    const fs_dk_down_params_t par = {
        .uv_step[0] = step_x,
        .uv_step[1] = step_y,
    };
    sg_begin_pass(sg_pass{ .action = { .colors[0] = { .load_action = SG_LOADACTION_DONTCARE } },
        .attachments = {
            .colors[0] = output->view_attachment,
        } });
    sg_apply_pipeline(ctx->pip_dk_down);
    sg_apply_bindings(sg_bindings{
        .views[VIEW_tex] = input.view_sample,
        .samplers[SMP_smp] = ctx->smp_linear,
    });
    sg_apply_uniforms(UB_fs_dk_down_params, SG_RANGE(par));
    sg_draw(0, 4, 1);
    sg_end_pass();
    return output;
}

static Texture *kawase_upsample(
    BlurContext *ctx, const Texture &input, int full_size_x, int full_size_y, int divisor, float ratio, Texture *output)
{
    int size_x = std::max(full_size_x / divisor, 1);
    int size_y = std::max(full_size_y / divisor, 1);
    if (output == nullptr)
        output = new Texture(size_x, size_y, "kawase_up");
    // Note: "step" is based on *input* size, which is 2x smaller than output
    float step_x = ratio / input.width;
    float step_y = ratio / input.height;

    const fs_dk_up_params_t par = {
        .uv_step[0] = step_x,
        .uv_step[1] = step_y,
    };
    sg_begin_pass(sg_pass{ .action = { .colors[0] = { .load_action = SG_LOADACTION_DONTCARE } },
        .attachments = {
            .colors[0] = output->view_attachment,
        } });
    sg_apply_pipeline(ctx->pip_dk_up);
    sg_apply_bindings(sg_bindings{
        .views[VIEW_tex] = input.view_sample,
        .samplers[SMP_smp] = ctx->smp_linear,
    });
    sg_apply_uniforms(UB_fs_dk_up_params, SG_RANGE(par));
    sg_draw(0, 4, 1);
    sg_end_pass();
    return output;
}

static void kawase_mix(BlurContext *ctx, const Texture &input, Texture &output, float ratio)
{
    const fs_dk_mix_params_t par = {
        .ratio = ratio,
    };
    sg_begin_pass(sg_pass{ .action = { .colors[0] = { .load_action = SG_LOADACTION_LOAD } },
        .attachments = {
            .colors[0] = output.view_attachment,
        } });
    sg_apply_pipeline(ctx->pip_dk_mix);
    sg_apply_bindings(sg_bindings{
        .views[VIEW_tex] = input.view_sample,
        .samplers[SMP_smp] = ctx->smp_nearest,
    });
    sg_apply_uniforms(UB_fs_dk_mix_params, SG_RANGE(par));
    sg_draw(0, 4, 1);
    sg_end_pass();
    return output;
}

static void kawase_copy(BlurContext *ctx, const Texture &input, Texture &output)
{
    sg_begin_pass(sg_pass{ .action = { .colors[0] = { .load_action = SG_LOADACTION_DONTCARE } },
        .attachments = {
            .colors[0] = output.view_attachment,
        } });
    sg_apply_pipeline(ctx->pip_dk_copy);
    sg_apply_bindings(sg_bindings{
        .views[VIEW_tex] = input.view_sample,
        .samplers[SMP_smp] = ctx->smp_nearest,
    });
    sg_draw(0, 4, 1);
    sg_end_pass();
    return output;
}

static void blur_dual_kawase(BlurContext *ctx, Texture *input, Texture *output, const BlurParams &params)
{
    // Dual Kawase is isotropic, so blur based on max(x,y)
    float radius = std::max(params.radius_x, params.radius_y);

    // Amount of Kawase "steps" to do; this more or less matches the blur
    // amount of Gaussian.
    float num_steps = radius / 3.0f;
    if (num_steps <= 0.0f)
    {
        kawase_copy(ctx, *input, *output);
        return;
    }

    const int full_size_x = input->width;
    const int full_size_y = input->height;

    Texture *curr = input;

    // Downsample
    int last_pass = 0;
    for (int i = 2; i <= num_steps; i *= 2)
    {
        Texture *tmp = kawase_downsample(ctx, *curr, full_size_x, full_size_y, i, 1.0f);
        if (curr != input)
            delete curr;
        curr = tmp;
        last_pass = i;
    }

    float residual = num_steps - last_pass;
    if (residual > 0.0f)
    {
        int next_pass = std::max(last_pass, 1) * 2;
        float ratio = residual / (next_pass - last_pass);
        last_pass = std::max(last_pass, 1);

        // Downsample and upsample one more step
        Texture *extra_down = kawase_downsample(ctx, *curr, full_size_x, full_size_y, next_pass, 0.5f + 0.5f * ratio);
        Texture *extra_up =
            kawase_upsample(ctx, *extra_down, full_size_x, full_size_y, last_pass, 0.5f + 0.5f * ratio, nullptr);
        delete extra_down;

        // Mix current with that extra step based on ratio
        kawase_mix(ctx, *curr, *extra_up, 1.0f - ratio);
        if (curr != input)
            delete curr;
        curr = extra_up;

        // If there will be no further upsamples, copy to output.
        if (last_pass < 2)
        {
            kawase_copy(ctx, *curr, *output);
            delete curr;
        }
    }

    // Upsample.
    for (int i = last_pass / 2; i >= 1; i /= 2)
    {
        bool is_last = i == 1;
        Texture *tmp = kawase_upsample(ctx, *curr, full_size_x, full_size_y, i, 1.0f, is_last ? output : nullptr);
        if (curr != input)
            delete curr;
        curr = tmp;
    }
}

// ======== Split Kawase blur

static Texture *split_kawase_step(BlurContext *ctx, const FilterConfig &cfg, const Texture &input, int full_size_x,
    int full_size_y, int divisor, bool horizontal)
{
    int size_x = horizontal ? std::max(full_size_x / divisor, 1) : full_size_x;
    int size_y = horizontal ? full_size_y : std::max(full_size_y / divisor, 1);
    Texture *output = new Texture(size_x, size_y, "split_kawase_step");
    // X or Y component is set to zero based on whether we are doing horizontal or vertical pass.
    float step_x = horizontal ? 1.0f / input.width : 0.0f;
    float step_y = horizontal ? 0.0f : 1.0f / input.height;

    const fs_split_kawase_step_params_t par = {
        .uv_step[0] = step_x,
        .uv_step[1] = step_y,
        .w00 = cfg.w00,
        .w05 = cfg.w05,
        .w10 = cfg.w10,
        .w15 = cfg.w15,
        .w20 = cfg.w20,
    };
    sg_begin_pass(sg_pass{ .action = { .colors[0] = { .load_action = SG_LOADACTION_DONTCARE } },
        .attachments = {
            .colors[0] = output->view_attachment,
        } });
    sg_apply_pipeline(ctx->pip_split_kawase_step);
    sg_apply_bindings(sg_bindings{
        .views[VIEW_tex] = input.view_sample,
        .samplers[SMP_smp] = ctx->smp_linear,
    });
    sg_apply_uniforms(UB_fs_split_kawase_step_params, SG_RANGE(par));
    sg_draw(0, 4, 1);
    sg_end_pass();
    return output;
}

static void split_kawase_axis(BlurContext *ctx, const FilterConfig &cfg_down, const FilterConfig &cfg_up,
    Texture *input, Texture *output, float radius, bool horizontal)
{
    // Amount of Kawase "steps" to do; this more or less matches the blur amount of Gaussian.
    float num_steps = radius / 3.0f;

    const int full_size_x = input->width;
    const int full_size_y = input->height;

    Texture *curr = input;

    // Downsample
    int last_pass = 0;
    for (int i = 2; i <= num_steps; i *= 2)
    {
        Texture *tmp = split_kawase_step(ctx, cfg_down, *curr, full_size_x, full_size_y, i, horizontal);
        if (curr != input)
            delete curr;
        curr = tmp;
        last_pass = i;
    }

    float residual = num_steps - last_pass;
    if (residual > 0.0f)
    {
        int next_pass = std::max(last_pass, 1) * 2;
        float ratio = residual / (next_pass - last_pass);
        last_pass = std::max(last_pass, 1);

        // Downsample and upsample one more step
        Texture *extra_down = split_kawase_step(ctx, cfg_down, *curr, full_size_x, full_size_y, next_pass, horizontal);
        Texture *extra_up =
            split_kawase_step(ctx, cfg_up, *extra_down, full_size_x, full_size_y, last_pass, horizontal);
        delete extra_down;

        // Mix current with that extra step based on ratio
        kawase_mix(ctx, *curr, *extra_up, 1.0f - ratio);
        if (curr != input)
            delete curr;
        curr = extra_up;
    }

    // Upsample.
    for (int i = last_pass / 2; i >= 1; i /= 2)
    {
        bool is_last = i == 1;
        Texture *tmp = split_kawase_step(ctx, cfg_up, *curr, full_size_x, full_size_y, i, horizontal);
        if (curr != input)
            delete curr;
        curr = tmp;
    }

    kawase_copy(ctx, *curr, *output);
    if (curr != input)
        delete curr;
}

static void blur_split_kawase(BlurContext *ctx, Texture *input, Texture *output, const BlurParams &params)
{
    Texture *tmp = new Texture(input->width, input->height, "split-kawase-tmp");
    split_kawase_axis(ctx, params.split_cfg_down, params.split_cfg_up, input, tmp, params.radius_x, true);
    split_kawase_axis(ctx, params.split_cfg_down, params.split_cfg_up, tmp, output, params.radius_y, false);
    delete tmp;
}

// ======== Main entry

void blur_calc(BlurContext *ctx, Texture *input, Texture *output, const BlurParams &params)
{
    if (params.mode == BLUR_DUAL_KAWASE)
        blur_dual_kawase(ctx, input, output, params);
    else if (params.mode == BLUR_SPLIT_KAWASE)
        blur_split_kawase(ctx, input, output, params);
    else
        blur_separable(ctx, input, output, params);
}
