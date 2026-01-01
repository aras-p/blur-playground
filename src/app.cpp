#define SOKOL_IMPL
#if defined(__APPLE__)
#define SOKOL_METAL
#elif defined(_WIN32)
#define SOKOL_D3D11
#else
#define SOKOL_GLCORE
#endif

#include "../lib/imgui/imgui.h"
#include "../lib/sokol/sokol_app.h"
#include "../lib/sokol/sokol_gfx.h"
#include "../lib/sokol/sokol_glue.h"
#include "../lib/sokol/sokol_log.h"
#define SOKOL_IMGUI_IMPL
#include "../lib/sokol/util/sokol_imgui.h"
#include "../lib/tinyexr/tinyexr.h"

#include "../shaders/blur.glsl.h"
#include "../shaders/display_tex.glsl.h"

#include <string>
#include <vector>

enum BlurMode
{
    BLUR_GAUSSIAN = 0,
    BLUR_DUAL_KAWASE = 1,
};

static sg_pixel_format pixel_format_from_channels(int channels)
{
    switch (channels)
    {
    case 1:
        return SG_PIXELFORMAT_R32F;
    case 2:
        return SG_PIXELFORMAT_RG32F;
    case 4:
        return SG_PIXELFORMAT_RGBA32F;
    default:
        return SG_PIXELFORMAT_NONE;
    }
}

struct Texture
{
    // texture for sampling
    Texture(int width, int height, int channels, const float *data, const char *dbg_name) : width(width), height(height)
    {
        this->image = sg_make_image(sg_image_desc{
            .width = width,
            .height = height,
            .pixel_format = pixel_format_from_channels(channels),
            .label = dbg_name,
            .num_mipmaps = 1,
            .data.mip_levels[0] = { .ptr = data, .size = sizeof(float) * channels * width * height },
        });
        this->view_sample = sg_make_view(sg_view_desc{ .texture.image = this->image });
    }
    // render target attachment
    Texture(int width, int height, const char *dbg_name) : width(width), height(height)
    {
        this->image = sg_make_image(sg_image_desc{
            .width = width,
            .height = height,
            .pixel_format = pixel_format_from_channels(4),
            .label = dbg_name,
            .num_mipmaps = 1,
            .sample_count = 1,
            .usage.color_attachment = true,
        });
        this->view_sample = sg_make_view(sg_view_desc{ .texture.image = this->image });
        this->view_attachment = sg_make_view(sg_view_desc{ .color_attachment.image = this->image });
    }

    ~Texture()
    {
        sg_destroy_view(view_sample);
        sg_destroy_view(view_attachment);
        sg_destroy_image(image);
    }
    int width = 0;
    int height = 0;
    sg_image image = {};
    sg_view view_sample = {};
    sg_view view_attachment = {};
};

static struct
{
    sg_pass_action pass_action;
    sg_pass_action pass_action_dontcare;
    Texture *tex_source = nullptr;
    Texture *tex_blurred_tmp = nullptr;
    Texture *tex_blurred = nullptr;
    sg_pipeline pip;
    sg_pipeline pip_gaussian;
    sg_pipeline pip_dk_down, pip_dk_up, pip_dk_mix;
    struct
    {
        sg_sampler linear;
        sg_sampler nearest;
    } smp;
    struct
    {
        int width;
        int height;
        std::string name;
    } img_info;
    struct
    {
        int blur_mode = BLUR_GAUSSIAN;
        bool show_original = false;
        int blur_x = 20.0f;
        int blur_y = 20.0f;
        bool lock_xy = true;
    } ui;
} state;

static void ui_draw();
static void apply_viewport();
static void load_exr_file(const char *filepath);

static void init()
{
    {
        sg_desc desc = {
            .environment = sglue_environment(),
            .logger.func = slog_func,
        };
        sg_setup(&desc);
    }
    {
        simgui_desc_t desc = {
            .logger.func = slog_func,
        };
        simgui_setup(&desc);
    }
    state.pass_action = {
        .colors[0] = { .load_action = SG_LOADACTION_CLEAR, .clear_value = { 0.0f, 0.0f, 0.0f, 1.0f } },
    };
    state.pass_action_dontcare = { .colors[0] = { .load_action = SG_LOADACTION_DONTCARE } };

    // a render pipeline
    {
        sg_pipeline_desc desc = {
            .shader = sg_make_shader(display_tex_shader_desc(sg_query_backend())),
            .primitive_type = SG_PRIMITIVETYPE_TRIANGLE_STRIP,
            .label = "pipe-display-tex",
        };
        state.pip = sg_make_pipeline(&desc);
    }
    {
        sg_sampler_desc desc = {
            .min_filter = SG_FILTER_LINEAR,
            .mag_filter = SG_FILTER_LINEAR,
            .mipmap_filter = SG_FILTER_NEAREST,
            .wrap_u = SG_WRAP_CLAMP_TO_EDGE,
            .wrap_v = SG_WRAP_CLAMP_TO_EDGE,
            .label = "linear-sampler",
        };
        state.smp.linear = sg_make_sampler(&desc);
    }
    {
        sg_sampler_desc desc = {
            .min_filter = SG_FILTER_NEAREST,
            .mag_filter = SG_FILTER_NEAREST,
            .mipmap_filter = SG_FILTER_NEAREST,
            .wrap_u = SG_WRAP_CLAMP_TO_EDGE,
            .wrap_v = SG_WRAP_CLAMP_TO_EDGE,
            .label = "nearest-sampler",
        };
        state.smp.nearest = sg_make_sampler(&desc);
    }

    // pipelines for blurring
    state.pip_gaussian = sg_make_pipeline(sg_pipeline_desc{
        .shader = sg_make_shader(blur_gaussian_shader_desc(sg_query_backend())),
        .colors[0].pixel_format = SG_PIXELFORMAT_RGBA32F,
        .depth.pixel_format = SG_PIXELFORMAT_NONE,
        .primitive_type = SG_PRIMITIVETYPE_TRIANGLE_STRIP,
        .label = "pipe-blur-gaussian",
    });
    state.pip_dk_down = sg_make_pipeline(sg_pipeline_desc{
        .shader = sg_make_shader(blur_dk_down_shader_desc(sg_query_backend())),
        .colors[0].pixel_format = SG_PIXELFORMAT_RGBA32F,
        .depth.pixel_format = SG_PIXELFORMAT_NONE,
        .primitive_type = SG_PRIMITIVETYPE_TRIANGLE_STRIP,
        .label = "pipe-dk-down",
    });
    state.pip_dk_up = sg_make_pipeline(sg_pipeline_desc{
        .shader = sg_make_shader(blur_dk_up_shader_desc(sg_query_backend())),
        .colors[0].pixel_format = SG_PIXELFORMAT_RGBA32F,
        .depth.pixel_format = SG_PIXELFORMAT_NONE,
        .primitive_type = SG_PRIMITIVETYPE_TRIANGLE_STRIP,
        .label = "pipe-dk-down",
    });
    state.pip_dk_mix = sg_make_pipeline(sg_pipeline_desc{
        .shader = sg_make_shader(blur_dk_mix_shader_desc(sg_query_backend())),
        .colors[0].pixel_format = SG_PIXELFORMAT_RGBA32F,
        .depth.pixel_format = SG_PIXELFORMAT_NONE,
        .primitive_type = SG_PRIMITIVETYPE_TRIANGLE_STRIP,
        .label = "pipe-dk-down",
    });

    load_exr_file("exr/test.exr");
}

static void frame()
{
    simgui_frame_desc_t frame_desc = {
        .width = sapp_width(),
        .height = sapp_height(),
        .delta_time = sapp_frame_duration(),
        .dpi_scale = sapp_dpi_scale(),
    };
    simgui_new_frame(&frame_desc);
    ui_draw();

    sg_pass pass = { .action = state.pass_action, .swapchain = sglue_swapchain() };
    sg_begin_pass(&pass);

    apply_viewport();
    if (state.tex_source)
    {
        sg_apply_pipeline(state.pip);
        {
            sg_bindings bind = {
                .views[VIEW_tex] =
                    state.ui.show_original ? state.tex_source->view_sample : state.tex_blurred->view_sample,
                .samplers[SMP_smp] = state.smp.nearest,
            };
            sg_apply_bindings(&bind);
        }
        sg_draw(0, 4, 1);
    }
    simgui_render();
    sg_end_pass();
    sg_commit();
}

// ======== Gaussian blur

static float gauss_kernel_value(float x)
{
    constexpr float scale = 1.6f;
    constexpr float two_scale2 = 2.0f * scale * scale;
    x = fabsf(x);
    x *= 3.0f * scale;
    return 1.0f / sqrtf(float(M_PI) * two_scale2) * expf(-x * x / two_scale2);
}

static Texture *calc_gaussian_weights(float radius)
{
    // Kernel size is radius+1, but since it is symmetric we only store
    // one half.
    const int size = ceilf(radius) + 1;
    std::vector<float> result(size);

    float sum = 0.0f;

    // Center weight
    const float center_weight = gauss_kernel_value(0.0f);
    result[0] = center_weight;
    sum += center_weight;

    // Other weights in the positive direction. Add double to the sum, to account for
    // the negative direction as well.
    const float scale = radius > 0.0f ? 1.0f / radius : 0.0f;
    for (int i = 1; i < size; ++i)
    {
        const float weight = gauss_kernel_value(i * scale);
        result[i] = weight;
        sum += weight * 2.0f;
    }

    // Normalize the weights
    for (int i = 0; i < size; ++i)
    {
        result[i] /= sum;
    }

    return new Texture(size, 1, 1, result.data(), "gaussian-kernel");
}

static void gaussian_pass(bool horizontal, float radius)
{
    Texture *weights = calc_gaussian_weights(radius);

    const fs_gaussian_params_t par = {
        .uv_step[0] = horizontal ? 1.0f / state.img_info.width : 0.0f,
        .uv_step[1] = horizontal ? 0.0f : 1.0f / state.img_info.height,
        .kernel_width = weights->width,
    };
    sg_pass pass = {
        .action = {
            .colors[0] = {
                .load_action = SG_LOADACTION_DONTCARE,
            },
        },
        .attachments = {
            .colors[0] = horizontal ? state.tex_blurred_tmp->view_attachment : state.tex_blurred->view_attachment,
        },
    };
    sg_begin_pass(&pass);
    sg_apply_pipeline(state.pip_gaussian);

    {
        sg_bindings bind = {
            .views[VIEW_tex] = horizontal ? state.tex_source->view_sample : state.tex_blurred_tmp->view_sample,
            .views[VIEW_tex_kernel] = weights->view_sample,
            .samplers[SMP_smp] = state.smp.nearest,
        };
        sg_apply_bindings(&bind);
    }
    sg_apply_uniforms(UB_fs_gaussian_params, SG_RANGE(par));
    sg_draw(0, 4, 1);
    sg_end_pass();

    delete weights;
}

static void blur_gaussian()
{
    gaussian_pass(true, state.ui.blur_x);
    gaussian_pass(false, state.ui.blur_y);
}

// ======== Dual Kawase blur

static Texture *kawase_downsample(const Texture &input, int full_size_x, int full_size_y, int divisor, float ratio)
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
    sg_begin_pass(sg_pass{ .action = state.pass_action_dontcare,
        .attachments = {
            .colors[0] = output->view_attachment,
        } });
    sg_apply_pipeline(state.pip_dk_down);
    sg_apply_bindings(sg_bindings{
        .views[VIEW_tex] = input.view_sample,
        .samplers[SMP_smp] = state.smp.linear,
    });
    sg_apply_uniforms(UB_fs_dk_down_params, SG_RANGE(par));
    sg_draw(0, 4, 1);
    sg_end_pass();
    return output;
}

static Texture *kawase_upsample(
    const Texture &input, int full_size_x, int full_size_y, int divisor, float ratio, Texture *output)
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
    sg_begin_pass(sg_pass{ .action = state.pass_action_dontcare,
        .attachments = {
            .colors[0] = output->view_attachment,
        } });
    sg_apply_pipeline(state.pip_dk_up);
    sg_apply_bindings(sg_bindings{
        .views[VIEW_tex] = input.view_sample,
        .samplers[SMP_smp] = state.smp.linear,
    });
    sg_apply_uniforms(UB_fs_dk_up_params, SG_RANGE(par));
    sg_draw(0, 4, 1);
    sg_end_pass();
    return output;
}

static Texture *kawase_mix(const Texture &input1, const Texture &input2, float ratio)
{
    Texture *output = new Texture(input1.width, input1.height, "kawase_mix");
    const fs_dk_mix_params_t par = {
        .ratio = ratio,
    };
    sg_begin_pass(sg_pass{ .action = state.pass_action_dontcare,
        .attachments = {
            .colors[0] = output->view_attachment,
        } });
    sg_apply_pipeline(state.pip_dk_mix);
    sg_apply_bindings(sg_bindings{
        .views[VIEW_tex] = input1.view_sample,
        .views[VIEW_tex2] = input2.view_sample,
        .samplers[SMP_smp] = state.smp.nearest,
    });
    sg_apply_uniforms(UB_fs_dk_mix_params, SG_RANGE(par));
    sg_draw(0, 4, 1);
    sg_end_pass();
    return output;
}

static void blur_dual_kawase()
{
    // Dual Kawase is isotropic, so blur based on max(x,y)
    float radius = std::max(state.ui.blur_x, state.ui.blur_y);

    // Amount of Kawase "steps" to do; this more or less matches the blur
    // amount of Gaussian.
    float num_steps = radius / 3.0f;

    const int full_size_x = state.img_info.width;
    const int full_size_y = state.img_info.height;

    Texture *curr = state.tex_source;

    // Downsample
    int last_pass = 1;
    for (int i = 2; i <= num_steps; i *= 2)
    {
        Texture *tmp = kawase_downsample(*curr, full_size_x, full_size_y, i, 1.0f);
        if (curr != state.tex_source)
            delete curr;
        curr = tmp;
        last_pass = i;
    }

    float residual = num_steps - last_pass;
    if (residual > 0.0f)
    {
        int next_pass = last_pass * 2;
        float ratio = residual / (next_pass - last_pass);

        // Downsample and upsample one more step
        Texture *extra_down = kawase_downsample(*curr, full_size_x, full_size_y, next_pass, 0.5f + 0.5f * ratio);
        Texture *extra_up =
            kawase_upsample(*extra_down, full_size_x, full_size_y, last_pass, 0.5f + 0.5f * ratio, nullptr);
        delete extra_down;

        // Mix current with that extra step based on ratio
        Texture *tmp = kawase_mix(*curr, *extra_up, ratio);
        delete extra_up;
        if (curr != state.tex_source)
            delete curr;
        curr = tmp;
    }

    // Upsample.
    for (int i = last_pass / 2; i >= 1; i /= 2)
    {
        bool is_last = i == 1;
        Texture *tmp = kawase_upsample(*curr, full_size_x, full_size_y, i, 1.0f, is_last ? state.tex_blurred : nullptr);
        if (curr != state.tex_source)
            delete curr;
        curr = tmp;
    }
}

static void update_blur()
{
    switch (state.ui.blur_mode)
    {
    case BLUR_GAUSSIAN:
        blur_gaussian();
        break;
    case BLUR_DUAL_KAWASE:
        blur_dual_kawase();
        break;
    }
}

static const char *get_filename_part(const char *path)
{
    const char *slash = strrchr(path, '/');
    const char *rslash = strrchr(path, '\\');
    if (slash && rslash)
    {
        return slash > rslash ? slash + 1 : rslash + 1;
    }
    if (slash)
        return slash + 1;
    if (rslash)
        return rslash + 1;
    return path;
}

static void load_exr_file(const char *filepath)
{
    float *img = nullptr;
    int width;
    int height;
    const char *err = nullptr;
    int ret = LoadEXR(&img, &width, &height, filepath, &err);
    if (ret != TINYEXR_SUCCESS)
    {
        printf("FAILED to read input exr: %s: %s\n", filepath, err);
        FreeEXRErrorMessage(err);
        return;
    }

    state.img_info.width = width;
    state.img_info.height = height;
    state.img_info.name = get_filename_part(filepath);

    delete state.tex_source;
    state.tex_source = new Texture(width, height, 4, img, "source-image");
    free(img);

    delete state.tex_blurred;
    delete state.tex_blurred_tmp;
    state.tex_blurred = new Texture(width, height, "blurred-image");
    state.tex_blurred_tmp = new Texture(width, height, "blurred-tmp");

    update_blur();
}

static void input(const sapp_event *ev)
{
    simgui_handle_event(ev);
    if (ev->type == SAPP_EVENTTYPE_FILES_DROPPED)
    {
        load_exr_file(sapp_get_dropped_file_path(0));
    }
}

static void cleanup()
{
    simgui_shutdown();
    sg_shutdown();
}

static void ui_draw()
{
    ImGui::SetNextWindowPos((ImVec2){ 30, 50 }, ImGuiCond_Once);
    ImGui::SetNextWindowBgAlpha(0.75f);
    if (ImGui::Begin("Controls", 0, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::Text("Width:   %d", state.img_info.width);
        ImGui::Text("Height:  %d", state.img_info.height);
        ImGui::Text("File:    %s", state.img_info.name.c_str());
        ImGui::Separator();

        bool changed = false;

        changed |= ImGui::RadioButton("Gaussian", &state.ui.blur_mode, BLUR_GAUSSIAN);
        ImGui::SameLine();
        changed |= ImGui::RadioButton("Dual Kawase", &state.ui.blur_mode, BLUR_DUAL_KAWASE);

        changed |= ImGui::SliderInt("Blur X", &state.ui.blur_x, 0, 2000, nullptr, ImGuiSliderFlags_Logarithmic);
        ImGui::BeginDisabled(state.ui.lock_xy);
        changed |= ImGui::SliderInt("Blur Y", &state.ui.blur_y, 0, 2000, nullptr, ImGuiSliderFlags_Logarithmic);
        ImGui::EndDisabled();
        changed |= ImGui::Checkbox("Lock X&Y", &state.ui.lock_xy);
        ImGui::Checkbox("Show Original", &state.ui.show_original);
        if (state.ui.lock_xy)
        {
            state.ui.blur_y = state.ui.blur_x;
        }

        if (changed)
        {
            update_blur();
        }

        // ImGui::Separator();
        // sg_stats stats = sg_query_stats();
        // ImGui::Text("Textures: %d", stats.total.images.alive);
    }
    ImGui::End();
}

// set viewport to keep image aspect ratio correct regardless of window size
static void apply_viewport()
{
    if ((state.img_info.width == 0) || (state.img_info.height == 0))
    {
        return;
    }
    const float border = 5.0f;
    float canvas_width = sapp_widthf() - 2.0f * border;
    float canvas_height = sapp_heightf() - 2.0f * border;
    if (canvas_width < 1.0f)
    {
        canvas_width = 1.0f;
    }
    if (canvas_height < 1.0f)
    {
        canvas_height = 1.0f;
    }
    const float canvas_aspect = canvas_width / canvas_height;
    const float img_width = (float)state.img_info.width;
    const float img_height = (float)state.img_info.height;
    const float img_aspect = img_width / img_height;
    float vp_x, vp_y, vp_w, vp_h;
    if (img_aspect < canvas_aspect)
    {
        vp_y = border;
        vp_h = canvas_height;
        vp_w = canvas_height * img_aspect;
        vp_x = border + (canvas_width - vp_w) * 0.5f;
    }
    else
    {
        vp_x = border;
        vp_w = canvas_width;
        vp_h = canvas_width / img_aspect;
        vp_y = border + (canvas_height - vp_h) * 0.5f;
    }
    sg_apply_viewportf(vp_x, vp_y, vp_w, vp_h, true);
}

sapp_desc sokol_main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;
    return (sapp_desc){
        .init_cb = init,
        .frame_cb = frame,
        .cleanup_cb = cleanup,
        .event_cb = input,
        .width = 800,
        .height = 600,
        .window_title = "Blur Playground",
        .icon.sokol_default = true,
        .logger.func = slog_func,
        .enable_dragndrop = true,
        .max_dropped_files = 1,
    };
}
