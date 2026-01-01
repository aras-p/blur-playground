
#include "blurs.h"
#include "texture.h"

#include "../lib/imgui/imgui.h"
#include "../lib/sokol/sokol_app.h"
#include "../lib/sokol/sokol_glue.h"
#include "../lib/sokol/sokol_log.h"
#include "../lib/sokol/util/sokol_imgui.h"
#include "../lib/tinyexr/tinyexr.h"

#include "../shaders/display_tex.glsl.h"

#include <string>
#include <vector>

static struct
{
    sg_pass_action pass_action;
    sg_pass_action pass_action_dontcare;
    Texture *tex_source = nullptr;
    Texture *tex_blurred = nullptr;
    BlurContext *blur_ctx = nullptr;
    sg_pipeline pip;
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
        bool log_slider = true;
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

    state.blur_ctx = blur_ctx_initialize();

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

static void update_blur()
{
    blur_calc(state.blur_ctx, state.tex_source, state.tex_blurred, BlurMode(state.ui.blur_mode), state.ui.blur_x,
        state.ui.blur_y);
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
    state.tex_blurred = new Texture(width, height, "blurred-image");

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
    blur_ctx_cleanup(state.blur_ctx);
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

        changed |= ImGui::RadioButton("Box", &state.ui.blur_mode, BLUR_BOX);
        changed |= ImGui::RadioButton("Tent", &state.ui.blur_mode, BLUR_TENT);
        changed |= ImGui::RadioButton("Gaussian", &state.ui.blur_mode, BLUR_GAUSSIAN);
        changed |= ImGui::RadioButton("Dual Kawase", &state.ui.blur_mode, BLUR_DUAL_KAWASE);

        changed |= ImGui::SliderInt(
            "Blur X", &state.ui.blur_x, 0, 2000, nullptr, state.ui.log_slider ? ImGuiSliderFlags_Logarithmic : 0);
        ImGui::BeginDisabled(state.ui.lock_xy);
        changed |= ImGui::SliderInt(
            "Blur Y", &state.ui.blur_y, 0, 2000, nullptr, state.ui.log_slider ? ImGuiSliderFlags_Logarithmic : 0);
        ImGui::EndDisabled();
        changed |= ImGui::Checkbox("Lock X&Y", &state.ui.lock_xy);
        ImGui::Checkbox("Log Sliders", &state.ui.log_slider);
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
