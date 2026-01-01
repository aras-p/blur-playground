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

#include "../shaders/display_tex.glsl.h"

#include <string>

static struct
{
    sg_pass_action pass_action;
    sg_image img_src;
    sg_image img_blurred;
    sg_view disp_tex_view;
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
        bool show_original = false;
        int blur_x = 20.0f;
        int blur_y = 20.0f;
        bool lock_xy = true;
    } ui;
} state;

static void ui_draw();
static void reinit_texview();
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

    // pre-allocate handles so we can keep rendering even no image has been loaded yet
    state.img_src = sg_alloc_image();
    state.img_blurred = sg_alloc_image();
    state.disp_tex_view = sg_alloc_view();

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

    // const fs_params_t fs_params = { .mip_lod = state.ui.mip_lod };
    sg_pass pass = { .action = state.pass_action, .swapchain = sglue_swapchain() };
    sg_begin_pass(&pass);
    apply_viewport();
    sg_apply_pipeline(state.pip);

    {
        sg_bindings bind = {
            .views[VIEW_tex] = state.disp_tex_view,
            .samplers[SMP_smp] = state.smp.nearest,
        };
        sg_apply_bindings(&bind);
    }
    // sg_apply_uniforms(UB_fs_params, &SG_RANGE(fs_params));
    sg_draw(0, 4, 1);
    simgui_render();
    sg_end_pass();
    sg_commit();
}

static void reinit_texview()
{
    sg_uninit_view(state.disp_tex_view);
    sg_view_desc desc = {
        .texture = { .image = state.ui.show_original ? state.img_src : state.img_blurred },
    };
    sg_init_view(state.disp_tex_view, &desc);
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

    sg_uninit_image(state.img_src);
    state.img_info.width = width;
    state.img_info.height = height;
    state.img_info.name = get_filename_part(filepath);
    {
        sg_image_desc img_desc = { .width = width,
            .height = height,
            .pixel_format = SG_PIXELFORMAT_RGBA32F,
            .label = "source-image",
            .num_mipmaps = 1,
            .data.mip_levels[0] = { .ptr = img, .size = sizeof(float) * width * height * 4 } };
        sg_init_image(state.img_src, &img_desc);
    }
    free(img);

    sg_uninit_image(state.img_blurred);
    {
        sg_image_desc img_desc = {
            .width = width,
            .height = height,
            .pixel_format = SG_PIXELFORMAT_RGBA32F,
            .usage.color_attachment = true,
            .sample_count = 1,
            .label = "blurred-image",
            .num_mipmaps = 1,
        };
        sg_init_image(state.img_blurred, &img_desc);
    }

    reinit_texview();
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
        // if (ImGui::Combo("Image", &state.ui.selected, files,
        // IM_ARRAYSIZE(files))) {
        //     fetch_async(files[state.ui.selected]);
        // }
        ImGui::Text("Width:   %d", state.img_info.width);
        ImGui::Text("Height:  %d", state.img_info.height);
        ImGui::Text("File:    %s", state.img_info.name.c_str());
        ImGui::Separator();
        ImGui::SliderInt("Blur X", &state.ui.blur_x, 1, 2000, nullptr, ImGuiSliderFlags_Logarithmic);
        ImGui::BeginDisabled(state.ui.lock_xy);
        ImGui::SliderInt("Blur Y", &state.ui.blur_y, 1, 2000, nullptr, ImGuiSliderFlags_Logarithmic);
        ImGui::EndDisabled();
        ImGui::Checkbox("Lock X&Y", &state.ui.lock_xy);
        if (ImGui::Checkbox("Show Original", &state.ui.show_original))
        {
            reinit_texview();
        }
        if (state.ui.lock_xy)
        {
            state.ui.blur_y = state.ui.blur_x;
        }
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
