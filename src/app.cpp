
#include "blurs.h"
#include "texture.h"

#include "../lib/imgui/imgui.h"
#include "../lib/sokol/sokol_app.h"
#include "../lib/sokol/sokol_glue.h"
#include "../lib/sokol/sokol_log.h"
#include "../lib/sokol/sokol_time.h"
#include "../lib/sokol/util/sokol_imgui.h"
#include "../lib/stb/stb_image.h"
#include "../lib/tinyexr/tinyexr.h"

#include "../shaders/display_tex.glsl.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
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
        BlurParams blur;
        bool show_original = false;
        bool log_slider = true;
        bool lock_xy = false;
        std::vector<std::string> image_files;
        int selected_file_index = -1;
        std::string dragged_file_name;
    } ui;
    struct
    {
        bool running = false;
        int current_index = 0;
        int total_cases = 0;
        double start_time = 0.0;
        double result_time = 0.0;
        bool has_result = false;
        std::vector<int> values;
        BlurParams prev_blur;
        bool prev_lock_xy = false;
    } bench;
} state;

static void ui_draw();
static void apply_viewport();
static void load_image_file(const char *filepath);
static void scan_image_files();
static void update_blur();

static void init()
{
    stm_setup();
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

    scan_image_files();
    if (!state.ui.image_files.empty())
    {
        state.ui.selected_file_index = 0;
        load_image_file(state.ui.image_files[0].c_str());
    }
}

static bool is_image_file(const std::filesystem::path &path)
{
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".exr" || ext == ".png" || ext == ".jpg" || ext == ".jpeg";
}

static void scan_image_files()
{
    state.ui.image_files.clear();

    auto scan_directory = [&](const std::filesystem::path &dir)
    {
        for (const auto &entry : std::filesystem::directory_iterator(dir))
        {
            if (entry.is_regular_file() && is_image_file(entry.path()))
            {
                state.ui.image_files.push_back(entry.path().string());
            }
        }
    };

    try
    {
        std::filesystem::path current_dir = std::filesystem::current_path();
        scan_directory(current_dir);
        for (const auto &entry : std::filesystem::directory_iterator(current_dir))
        {
            if (entry.is_directory())
            {
                scan_directory(entry.path());
            }
        }
    }
    catch (const std::filesystem::filesystem_error &)
    {
        // ignore errors
    }

    std::sort(state.ui.image_files.begin(), state.ui.image_files.end());
}

static void benchmark_generate_values()
{
    state.bench.values.clear();
    for (float v = 5.0f; v < 1000.0f; v *= 1.2f)
    {
        state.bench.values.push_back(std::max(1, (int)roundf(v)));
    }
}

static void benchmark_start()
{
    if (!state.tex_source)
        return;

    benchmark_generate_values();
    state.bench.prev_blur = state.ui.blur;
    state.bench.prev_lock_xy = state.ui.lock_xy;
    state.bench.running = true;
    state.bench.current_index = 0;
    int n = (int)state.bench.values.size();
    state.bench.total_cases = n * n;
    state.bench.start_time = stm_sec(stm_now());
    state.bench.has_result = false;
}

static void benchmark_stop()
{
    state.bench.running = false;
    state.bench.result_time = stm_sec(stm_now()) - state.bench.start_time;
    state.bench.has_result = true;
    // Restore blur to previous values
    state.ui.blur = state.bench.prev_blur;
    state.ui.lock_xy = state.bench.prev_lock_xy;
    update_blur();
}

static void benchmark_run_step()
{
    if (!state.bench.running)
        return;

    const int cases_per_frame = 10;
    int n = (int)state.bench.values.size();

    for (int i = 0; i < cases_per_frame && state.bench.current_index < state.bench.total_cases; ++i)
    {
        int idx = state.bench.current_index;
        int xi = idx % n;
        int yi = idx / n;
        int blur_x = state.bench.values[xi];
        int blur_y = state.bench.values[yi];
        state.ui.blur.radius_x = blur_x;
        state.ui.blur.radius_y = blur_y;
        state.ui.lock_xy = false;

        blur_calc(state.blur_ctx, state.tex_source, state.tex_blurred, state.ui.blur);

        state.bench.current_index++;
    }

    if (state.bench.current_index >= state.bench.total_cases)
    {
        benchmark_stop();
    }
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

    benchmark_run_step();

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

static void update_blur() { blur_calc(state.blur_ctx, state.tex_source, state.tex_blurred, state.ui.blur); }

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

static void load_image_file(const char *filepath)
{
    float *img = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;

    // first try stb_image
    float *data = stbi_loadf(filepath, &width, &height, &channels, 4);
    if (data)
    {
        delete state.tex_source;
        state.tex_source = new Texture(width, height, 4, data, "source-image");
        stbi_image_free(data);
    }
    else
    {
        // then try tinyexr
        const char *err = nullptr;
        int ret = LoadEXR(&img, &width, &height, filepath, &err);
        if (ret != TINYEXR_SUCCESS)
        {
            printf("FAILED to read input: %s: %s\n", filepath, err);
            FreeEXRErrorMessage(err);
            return;
        }
        delete state.tex_source;
        state.tex_source = new Texture(width, height, 4, img, "source-image");
        free(img);
    }

    delete state.tex_blurred;
    state.tex_blurred = new Texture(width, height, "blurred-image");

    update_blur();
}

static void input(const sapp_event *ev)
{
    simgui_handle_event(ev);
    if (ev->type == SAPP_EVENTTYPE_FILES_DROPPED && !state.bench.running)
    {
        const char *path = sapp_get_dropped_file_path(0);
        load_image_file(path);
        state.ui.dragged_file_name = get_filename_part(path);
        state.ui.selected_file_index = -1;
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
        // Benchmark UI
        if (state.bench.running)
        {
            float progress = (float)state.bench.current_index / (float)state.bench.total_cases;
            ImGui::Text("Benchmark running...");
            ImGui::ProgressBar(progress, ImVec2(-FLT_MIN, 0), nullptr);
            ImGui::Text("%d / %d cases", state.bench.current_index, state.bench.total_cases);
            ImGui::Separator();
        }
        else
        {
            ImGui::BeginDisabled(!state.tex_source);
            if (ImGui::Button("Run Benchmark"))
            {
                benchmark_start();
            }
            ImGui::EndDisabled();
            if (state.bench.has_result)
            {
                ImGui::SameLine();
                ImGui::Text("%.2f sec", state.bench.result_time);
            }
            ImGui::Separator();
        }

        // Disable controls during benchmark
        ImGui::BeginDisabled(state.bench.running);

        if (!state.ui.dragged_file_name.empty())
        {
            ImGui::Text("Dropped: %s", state.ui.dragged_file_name.c_str());
        }
        if (!state.ui.image_files.empty())
        {
            ImGui::Text("Files:");
            if (ImGui::BeginListBox("##image_files", ImVec2(-FLT_MIN, 5.25f * ImGui::GetTextLineHeightWithSpacing())))
            {
                for (int i = 0; i < (int)state.ui.image_files.size(); i++)
                {
                    const char *filename = get_filename_part(state.ui.image_files[i].c_str());
                    bool is_selected = (state.ui.selected_file_index == i);
                    if (ImGui::Selectable(filename, is_selected))
                    {
                        state.ui.selected_file_index = i;
                        state.ui.dragged_file_name.clear();
                        load_image_file(state.ui.image_files[i].c_str());
                    }
                    if (is_selected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndListBox();
            }
            ImGui::Separator();
        }

        if (state.tex_source)
        {
            ImGui::Text("Width:   %d", state.tex_source->width);
            ImGui::Text("Height:  %d", state.tex_source->height);
            ImGui::Separator();
        }

        bool changed = false;

        changed |= ImGui::RadioButton("Box", (int *)&state.ui.blur.mode, BLUR_BOX);
        changed |= ImGui::RadioButton("Tent", (int *)&state.ui.blur.mode, BLUR_TENT);
        changed |= ImGui::RadioButton("Gaussian", (int *)&state.ui.blur.mode, BLUR_GAUSSIAN);
        changed |= ImGui::RadioButton("Dual Kawase", (int *)&state.ui.blur.mode, BLUR_DUAL_KAWASE);

        changed |= ImGui::SliderInt("Blur X", &state.ui.blur.radius_x, 0, 2000, nullptr,
            state.ui.log_slider ? ImGuiSliderFlags_Logarithmic : 0);
        ImGui::BeginDisabled(state.ui.lock_xy);
        changed |= ImGui::SliderInt("Blur Y", &state.ui.blur.radius_y, 0, 2000, nullptr,
            state.ui.log_slider ? ImGuiSliderFlags_Logarithmic : 0);
        ImGui::EndDisabled();
        changed |= ImGui::Checkbox("Lock X&Y", &state.ui.lock_xy);
        ImGui::Checkbox("Log Sliders", &state.ui.log_slider);
        ImGui::Checkbox("Show Original", &state.ui.show_original);
        if (state.ui.lock_xy)
        {
            state.ui.blur.radius_y = state.ui.blur.radius_x;
        }

        if (changed)
        {
            update_blur();
        }

        ImGui::EndDisabled(); // End benchmark disable
    }
    ImGui::End();
}

// set viewport to keep image aspect ratio correct regardless of window size
static void apply_viewport()
{
    int width = state.tex_source ? state.tex_source->width : 0;
    int height = state.tex_source ? state.tex_source->height : 0;
    if ((width == 0) || (height == 0))
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
    const float img_width = (float)width;
    const float img_height = (float)height;
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
