#pragma once

struct Texture;

enum BlurMode
{
    BLUR_BOX,
    BLUR_TENT,
    BLUR_GAUSSIAN,
    BLUR_DUAL_KAWASE,
    BLUR_SPLIT_KAWASE,
};
static_assert(sizeof(BlurMode) == sizeof(int));

struct FilterConfig
{
    float w00; // center
    float w05; // +/- 0.5
    float w10; // +/- 1.0
    float w15; // +/- 1.5
    float w20; // +/- 2.0
};

struct BlurParams
{
    BlurMode mode = BLUR_SPLIT_KAWASE;
    int radius_x = 20;
    int radius_y = 20;

    FilterConfig split_cfg_down = { 2.120f, 0.750f, 1.300f, 0.100f, 0.150f };
    FilterConfig split_cfg_up = { 0.000f, 2.800f, 0.700f, 0.000f, 0.010f };
};

struct BlurContext;
BlurContext *blur_ctx_initialize();
void blur_ctx_cleanup(BlurContext *ctx);

void blur_calc(BlurContext *ctx, Texture *input, Texture *output, const BlurParams &params);
