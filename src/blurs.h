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

struct BlurParams
{
    BlurMode mode = BLUR_DUAL_KAWASE;
    int radius_x = 20;
    int radius_y = 20;
};

struct BlurContext;
BlurContext *blur_ctx_initialize();
void blur_ctx_cleanup(BlurContext *ctx);

void blur_calc(BlurContext *ctx, Texture *input, Texture *output, const BlurParams &params);
