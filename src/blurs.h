#pragma once

struct Texture;

enum BlurMode
{
    BLUR_BOX = 0,
    BLUR_TENT = 1,
    BLUR_GAUSSIAN = 2,
    BLUR_DUAL_KAWASE = 3,
};

struct BlurContext;
BlurContext *blur_ctx_initialize();
void blur_ctx_cleanup(BlurContext *ctx);

void blur_calc(BlurContext *ctx, Texture *input, Texture *output, BlurMode mode, float radius_x, float radius_y);
