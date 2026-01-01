#pragma once

struct Texture;

struct BlurContext;

BlurContext *blur_ctx_initialize();
void blur_ctx_cleanup(BlurContext *ctx);

void blur_gaussian(BlurContext *ctx, Texture *input, Texture *output, float radius_x, float radius_y);
void blur_dual_kawase(BlurContext *ctx, Texture *input, Texture *output, float radius_x, float radius_y);
