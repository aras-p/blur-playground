#pragma once

#include "../lib/sokol/sokol_gfx.h"

static inline sg_pixel_format pixel_format_from_channels(int channels)
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
