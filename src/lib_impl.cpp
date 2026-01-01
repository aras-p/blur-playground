
#if defined(__APPLE__)
#define SOKOL_METAL
#elif defined(_WIN32)
#define SOKOL_D3D11
#else
#define SOKOL_GLCORE
#endif

#define SOKOL_IMPL

#include "../lib/sokol/sokol_app.h"
#include "../lib/sokol/sokol_gfx.h"
#include "../lib/sokol/sokol_glue.h"
#include "../lib/sokol/sokol_log.h"
#include "../lib/sokol/sokol_time.h"

#include "../lib/imgui/imgui.h"
#include "../lib/sokol/util/sokol_imgui.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_BMP
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STBI_NO_GIF
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_PNM
#include "../lib/stb/stb_image.h"
