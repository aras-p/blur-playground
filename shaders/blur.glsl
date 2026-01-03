#pragma sokol @vs vs_blur
const vec2 pos[4] = {
    vec2(-1.0, -1.0),
    vec2(+1.0, -1.0),
    vec2(-1.0, +1.0),
    vec2(+1.0, +1.0),
};
out vec2 uv;

void main()
{
    gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
    uv = pos[gl_VertexIndex] * vec2(0.5, -0.5) + 0.5;
}
#pragma sokol @end

// ======== Separable kernel

#pragma sokol @fs fs_separable
in vec2 uv;
layout(binding = 0) uniform fs_separable_params
{
    vec2 uv_step;
    int kernel_width;
};
layout(binding = 0) uniform texture2D tex;
layout(binding = 1) uniform texture2D tex_kernel;
layout(binding = 0) uniform sampler smp;
out vec4 frag_color;

void main()
{
    vec4 col = texture(sampler2D(tex, smp), uv) * texelFetch(sampler2D(tex_kernel, smp), ivec2(0, 0), 0).r;
    for (int i = 1; i < kernel_width; ++i)
    {
        float w = texelFetch(sampler2D(tex_kernel, smp), ivec2(i, 0), 0).r;
        col += texture(sampler2D(tex, smp), uv - uv_step * i) * w;
        col += texture(sampler2D(tex, smp), uv + uv_step * i) * w;
    }
    frag_color = col;
}
#pragma sokol @end

// ======== Dual Kawase

#pragma sokol @fs fs_dk_down
in vec2 uv;
layout(binding = 0) uniform fs_dk_down_params { vec2 uv_step; };
layout(binding = 0) uniform texture2D tex;
layout(binding = 0) uniform sampler smp;
out vec4 frag_color;

void main()
{
    vec4 col = vec4(0.0);
    col += texture(sampler2D(tex, smp), uv) * 4.0;
    col += texture(sampler2D(tex, smp), uv + uv_step * vec2(-0.5, -0.5));
    col += texture(sampler2D(tex, smp), uv + uv_step * vec2(-0.5, +0.5));
    col += texture(sampler2D(tex, smp), uv + uv_step * vec2(+0.5, -0.5));
    col += texture(sampler2D(tex, smp), uv + uv_step * vec2(+0.5, +0.5));
    frag_color = col / 8.0;
}
#pragma sokol @end

#pragma sokol @fs fs_dk_up
in vec2 uv;
layout(binding = 0) uniform fs_dk_up_params { vec2 uv_step; };
layout(binding = 0) uniform texture2D tex;
layout(binding = 0) uniform sampler smp;
out vec4 frag_color;

void main()
{
    vec4 col = vec4(0.0);
    col += texture(sampler2D(tex, smp), uv + uv_step * vec2(0, -1));
    col += texture(sampler2D(tex, smp), uv + uv_step * vec2(0, +1));
    col += texture(sampler2D(tex, smp), uv + uv_step * vec2(-1, 0));
    col += texture(sampler2D(tex, smp), uv + uv_step * vec2(+1, 0));
    col += texture(sampler2D(tex, smp), uv + uv_step * vec2(-0.5, -0.5)) * 2.0;
    col += texture(sampler2D(tex, smp), uv + uv_step * vec2(-0.5, +0.5)) * 2.0;
    col += texture(sampler2D(tex, smp), uv + uv_step * vec2(+0.5, -0.5)) * 2.0;
    col += texture(sampler2D(tex, smp), uv + uv_step * vec2(+0.5, +0.5)) * 2.0;
    frag_color = col / 12.0;
}
#pragma sokol @end

#pragma sokol @fs fs_dk_mix
in vec2 uv;
layout(binding = 0) uniform fs_dk_mix_params { float ratio; };
layout(binding = 0) uniform texture2D tex;
layout(binding = 0) uniform sampler smp;
out vec4 frag_color;

void main()
{
    vec4 col = texture(sampler2D(tex, smp), uv);
    col.a = ratio;
    frag_color = col;
}
#pragma sokol @end

#pragma sokol @fs fs_dk_copy
in vec2 uv;
layout(binding = 0) uniform texture2D tex;
layout(binding = 0) uniform sampler smp;
out vec4 frag_color;

void main() { frag_color = texture(sampler2D(tex, smp), uv); }
#pragma sokol @end

// ======== Split Kawase

#pragma sokol @fs fs_split_kawase_step
in vec2 uv;
layout(binding = 0) uniform fs_split_kawase_step_params
{
    vec2 uv_step;
    float w00, w05, w10, w15, w20;
};
layout(binding = 0) uniform texture2D tex;
layout(binding = 0) uniform sampler smp;
out vec4 frag_color;

void main()
{
    vec4 col = vec4(0.0);
    vec2 st = uv_step;
    col += texture(sampler2D(tex, smp), uv) * w00;
    col += texture(sampler2D(tex, smp), uv + st * (-0.5)) * w05;
    col += texture(sampler2D(tex, smp), uv + st * (+0.5)) * w05;
    col += texture(sampler2D(tex, smp), uv + st * (-1.0)) * w10;
    col += texture(sampler2D(tex, smp), uv + st * (+1.0)) * w10;
    col += texture(sampler2D(tex, smp), uv + st * (-1.5)) * w15;
    col += texture(sampler2D(tex, smp), uv + st * (+1.5)) * w15;
    col += texture(sampler2D(tex, smp), uv + st * (-2.0)) * w20;
    col += texture(sampler2D(tex, smp), uv + st * (+2.0)) * w20;
    frag_color = col * (1.0f / (w00 + (w05 + w10 + w15 + w20) * 2.0));
}
#pragma sokol @end

#pragma sokol @program blur_separable vs_blur fs_separable
#pragma sokol @program blur_dk_down vs_blur fs_dk_down
#pragma sokol @program blur_dk_up vs_blur fs_dk_up
#pragma sokol @program blur_dk_mix vs_blur fs_dk_mix
#pragma sokol @program blur_dk_copy vs_blur fs_dk_copy
#pragma sokol @program blur_split_kawase_step vs_blur fs_split_kawase_step
