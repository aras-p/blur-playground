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

#pragma sokol @fs fs_gaussian

in vec2 uv;
layout(binding = 0) uniform fs_gaussian_params
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

#pragma sokol @program blur_gaussian vs_blur fs_gaussian
