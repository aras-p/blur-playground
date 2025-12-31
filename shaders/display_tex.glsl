@vs vs
const vec2 pos[4] = {
    vec2(-1.0, -1.0),
    vec2(+1.0, -1.0),
    vec2(-1.0, +1.0),
    vec2(+1.0, +1.0),
};
out vec2 uv;

void main() {
    gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
    uv = pos[gl_VertexIndex] * vec2(0.5, -0.5) + 0.5;
}
@end

@fs fs

vec3 linear_to_srgb(vec3 linear) { return pow(linear, vec3(1.0/2.2)); }
    
in vec2 uv;
//layout(binding=0) uniform fs_params {
//    float mip_lod;
//};
layout(binding=0) uniform texture2D tex;
layout(binding=0) uniform sampler smp;
out vec4 frag_color;

void main() {
    //frag_color = textureLod(sampler2D(tex, smp), uv, mip_lod);
    vec4 col = texture(sampler2D(tex, smp), uv);
    col.rgb = linear_to_srgb(col.rgb);
    frag_color = col;
}
@end

@program display_tex vs fs
