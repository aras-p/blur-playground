#pragma sokol @cs cs_rmse
layout(binding = 0, rgba32f) uniform image2D err_tex1;
layout(binding = 1, rgba32f) uniform image2D err_tex2;
struct RmseVal {
    float val;
};
layout(binding = 2) buffer err_rmse_out { RmseVal rmse_out[]; };

layout(binding=0) uniform cs_rmse_params {
    int numGroupsX;
};

#define GRP_SIZE 16

shared float sharedData[GRP_SIZE * GRP_SIZE];

layout(local_size_x=GRP_SIZE, local_size_y=GRP_SIZE, local_size_z=1) in;
void main()
{
    ivec2 texSize = imageSize(err_tex1);
    ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);
    uint localIndex = gl_LocalInvocationID.y * GRP_SIZE + gl_LocalInvocationID.x;

    // Load pixels, compute squared diff, store in shared memory
    float squaredDiff = 0.0;
    if (pixelCoord.x < texSize.x && pixelCoord.y < texSize.y) {
        vec3 col1 = imageLoad(err_tex1, pixelCoord).rgb;
        vec3 col2 = imageLoad(err_tex2, pixelCoord).rgb;
        vec3 diff = col1 - col2;
        squaredDiff = dot(diff, diff);
    }
    sharedData[localIndex] = squaredDiff;
    barrier();
    
    // Parallel sum of diffs in shared mem
    for (uint stride = GRP_SIZE * GRP_SIZE / 2; stride > 0; stride >>= 1) {
        if (localIndex < stride) {
            sharedData[localIndex] += sharedData[localIndex + stride];
        }
        barrier();
    }
    
    // First thread writes out group's sum
    if (localIndex == 0) {
        uint workgroupIndex = gl_WorkGroupID.y * numGroupsX + gl_WorkGroupID.x;
        rmse_out[workgroupIndex].val = sharedData[0];
    }
}
#pragma sokol @end

#pragma sokol @program comp_rmse cs_rmse
