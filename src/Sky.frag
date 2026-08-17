#version 460

#define M_PI    3.14159265358979323846
#define M_1_PI  0.318309886183790671538
#define M_1_2PI 0.5 * M_1_PI

layout(location = 0) in vec2 inTexCoord;

layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform CameraUniformDynamic {
    mat4 view;
    mat4 view_inv;
    mat4 proj;
    mat4 proj_inv;
} camera;

layout(push_constant) uniform PushConstant {
    vec2 viewPort;
    float sunElevation;
    float sunRotation;
    vec3 minBounds;
	int renderMode;
    vec3 maxBounds;
    float exposure;
    float triplaneMipmap;
} pushConstant;

// https://www.scratchapixel.com/lessons/procedural-generation-virtual-worlds/simulating-sky/simulating-colors-of-the-sky.html

const float earthRadius = 6360e3;
const float atmosphereRadius = 6420e3;
const float Hr = 7994.0;	// Thickness of the atmosphere if density was uniform (Hr)
const float Hm = 1200.0;	// Same as above but for Mie scattering (Hm) 
const float PI = radians(180.0);
const vec3 betaR = vec3(3.8e-6, 13.5e-6, 33.1e-6);
const vec3 betaM = vec3(21e-6);
const float inf = 3.402823466e+38;

bool solveQuadratic(float a, float b, float c, inout float x1, inout float x2) 
{ 
    if (b == 0.0) { 
        if (a == 0.0) return false; 
        x1 = 0.0; x2 = sqrt(-c / a); 
        return true; 
    } 
    float discr = b * b - 4.0 * a * c; 
 
    if (discr < 0.0) return false; 
 
    float q = (b < 0.0) ? -0.5 * (b - sqrt(discr)) : -0.5 * (b + sqrt(discr)); 
    x1 = q / a; 
    x2 = c / q; 
 
    return true; 
} 

bool raySphereIntersect(const vec3 orig, const vec3 dir, const float radius, inout float t0, inout float t1) 
{ 
    float A = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z; 
    float B = 2.0 * (dir.x * orig.x + dir.y * orig.y + dir.z * orig.z); 
    float C = orig.x * orig.x + orig.y * orig.y + orig.z * orig.z - radius * radius; 
 
    if (!solveQuadratic(A, B, C, t0, t1)) return false; 
 
    if (t0 > t1) t0 = t1, t1 = t0;
 
    return true; 
} 


vec3 computeIncidentLight(const vec3 orig, const vec3 dir, float tmin, float tmax)
{
    float t0, t1;
    if (!raySphereIntersect(orig, dir, atmosphereRadius, t0, t1) || t1 < 0.0) return vec3(0.0); 
    if (t0 > tmin && t0 > 0.0) tmin = t0; 
    if (t1 < tmax) tmax = t1; 
    uint numSamples = 16u; 
    uint numSamplesLight = 8u; 
    float segmentLength = (tmax - tmin) / float(numSamples); 
    float tCurrent = tmin; 
    vec3 sumR = vec3(0.0), sumM = vec3(0.0); // mie and rayleigh contribution 
    float opticalDepthR = 0.0, opticalDepthM = 0.0;
    vec3 sunDirection = vec3(cos(pushConstant.sunRotation) * cos(pushConstant.sunElevation),
                             sin(pushConstant.sunElevation),
                             sin(pushConstant.sunRotation) * cos(pushConstant.sunElevation));
    float mu = dot(dir, sunDirection); // mu in the paper which is the cosine of the angle between the sun direction and the ray direction 
    float phaseR = 3.0 / (16.0 * PI) * (1.0 + mu * mu); 
    float g = 0.76; 
    float phaseM = 3.0 / (8.0 * PI) * ((1.0 - g * g) * (1.0 + mu * mu)) / ((2.0 + g * g) * pow(1.0 + g * g - 2.0 * g * mu, 1.5)); 
    for (uint i = 0u; i < numSamples; ++i) { 
        vec3 samplePosition = orig + (tCurrent + segmentLength * 0.5) * dir; 
        float height = length(samplePosition) - earthRadius; 
        // compute optical depth for light
        float hr = exp(-height / Hr) * segmentLength; 
        float hm = exp(-height / Hm) * segmentLength; 
        opticalDepthR += hr; 
        opticalDepthM += hm; 
        // light optical depth
        float t0Light, t1Light; 
        raySphereIntersect(samplePosition, sunDirection, atmosphereRadius, t0Light, t1Light); 
        float segmentLengthLight = t1Light / float(numSamplesLight), tCurrentLight = 0.0; 
        float opticalDepthLightR = 0.0, opticalDepthLightM = 0.0; 
        uint j; 
        for (j = 0u; j < numSamplesLight; ++j) { 
            vec3 samplePositionLight = samplePosition + (tCurrentLight + segmentLengthLight * 0.5f) * sunDirection; 
            float heightLight = length(samplePositionLight) - earthRadius; 
            if (heightLight < 0.0) break; 
            opticalDepthLightR += exp(-heightLight / Hr) * segmentLengthLight; 
            opticalDepthLightM += exp(-heightLight / Hm) * segmentLengthLight; 
            tCurrentLight += segmentLengthLight; 
        } 
        if (j == numSamplesLight) { 
            vec3 tau = betaR * (opticalDepthR + opticalDepthLightR) + betaM * 1.1f * (opticalDepthM + opticalDepthLightM); 
            vec3 attenuation = vec3(exp(-tau.x), exp(-tau.y), exp(-tau.z)); 
            sumR += attenuation * hr; 
            sumM += attenuation * hm; 
        } 
        tCurrent += segmentLength; 
    }    
    return (sumR * betaR * phaseR + sumM * betaM * phaseM) * 20.0;
}

void main()
{
	const vec2 ray_nds = 2.0 * gl_FragCoord.xy / pushConstant.viewPort - 1.0;
	const vec4 ray_clip = vec4(ray_nds, -1.0, 1.0);
	vec4 ray_view = camera.proj_inv * ray_clip;
	ray_view = vec4(ray_view.xy, -1.0, 0.0);
	vec3 ray_world = (camera.view_inv * ray_view).xyz;
	ray_world = normalize(ray_world);

    vec3 cameraOrigin = (camera.view_inv * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    vec3 origin = vec3(0.0, earthRadius + 750.0, 0.0);

	fragColor = vec4(computeIncidentLight(origin, ray_world, 0.0, inf), 1.0);

    if (ray_world.y < 0.0) {
        const vec4 horizon = vec4(85.0 / 255.0, 120.0 / 255.0, 189.0 / 255.0, 1.0);
        const vec4 deep = vec4(19.0 / 255.0, 35.0 / 255.0, 69.0 / 255.0, 1.0);
        const vec4 sea = mix(horizon, deep, dot(ray_world, vec3(0.0, -1.0, 0.0)));
        fragColor = mix(vec4(0.02, 0.02, 0.04, 1.0), sea, clamp(pushConstant.sunElevation, 0.0, 1.0));
    }
}
