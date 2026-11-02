#include <metal_stdlib>
using namespace metal;

#include "../../GrassShaderTypes/include/GrassShaderTypes.h"

/// Rotates `v` around the unit vector `axis` by `angle` radians (Rodrigues' rotation formula).
static float3 rotate(float3 v, float3 axis, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

/// A cheap per-blade pseudo-random number in [0, 1).
static float hash(uint seed, float salt) {
    return fract(sin(float(seed) * salt) * 43758.5453);
}

static float hash2(float2 c) {
    return fract(sin(dot(c, float2(127.1, 311.7))) * 43758.5453);
}

/// Smooth value noise in [0, 1], used to vary blades in patches rather than one by one.
static float valueNoise(float2 p) {
    float2 cell = floor(p);
    float2 f = fract(p);
    float2 w = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash2(cell), hash2(cell + float2(1, 0)), w.x),
               mix(hash2(cell + float2(0, 1)), hash2(cell + float2(1, 1)), w.x), w.y);
}

/// Builds the rest pose of every blade and writes the mesh indices.
///
/// Each blade is a strip of three quads (eight vertices) that tapers to a point and
/// curls progressively towards the tip. Every blade gets a random yaw, lean, height and
/// curl, and height also varies in larger patches so the field is not a uniform carpet.
///
///     v6 -- v7    tip (pointed)
///     |  \   |
///     v4 -- v5
///     |  \   |
///     v2 -- v3
///     |  \   |
///     v0 -- v1    root
///
/// UVs: `v` runs from 0 at the root to 1 at the tip. `u` is a per-blade random value, the
/// same for all of the blade's vertices, so a texture's horizontal axis can hold color
/// variations and its vertical axis a root-to-tip gradient.
kernel void generateGrassBlades(device GrassVertex *restVertices [[buffer(0)]],
                                device uint *indices [[buffer(1)]],
                                constant float3 *roots [[buffer(2)]],
                                constant GrassParams &params [[buffer(3)]],
                                uint blade [[thread_position_in_grid]]) {
    if (blade >= params.bladeCount) { return; }

    const float3 up = float3(0, 1, 0);
    float3 root = roots[blade];

    float yaw = hash(blade, 12.9898) * 2.0 * M_PI_F;
    float3 facing = float3(sin(yaw), 0, cos(yaw));
    float3 right = float3(cos(yaw), 0, -sin(yaw));

    float patch = 0.7 + 0.6 * valueNoise(root.xz * 0.7);
    float height = params.bladeHeight * patch * (0.65 + hash(blade, 78.233) * 0.7);
    float width = params.bladeWidth * (0.75 + hash(blade, 39.3468) * 0.5);
    float curl = params.bladeCurvature * (0.4 + hash(blade, 93.9898) * 1.2);

    // A slight random lean so blades are not all perfectly vertical.
    float leanAngle = hash(blade, 51.7241) * 0.3;
    float leanYaw = hash(blade, 17.1717) * 2.0 * M_PI_F;
    float3 leanAxis = float3(cos(leanYaw), 0, sin(leanYaw));
    float3 stem = rotate(up, leanAxis, leanAngle);

    // Segment lengths as fractions of the height, and widths as fractions of the base width.
    const float lengths[3] = { 0.42, 0.33, 0.25 };
    const float widths[4] = { 1.0, 0.82, 0.5, 0.02 };

    float tone = hash(blade, 63.7264);
    // Tilting normals towards the sky softens the lighting, so blades read as a lawn instead
    // of shiny cards. Each edge also leans outwards a little, so a blade looks rounded.
    float3 baseNormal = normalize(facing * 0.55 + up * 0.85);

    uint firstVertex = blade * GRASS_VERTICES_PER_BLADE;
    float3 center = root;
    float along = 0;
    for (uint row = 0; row < 4; row++) {
        float halfWidth = width * widths[row] * 0.5;
        GrassVertex leftEdge = { center - right * halfWidth, normalize(baseNormal - right * 0.35), float2(tone, along) };
        GrassVertex rightEdge = { center + right * halfWidth, normalize(baseNormal + right * 0.35), float2(tone, along) };
        uint v = firstVertex + row * 2;
        restVertices[v] = leftEdge;
        restVertices[v + 1] = rightEdge;

        if (row < 3) {
            float3 direction = rotate(stem, right, curl * float(row + 1));
            center += direction * height * lengths[row];
            along += lengths[row];
        }
    }

    uint i = blade * GRASS_INDICES_PER_BLADE;
    for (uint quad = 0; quad < 3; quad++) {
        uint v = firstVertex + quad * 2;
        indices[i++] = v;     indices[i++] = v + 1; indices[i++] = v + 2;
        indices[i++] = v + 2; indices[i++] = v + 1; indices[i++] = v + 3;
    }
}

/// Sways every blade in the wind by rotating it around its root.
///
/// The motion has two parts: gusts that roll across the field as a wave along the wind
/// direction, pushing blades downwind, and a small per-blade flutter. Higher vertices bend
/// more (quadratically).
kernel void updateGrassVertices(device const GrassVertex *restVertices [[buffer(0)]],
                                device GrassVertex *vertices [[buffer(1)]],
                                constant GrassParams &params [[buffer(2)]],
                                uint vertexID [[thread_position_in_grid]]) {
    if (vertexID >= params.bladeCount * GRASS_VERTICES_PER_BLADE) { return; }

    uint blade = vertexID / GRASS_VERTICES_PER_BLADE;
    uint firstVertex = blade * GRASS_VERTICES_PER_BLADE;

    GrassVertex rest = restVertices[vertexID];
    float3 root = (restVertices[firstVertex].position + restVertices[firstVertex + 1].position) * 0.5;
    float along = rest.uv.y;

    float3 wind = params.windDirection;
    float gustPhase = dot(root.xz, wind.xz) * (2.0 * M_PI_F / params.windGustLength) - params.time * params.windFrequency;
    float gust = 0.5 + 0.5 * sin(gustPhase + 0.6 * sin(gustPhase * 0.37));
    float flutter = sin(params.time * params.windFrequency * 3.1 + hash(blade, 789.123) * 2.0 * M_PI_F);
    float sway = params.windStrength * along * along * (0.25 + 0.75 * gust + 0.15 * flutter);

    float3 axis = normalize(cross(float3(0, 1, 0), wind));

    GrassVertex result;
    result.position = root + rotate(rest.position - root, axis, sway);
    result.normal = rotate(rest.normal, axis, sway);
    result.uv = rest.uv;
    vertices[vertexID] = result;
}
