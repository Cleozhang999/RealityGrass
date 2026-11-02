#pragma once

#include <simd/simd.h>

/// Number of vertices in one grass blade: four rows of two.
#define GRASS_VERTICES_PER_BLADE 8
/// Number of indices in one grass blade: three quads of two triangles.
#define GRASS_INDICES_PER_BLADE 18

/// One vertex of a grass blade. Must match the `LowLevelMesh` layout in `GrassFieldEntity`.
struct GrassVertex {
    simd_float3 position;
    simd_float3 normal;
    simd_float2 uv;
};

/// Per-field parameters passed to both compute kernels.
struct GrassParams {
    /// Number of blades in the field.
    unsigned int bladeCount;
    /// Average blade height, in meters.
    float bladeHeight;
    /// Average blade width at the base, in meters.
    float bladeWidth;
    /// How far a blade tilts forward along its length, in radians per segment.
    float bladeCurvature;
    /// Seconds since the field was created.
    float time;
    /// Peak wind bend at the tip, in radians.
    float windStrength;
    /// How fast the wind oscillates, in radians per second.
    float windFrequency;
    /// Distance in meters between gust fronts rolling across the field.
    float windGustLength;
    /// Horizontal (XZ) direction the wind blows towards. Normalized.
    simd_float3 windDirection;
};
