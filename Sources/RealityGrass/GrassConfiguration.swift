import simd

/// Wind that sways a grass field. You can change it at any time without rebuilding the field.
public struct GrassWind: Sendable, Equatable {
    /// Bend at the blade tip inside a gust, in radians.
    public var strength: Float
    /// How lively the wind is: gust clumps travel about `frequency × gustLength / 2` meters
    /// per second, and blades flutter faster as it rises.
    public var frequency: Float
    /// Direction the wind blows towards. Only the horizontal (XZ) part is used.
    public var direction: SIMD3<Float>
    /// Typical size in meters of the gust clumps moving across the field.
    public var gustLength: Float

    public init(strength: Float = 0.3, frequency: Float = 1.5, direction: SIMD3<Float> = [1, 0, 0], gustLength: Float = 4) {
        self.strength = strength
        self.frequency = frequency
        self.direction = direction
        self.gustLength = gustLength
    }

    public static let calm = GrassWind(strength: 0)
    public static let breeze = GrassWind(strength: 0.3, frequency: 1.5)
    public static let gusty = GrassWind(strength: 0.7, frequency: 3)

    /// `direction` flattened onto the XZ plane and normalized. Falls back to +X when the
    /// direction is vertical or zero, because the shader needs a horizontal wind axis.
    var horizontalDirection: SIMD3<Float> {
        let flat = SIMD3<Float>(direction.x, 0, direction.z)
        let length = simd_length(flat)
        return length > 1e-5 ? flat / length : [1, 0, 0]
    }
}

/// The look of the individual blades.
public struct GrassBladeShape: Sendable, Equatable {
    /// Average blade height in meters. Blades vary individually and in patches, from about 0.45 to 1.75 times this.
    public var height: Float
    /// Average blade width at the base in meters. Each blade varies by ±25 %.
    public var width: Float
    /// How much blades curl over towards the tip, in radians per segment. Each blade varies.
    public var curvature: Float

    public init(height: Float = 0.25, width: Float = 0.02, curvature: Float = 0.15) {
        self.height = height
        self.width = width
        self.curvature = curvature
    }
}
